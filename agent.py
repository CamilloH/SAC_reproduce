import os 
import torch 
import torch.nn.functional as F 
import numpy as np
from networks import ActorNetwork, CriticNetwork, ReplayBuffer, Transition


class Agent():
    #TODO action number im not so sure about the input dims here
    def __init__(self, input_dims = [8], env = None, gamma = 0.99, n_actions = 12,
                  capacity = 500000, tau = 0.02, batch_size = 256):
        self.env_name = 'Hopper-v4'
        self.gamma = gamma 
        #target network update rate were doing the soft copy of the params
        self.tau = tau
        
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.target_entropy = - n_actions
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cpu')
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=0.0008)

        self.action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)

        self.actor = ActorNetwork(input_dims, n_actions,self.action_scale, self.action_bias)   
        self.Qnet_1 = CriticNetwork(input_dims, n_actions)
        self.Qnet_2 = CriticNetwork(input_dims, n_actions)
        self.Qnet_1_target = CriticNetwork(input_dims, n_actions)
        self.Qnet_2_target = CriticNetwork(input_dims, n_actions)
        
    def chooseAction(self, observation): 
        actions, _, _ = self.actor.sample_normal(observation)

        return actions.cpu().detach().numpy()
    
    def storeTransition(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def updateTargetNetworks(self, tau = None): 
        if tau is None:
            tau = self.tau
        #we want to update the target networks with the parameters of the primary networks
        #we will do this by soft copying the parameters of the primary networks to the target networks 
        #we will do this by taking the parameters of the primary networks and multiplying them by tau and adding them to the target networks
        #we will do this for both the actor and the critic netwo rks 
        #we will also set the target networks to eval mode 
        for target_param, param in zip(self.Qnet_1_target.parameters(), self.Qnet_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.Qnet_2_target.parameters(), self.Qnet_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def saveModel(self):
        torch.save(self.actor.state_dict(), self.env_name + "_weights.pth")

    def loadModel(self): 
        self.actor.load_state_dict(torch.load(self.env_name + "_weights.pth") )
    
    def learn_alpha(self, state):
        with torch.no_grad():
            _, log_probs, _ = self.actor.sample_normal(state)
            
        alpha_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

    def learn(self): 
        if len(self.memory) < self.batch_size:
            return
        #could put this into the smaple method of the replay buffer
        #TODO maybe data types can produce an error here
        #TODO maybe it needs to.device? 
        memory_batch = self.memory.sample(self.batch_size)
        memory_batch = Transition(*zip(*memory_batch))       
        state_batch = torch.stack(memory_batch.state)
        action_batch = torch.stack(memory_batch.action)
        reward_batch = torch.stack(memory_batch.reward).squeeze()
        next_state_batch = torch.stack(memory_batch.next_state)
        #we squeeze so the indexing is in the correct shapes 
        done_batch = torch.stack(memory_batch.done).squeeze()

####values for the critic networks ##### 

        #get the actions and log probs for the  current policy
        #state needs to be supplied by the environment 
        #policy sampled with next states 
        actions, log_probs, _ = self.actor.sample_normal(next_state_batch, reparametrize = False)
        #apparently the view(-1) call collapses the dimensions along the batches - so we get a 1D tensor which is what we want because we output scalars
        log_probs = log_probs.view(-1) 
        value_1 = self.Qnet_1(state_batch, action_batch).view(-1) # Q(state,action)
        with torch.no_grad():
            target_value1 = self.Qnet_1_target(next_state_batch, actions).view(-1) #Q_target(next_state, actor(next_state))
            target_value1[done_batch] = 0.0  #this should set terminal states to 0

        value_2 = self.Qnet_2(state_batch, action_batch).view(-1) # Q(state,action) 
        with torch.no_grad():
            target_value2 = self.Qnet_2_target(next_state_batch, actions).view(-1) #Q_target(next_state, actor(next_state))
            target_value2[done_batch] = 0.0  #this should set terminal states to 0

        #trick from the paper - double Q learning
        min_target = torch.min(target_value1, target_value2)
        done_batch_float = done_batch.float()       
        target = reward_batch + self.gamma*(1 - done_batch_float) * (min_target - self.alpha * log_probs)

        #losses        
        q1_loss = self.Qnet_1.criterion(value_1, target.detach())
        q2_loss = self.Qnet_2.criterion(value_2, target.detach())
##### values for the policy network ######
        #here we need the reparametrize for the gradient calculation
        actions, log_probs, _ = self.actor.sample_normal(next_state_batch, reparametrize = True)    
        q_actor1 = self.Qnet_1(state_batch, actions).view(-1)
        q_actor2 = self.Qnet_2(state_batch, actions).view(-1)
        min_q_actor = torch.min(q_actor1, q_actor2)
        actor_loss = (self.alpha * log_probs - min_q_actor ).mean()

       

        self.Qnet_1.optimizer.zero_grad()
        self.Qnet_2.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()

        q1_loss.backward()
        q2_loss.backward()
        actor_loss.backward()

        self.Qnet_1.optimizer.step()
        self.Qnet_2.optimizer.step()
        self.actor.optimizer.step()

        self.learn_alpha(state_batch)

        self.updateTargetNetworks(tau = 0.02)
