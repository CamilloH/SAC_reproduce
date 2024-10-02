import random
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from collections import namedtuple, deque



#This is a namedtupel I can now create an object Transition with the attributes state, action, next_state, reward
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer(object): 

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    # *args is used to pass a variable number of arguments to a function
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
#we will initialize 4 Q networks 2 primary and 2 targets 
class CriticNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(CriticNetwork, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Linear(n_actions + n_observations ,400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            #we only want to output one action value Q(s,a)
            nn.Linear(300, 1)
        )
        #we need an optimizer to define how the calculated gradients are applied to the weights, we will also need a loss function which comes later 
        #we define this optimizer within the class itself because were using 2 Qnets and 1 Policy net - so we need to be able to call them individually 
        self.optimizer = optim.Adam(self.parameters(), lr=0.0006)
        self.criterion = torch.nn.MSELoss()
        self.device = 'cpu'
        self.to(self.device)
        
    def forward(self, state, actions):
        # Define the forward pass this should flatten actions and state so we have all the info we need 
        # - I think the exact dimensions should not matter 
        action_value = self.network(torch.cat([state, actions], dim = 1))
        return action_value 
    
class ActorNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, action_scale, action_bias):
        super(ActorNetwork, self).__init__()
        self.n_observations = n_observations
        #these are actually continuous but we might have multiple components to the action - like multiple joints  
        self.n_actions = n_actions
        self.reparam_noise = 1e-6
        self.action_bias = action_bias.to('cpu')
        self.action_scale = action_scale.to('cpu')
        
        self.fc1 = nn.Linear(n_observations, 400)
        self.fc2 = nn.Linear(400, 300)
        self. relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.mean = nn.Linear(300, n_actions)
        self.std = nn.Linear(300, n_actions)
        
        self.optimizer= optim.Adam(self.parameters(), lr=0.0003)
        self.device = 'cpu'
        self.to(self.device)
    
    def forward(self, state):
        output = self.fc1(state)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)

        mean = self.mean(output)
        std = self.std(output)
        std = torch.tanh(std)
        std = -5 + 0.5 * (2 - (-5)) *(std +1)
        #this is apparently approximately how they do it in the paper 

        return mean, std 


    def sample_normal(self, state, reparametrize = True): 
        #I dont think this is really the log std but rather the tanh std - but i guess thats how they do it 
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        action_probabilities = distributions.Normal(mean, std)
        #this is the reparametrization trick - we sample from a normal distribution and then we multiply it with the std and add the mean
        # so actions = mean + std * epsilon where epsilon is sampled from a normal distribution
        if reparametrize:
            actions = action_probabilities.rsample()
        else:
            actions = action_probabilities.sample() 
        #here we could multiply it with a max_actions number which is the max action we can take if its not between -1,1
        actions_tanh = torch.tanh(actions) #y_t
        action = actions_tanh * self.action_scale + self.action_bias
        #we need this to calculate the loss function later (look equation @spinningUp)
        #we take the log prob of the actions - bUT NOT THE TANH ACTIONS 
        log_probs = action_probabilities.log_prob(actions)
        #some substractions idk why - corrects for the tanh nonlinearity
        log_probs -= torch.log(self.action_scale * (1 - actions_tanh.pow(2)) + self.reparam_noise)
        log_probs[torch.isnan(log_probs)] = 0
        log_probs = log_probs.sum()
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_probs, mean
