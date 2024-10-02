import math
import random
import time

import gymnasium as gym 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

from agent import Agent


env = gym.make("Hopper-v4", render_mode = 'human')
observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = Agent(env = env, input_dims=observation_size, n_actions=action_size)

mps_device = torch.device("cpu") 
n_games = 5000
score_history = []

use_trained = True
save_model = False
continue_training = False #the transitions are not saved though - maybe stupid 

if use_trained: 
    agent.loadModel()
    n_games = 5

for i in range(n_games): 
    state = torch.tensor(env.reset()[0], dtype=torch.float).to(mps_device)
    done = False 
    score = 0 
    highscore = 0
    cheetah_count = 0 
    while not done: 
        action = agent.chooseAction(state)
        #as the 4th return value they return flase?
        next_state, reward, done, _ , info  = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float).to(mps_device)
        action = torch.tensor(action, dtype=torch.float).to(mps_device)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0).to(mps_device)
        done = torch.tensor(done, dtype = torch.bool).unsqueeze(0).to(mps_device)

        score += reward
        agent.storeTransition(state, action, next_state, reward, done)

        if (not use_trained) or continue_training:
            agent.learn() 
        state = next_state
        # cheetah_count+= 1
        # if cheetah_count > 600:
        #     done = True
  
    score_history.append(score.cpu().numpy())
    avg_score = np.mean(score_history[-100:])

    if  avg_score > highscore and save_model:
        highscore = avg_score
        agent.saveModel()

    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

plt.plot(score_history)





