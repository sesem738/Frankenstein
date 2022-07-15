from statistics import mode
import gym
import time
import torch
import datetime
import itertools
import pathlib

# Suppress DeprecationWarning before importing highway_env
import warnings
warnings.simplefilter("ignore")

import highway_env
import numpy as np
from agent import SAC
from collections import deque
import sys
from envs.pomdp_wrapper import POMDPWrapper
from torch.utils.tensorboard import SummaryWriter


seed = 0
eval = True
episodes = 100
torch.manual_seed(seed)
np.random.seed(seed)

env_name = "racetrack-v0"
env = POMDPWrapper(env_name, 'nothing')

agent = SAC(np.prod(env.observation_space.shape), env.action_space, model="GTrXL")
agent.load_checkpoint(sys.argv[1], True)

scores_deque = deque(maxlen=100)
scores = []

for i_episode in range(episodes + 1):
    
    state = env.reset()
    score = 0                    
    time_start = time.time()
    
    while True:

        env.render()
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        score += reward 
        state = next_state

        if done:
            break
            
    s = (int)(time.time() - time_start)
    
    scores_deque.append(score)
    scores.append(score)    
    
    print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}'\
                .format(i_episode, np.mean(scores_deque), score, s//3600, s%3600//60, s%60)) 

env.close()