import gym
import time
import torch
import datetime
import itertools
import pathlib
import numpy as np
from agent import SAC
from collections import deque
from torch.utils.tensorboard import SummaryWriter

seed = 0
eval = True
episodes = 100
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make('Pendulum-v1')

agent = SAC(env.observation_space.shape[0], env.action_space)
agent.load_checkpoint('checkpoints/sac_checkpoint_Pendulum_', True)
writer = SummaryWriter('runs/{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'Pendulum'))

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