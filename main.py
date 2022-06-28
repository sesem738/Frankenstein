import gym
import torch
import argparse
import datetime

# Suppress DeprecationWarning before importing highway_env
import warnings
warnings.simplefilter("ignore")

import highway_env
import itertools
import numpy as np
from agent import SAC
from memory import ReplayMemory
from envs.pomdp_wrapper import POMDPWrapper
from torch.utils.tensorboard import SummaryWriter

updates_per_step = 1
eval = True
seed = 0
batch_size = 256
num_steps = 10000001
start_steps = 1000
replay_size = 100000


# Environment
env_name = "racetrack-v0"
env = POMDPWrapper(env_name, 'nothing')
env.action_space.seed(1)

torch.manual_seed(1)
np.random.seed(1)

# Agent
agent = SAC(np.prod(env.observation_space.shape), env.action_space)


#Tesnorboard
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(current_time, env_name,
                                                             "Gaussian", "autotune"))

# Memory
memory = ReplayMemory(replay_size, 1)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > batch_size:
            # Number of updates per step in environment
            for i in range(updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        mask = 1 if episode_steps == 5000 else float(not done) # ******COME BACK TO THIS********

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and eval is True:
        avg_reward = 0.
        episodes = 10
        if i_episode % 10 == 0:
            agent.save_checkpoint(f'{env_name}_{current_time}', str(i_episode))

        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

