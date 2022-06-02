import gym
import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
from agent import SAC
from pathlib import Path
from memory import ReplayMemory
from metrics import MetricLogger

def main():

    eval = True
    seed = 1
    episodes = 100_000
    batch_size = 256
    replay_size = 100_00
    burnin = 2500
    
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)
    checkpoint = Path('online_25.chkpt')

    # Environment    
    env = gym.make('Pendulum-v1')
    env.seed(seed)
    env.action_space.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Agent
    agent = SAC(env.observation_space.shape[0],env.action_space)

    # Memory 
    memory = ReplayMemory(replay_size, seed)
    
    # Metrics
    logger = MetricLogger(save_dir)

    updates = 0
    total_steps = 0
    episode_steps = 0
    episode_reward = 0

    # Training Loop
    for ep in tqdm(range(1, episodes + 1), ascii==True, unit ='episodes'):
        
        state = env.reset()

        try:

            while True:
                if total_steps < burnin:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action)
                done_bool = float(done) if episode_steps < env._max_episode_steps else 0

                memory.push(state, action, reward, next_state, done_bool)

                state = next_state
                
                total_steps += 1
                episode_reward += reward

                # Train agent after collecting sufficient data
                if len(memory) > batch_size:
                    # Update parameters of all the networks
                    _,_,_,_,_ = agent.update_parameters(memory, batch_size, updates)
                    updates += 1
                
                logger.log_step(episode_reward)
                
                if done:
                    episode_reward = 0
                    episode_steps = 0
                    break
        
        finally:
            logger.log_episode(episode_steps)

        if ep % 1 == 0:
            logger.record(
                episode=ep,
                step=total_steps
            )

    # if epi % 10 == 0 and eval is True:
    #     avg_reward = 0.
    #     eval_episodes = 10
    #     for _  in range(eval_episodes):
    #         state = env.reset()
    #         episode_reward = 0
    #         done = False
    #         while not done:
    #             action = agent.select_action(state, evaluate=True)

    #             next_state, reward, done, _ = env.step(action)
    #             episode_reward += reward

    #             state = next_state
    #         avg_reward += episode_reward
    #     avg_reward /= eval_episodes



    

if __name__ == '__main__':
    main()