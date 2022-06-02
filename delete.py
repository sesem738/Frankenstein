import random
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from agent import Agent
from metrics import MetricLogger
from playground import PlayGround


N_ACTIONS               = 6
INPUT_SHAPE             = (12,150,150)
REPLAY_SIZE             = 100000
BATCH_SIZE              = 256
ALPHA                   = 0.00036
EPSILON                 = 1.0
EPSILON_DACAY           = 0.9999995 
EPISODES                = 100000
BURNIN                  = 5000
SAVE_EVERY              = 15000
UPDATE_EVERY            = 2000
TRAIN_EVERY             = 10
DISCOUNT                = 0.99


def main():

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)
    checkpoint = Path('online_25.chkpt')
    # checkpoint = None
    random.seed(1)
    np.random.seed(1)

    # Instantiate Environments
    tesla = Agent(alpha=ALPHA, gamma=DISCOUNT, epsilon=EPSILON, epsilon_decay=EPSILON_DACAY,\
                max_size=REPLAY_SIZE, burnin=BURNIN,input_shape=INPUT_SHAPE, n_actions=N_ACTIONS,\
                batch_size=BATCH_SIZE, save_every=SAVE_EVERY, update_every=UPDATE_EVERY,\
                train_every=TRAIN_EVERY, save_dir=save_dir, checkpoint=checkpoint)
    env = PlayGround()
    logger = MetricLogger(save_dir=save_dir)


    for epi in tqdm(range(1,EPISODES + 1), ascii=True, unit='episodes'):
        
        state = env.reset()

        try:
        
            while True:
                action = tesla.act(state)
                next_state, reward, done, _ = env.step(action)
                tesla.cache(state, action, reward, next_state, done)
                
                q,loss = tesla.learn()
                logger.log_step(reward, loss, q)

                state = next_state
                # print(reward, loss)

                if done:
                    break
        finally:
            logger.log_episode(env.tick_count)
            env.destroy()


        if epi % 20 == 0:
            logger.record(
                episode=epi,
                epsilon=tesla.epsilon,
                step=tesla.curr_step
            )
        
        if env.to_quit():
            env.set_synchronous_mode(False)
            break


if __name__ == '__main__':
    main()
