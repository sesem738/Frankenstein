import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, capacity):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = capacity
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

    def prior_samples(self, batch_size, his_len):
        ind = np.random.randint(his_len, self.size, size=batch_size)
        # History
        obs = np.zeros([batch_size, his_len, self.state_dim])
        actions = np.zeros([batch_size, his_len, self.action_dim])
        next_obs = np.zeros([batch_size, his_len, self.state_dim])
        rewards = np.zeros([batch_size, his_len, 1])
        done = np.zeros([batch_size, his_len, 1])
        his_obs_len = his_len * np.ones(batch_size)

        for i, id in enumerate(ind):
            start_id = id-his_len
            if start_id < 0:
                start_id = 0
            if len(np.where(self.done[start_id:id] == 1)[0]) != 0:
                    start_id = start_id + (np.where(self.done[start_id:id] == 1)[0][-1]) + 1
            obs[i] = self.state[start_id:id]
            actions[i] = self.action[start_id:id]
            next_obs[i] = self.next_state[start_id:id]
            rewards[i] = self.reward[start_id:id]
            done[i] = self.done[start_id:id]

        return (
            torch.FloatTensor(obs).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(next_obs).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(done).to(self.device)
        )
