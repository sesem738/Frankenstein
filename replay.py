import numpy as np

# Intially history len must be small if memory buffer is sparsely populated or actions are only ever randomly sampled
# Could implement scaling his_len as a workaround


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
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.done[ind]
		)

    def prior_samples(self, batch_size, his_len):
        ind = np.random.randint(his_len, len(self), size=batch_size)
        # History
        obs = np.zeros([batch_size, his_len, self.state_dim])
        actions = np.zeros([batch_size, his_len, self.action_dim])
        next_obs = np.zeros([batch_size, his_len, self.state_dim])
        rewards = np.zeros([batch_size, his_len, 1])
        done = np.zeros([batch_size, his_len, 1])
        # his_obs_len = his_len * np.ones(batch_size)

        for i, id in enumerate(ind):
            
            start_id, id = self._get_valid_sequence(id, his_len)
            obs[i] = self.state[start_id:id]
            actions[i] = self.action[start_id:id]
            next_obs[i] = self.next_state[start_id:id]
            rewards[i] = self.reward[start_id:id]
            done[i] = self.done[start_id:id]

        return (obs, actions, next_obs, rewards, done)

    def _get_valid_sequence(self, id:int, his_len:int):
        """Checks if the sequence of transitions is valid, i.e., has and terminal transitions.
        Returns id's for a valid sequence"""
        start_id = id-his_len
        
        not_valid = True
        while not_valid:

            # Check sequence for termination
            if len(np.where(self.done[start_id:id] == 1)[0]) != 0:
                temp = start_id + (np.where(self.done[start_id:id] == 1)[0][-1]) + 1
                if (temp + his_len) >= len(self):
                    start_id = np.random.randint(0, len(self)-his_len,1)[0]
                else:
                    start_id = temp
                id = start_id + his_len
            
            if (len(np.where(self.done[start_id:id] == 1)[0]) == 0):
                not_valid = False
        
        return start_id, id
        
    
    def __len__(self):
        return self.size
