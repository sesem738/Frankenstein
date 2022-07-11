import os
import pickle
import torch
import numpy as np

class ReplayMemory:
    def __init__(self, max_size, seed, obs_dim, act_dim):
        np.random.seed(seed)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        """
        :param batch_size:
        :param max_hist_len: the length of experiences before current experience
        :return:
        """
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size)
        # History
        if max_hist_len == 0:
            hist_obs = np.zeros([batch_size, 1, self.obs_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_obs_len = np.zeros(batch_size)
            hist_obs2_len = np.zeros(batch_size)
        else:
            hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_obs_len = max_hist_len * np.ones(batch_size)
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_obs2_len = max_hist_len * np.ones(batch_size)

            # Extract history experiences before sampled index
            for i, id in enumerate(idxs):
                hist_start_id = id - max_hist_len
                if hist_start_id < 0:
                    hist_start_id = 0
                # If exist done before the last experience (not including the done in id), start from the index next to the done.
                if len(np.where(self.done_buf[hist_start_id:id] == 1)[0]) != 0:
                    hist_start_id = hist_start_id + (np.where(self.done_buf[hist_start_id:id] == 1)[0][-1]) + 1
                hist_seg_len = id - hist_start_id
                hist_obs_len[i] = hist_seg_len
                hist_obs[i] = self.obs_buf[hist_start_id:id]
                # If the first experience of an episode is sampled, the hist lengths are different for obs and obs2.
                if hist_seg_len == 0:
                    hist_obs2_len[i] = 1
                else:
                    hist_obs2_len[i] = hist_seg_len
                hist_obs2[i] = self.obs2_buf[hist_start_id:id]

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs=hist_obs,
                     hist_obs2=hist_obs2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def __len__(self):
        return self.size
    
    def __repr__(self):
        return f"Buffer of max_size {self.max_size} filled till {self.size}"
