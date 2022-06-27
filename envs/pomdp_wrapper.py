import gym
import numpy as np


class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env_name, pomdp_type='nothing'):
        super().__init__(gym.make(env_name))
        self.pomdp_type = pomdp_type
        self.flicker_prob = 0.5
        self.random_noise_sigma = 0.1
        self.random_sensor_missing_prob = 0.1

        if self.pomdp_type == 'nothing':
            pass
        elif pomdp_type == 'remove_velocity':
            # Remove Velocity info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif pomdp_type == 'flickering':
            pass
        elif self.pomdp_type == 'random_noise':
            pass
        elif self.pomdp_type == 'random_sensor_missing':
            pass
        elif self.pomdp_type == 'remove_velocity_and_flickering':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'remove_velocity_and_random_noise':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'remove_velocity_and_random_sensor_missing':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'flickering_and_random_noise':
            pass
        elif self.pomdp_type == 'random_noise_and_random_sensor_missing':
            pass
        elif self.pomdp_type == 'random_sensor_missing_and_random_noise':
            pass
        else:
            raise ValueError("POMDP_type was not specified!")

    def observation(self, obs):
        # Single source of POMDP
        if self.pomdp_type == 'nothing':
            # print("Pre-Flatten: ", obs.shape)
            # check = obs.flatten()
            # print("Post-Flatten: ", check.shape)
            return obs.flatten()
            
        elif self.pomdp_type == 'remove_velocity':
            return obs.flatten()[self.remain_obs_idx]
        elif self.pomdp_type == 'flickering':
            # Note: flickering is equivalent to:
            #   flickering_and_random_sensor_missing, random_noise_and_flickering, random_sensor_missing_and_flickering
            if np.random.rand() <= self.flicker_prob:
                flat = obs.flatten()
                return np.zeros(flat.shape)
            else:
                return obs.flatten()
        elif self.pomdp_type == 'random_noise':
            return (obs + np.random.normal(0, self.random_noise_sigma, obs.shape))
        elif self.pomdp_type == 'random_sensor_missing':
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            return obs.flatten()
        # Multiple source of POMDP
        elif self.pomdp_type == 'remove_velocity_and_flickering':
            # Note: remove_velocity_and_flickering is equivalent to flickering_and_remove_velocity
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(new_obs.shape)
            else:
                return new_obs
        elif self.pomdp_type == 'remove_velocity_and_random_noise':
            # Note: remove_velocity_and_random_noise is equivalent to random_noise_and_remove_velocity
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Add random noise
            return (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.pomdp_type == 'remove_velocity_and_random_sensor_missing':
            # Note: remove_velocity_and_random_sensor_missing is equivalent to random_sensor_missing_and_remove_velocity
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
            return new_obs
        elif self.pomdp_type == 'flickering_and_random_noise':
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                new_obs = np.zeros(obs.shape)
            else:
                new_obs = obs
            # Add random noise
            return (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.pomdp_type == 'random_noise_and_random_sensor_missing':
            # Random noise
            new_obs = (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
            return new_obs
        elif self.pomdp_type == 'random_sensor_missing_and_random_noise':
            # Random sensor missing
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            # Random noise
            return (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
        else:
            raise ValueError("pomdp_type was not in ['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing']!")

    def _remove_velocity(self, env_name):
        # 1. Highway
        if env_name == "highway-v0":
            remain_obs_idx = np.arange(0,2)
        elif env_name == "roundabout-v0":
            remain_obs_idx = np.arange(0,2)
        elif env_name == "merge-v0":
            remain_obs_idx = np.arange(0,2)
        elif env_name == "racetrack-v0":
            remain_obs_idx = np.arange(0,2)
        elif env_name == "u-turn-v0":
            remain_obs_idx = np.arange(0,2)
        else:
            raise ValueError('POMDP for {} is not defined!'.format(env_name))
        
        # Redefine observation_space
        obs_low = np.array([-np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        obs_high = np.array([np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        observation_space = gym.spaces.Box(obs_low, obs_high)
        return remain_obs_idx, observation_space
