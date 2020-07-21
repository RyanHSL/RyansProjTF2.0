import numpy as np

"""The experience replay memory"""
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf1 = np.zeros([size, obs_dim], dtype=np.float32) # A buffer to store the current state
        self.obs1_buf2 = np.zeros([size, obs_dim], dtype=np.float32)# A buffer to store the next state
        self.acts_buf = np.zeros(size, dtype=np.uint8)# A buffer to store actions. int(0:26) unint8
        self.rews_buf = np.zeros(size, dtype=np.float32)# A buffer to store the reward. float32
        self.done_buf = np.zeros(size, dtype=np.uint8)# A buffer to store the done flag. unint8
        self.ptr, self.size, self.max_size = 0, 0, size# Initialize the pointer and the size to be 0 and the max size to be size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf1[self.ptr] = obs
        self.obs1_buf2[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size # Make it circular
        self.size = min(self.size + 1, self.max_size) # The size cannot be greater than the max size

    # This function chooses random indices from 0 up to the size of the buffer
    # Then returns a dictionary containing the state's actions, rewards and so forth indexed by those indices
    def sample_batch(self, batch_size=32):
        indxs = np.random.randint(0, self.size, size=batch_size)

        return dict(s = self.obs1_buf1[indxs],
                    s2 = self.obs1_buf2[indxs],
                    a = self.acts_buf[indxs],
                    r = self.rews_buf[indxs],
                    d = self.done_buf[indxs])