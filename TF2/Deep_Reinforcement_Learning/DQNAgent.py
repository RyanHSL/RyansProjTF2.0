from rl_trader import mlp

import ReplayBuffer
import MultiStockEnv
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # It corresponds to the number of inputs of the neuron that work respectively
        self.action_size = action_size # It corresponds to the number of outputs of the neuron that work respectively
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95 # Discount rate
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        # Choose a random value between 0 and 1 and compare it with epsilon
        # If it is smaller than epsilon then perform a random action using np.random.choice
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        # Otherwise perform a greedy action by grabbing all the Q values for the input state then action
        act_values = self.model.predict(state) # act_values.shape = (batch_size, number_of_outputs)

        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        # First check if replay buffer contains enough data
        if self.memory.size < batch_size:
            return
        # Sample a batch of data from the replay memory
        miniBatch = self.memory.sample_batch(batch_size)
        states = miniBatch['s']
        next_states = miniBatch['s2']
        action = miniBatch['a']
        rewards = miniBatch['r']
        done = miniBatch['d']
        # Calculate the tentative target: Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis = 1)
        # The value of terminal states is zero (In fact there is no terminal state in stock market)
        # So set the target to be the reward only
        target[done] = rewards[done]
        """
        With the Keras API, the target (usually) must have the same shape as the prediction
        However, I only need to update the network for the actions which were actually taken
        I can accomplish this by setting the target to be equal to the prediction for all values
        Then only change the targets for the actions taken
        Q(s, a)
        Currently the targets are just a 1D array of length batch_size but the model predictions
        are a 2D array of batch_size by number of actions
        For each sample, there must be a target for each action even if that action was not the 
        one taken by the agent.
        In order to do this I need to use the model to make a prediction for each state and each
        action
        """
        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), action] = target
        # Run one training step
        self.model.train_on_batch(states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)