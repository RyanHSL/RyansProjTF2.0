# If using import it will import modules instead of class objects. Modules are not callable.
from ReplayBuffer import ReplayBuffer
from MultiStockEnv import MultiStockEnv
from DQNAgent import DQNAgent

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import itertools
import argparse
import re
import os
import pickle

# Define a function to get the stock data. (AAPL, MSI, SBUX)
def get_data():
    # Return a T x 3 list of stock prices
    # each row is a different stcck
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX
    df = pd.read_csv("aapl_msi_sbux.csv")

    return df.values

# Define a get scaler function which takes in an environment object.
# In order to get the right parameter for a scaler, I need to play a episode randomly and store each of the state I encounter
# There is no need to have an agent since such an agent will not be trained anyway
# It can be run multiple times to be more accurate
def get_scaler(env):
    # Return sklearn scaler object to scale the states
    # Note: replay buffer should also be populated here
    # Initialize an empty states list
    states = []
    # Loop through all items in env
    for _ in range(env.n_step):
        # Pass a randomly chosen action from the action_space to the action object
        action = np.random.choice(env.action_space)
        # Use env.step(action) to get the state, reward, done, and info
        state, reward, done, info = env.step(action)
        # Append the state to the states list
        states.append(state)
        # Break the loop if the done flag is true
        if done:
            break
    # Instantiate the StandardScaler object
    scaler = StandardScaler()
    # Use the scaler object to fit the states list and return it
    scaler.fit(states)

    return scaler
# Define a make_dir function to make a directory if there is not one
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return
# # Define a mlp function to do the Multi-layer Perceptron
# def mlp(input_dim, n_action, n_hidden_layers = 1, hidden_dim = 32):
#     # Input layer
#     i = Input(shape=(input_dim, ))
#     x = i
#     # Hidden layer
#     for _ in range(n_hidden_layers):
#         # Insert a dense layer
#         x = Dense(hidden_dim, activation="relu")(x)
#     # Final layer (output size: n_action)
#     # Since I am doing regression, the output does not have an activation function
#     x = Dense(n_action)(x)
#     # Make the model and compile it and finally return it
#     model = Model(i, x)
#
#     model.compile(optimizer=Adam,
#                   loss="mse")
#
#     return model

def play_one_episode(agent, env, is_train):
    # Note: after transforming states are already 1xD
    # Reset the state then transform it and set the done flag to be False
    state = env.reset()
    state = scaler.transform([state]) # states.shape = (N, 7)
    done = False
    while not done:
        # Get the action through agent.act(state)
        action = agent.act(state)
        # Get the next_state, reward, done, info through env.step(action)
        next_state, reward, done, info = env.step(action)
        # Use scaler to transform the next_state
        next_state = scaler.transform([next_state])
        if is_train == "train":
            # Add the latest transition to our replay buffer
            # Update the replay memory in agent with state, action, reward, next_state and done
            agent.update_replay_memory(state, action, reward, next_state, done)
            # Replay the agent to run one step gradient descent
            agent.replay()
        # Set state to be next_state for the next iteration of the loop
        state = next_state
    # return the current value of the portfolio
    return info["cur_val"]

if __name__ == '__main__':
    # Config
    models_folder = "rl_trader_models"
    rewards_folder = "rl_trader_rewards"
    num_episodes = 2000
    batch_size = 32
    initial_investment = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, required=True, help='either "train" or "test"')
    args = parser.parse_args()

    make_dir(models_folder)
    make_dir(rewards_folder)

    data = get_data()
    n_timesteps, n_stocks = data.shape

    n_train = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # Store the final value of the portfolio (end of the episode)
    portfolio_value = []

    if args.mode == "test":
        # Then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # Make sure epsilon is not 1
        # No need to run multiple episode if epsilon = 0, it's is deterministic
        agent.epsilon = 0.01
        # Load trained weights
        agent.load(f"{models_folder}/dqn.h5")

    # Play the game run_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, args.mode)
        dt = datetime.now() - t0
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration:{dt}")
        portfolio_value.append(val)

    # Save the weights when I am done
    if args.mode == "train":
        # Save the DQN
        agent.save(f"{models_folder}/dqn.h5")

        # Save the scaler
        with open(f"{models_folder}/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    # Save portfolio value for each episode
    np.save(f"{rewards_folder}/{args.mode}.npy", portfolio_value)
