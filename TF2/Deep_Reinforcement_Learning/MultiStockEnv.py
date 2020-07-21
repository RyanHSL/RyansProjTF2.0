import numpy as np
import itertools

class MultiStockEnv:
    """
    A 3-stock trading environment.
    State: vector of size 7 (n_stock * 2 + 1):
        shares of stock 1 owned
        shares of stock 2 owned
        shares of stock 3 owned
        price of stock 1 (using daily close price)
        price of stock 2
        price of stock 3
        cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (3^3) possibilities:
        for each stock it can:
        0 = sell
        1 = hold
        2 = buy
    """
    def __init__(self, data, initial_investment = 10000):
        # Data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # Instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3**self.n_stock)

        """
        action permutations
        returns a nested list with elements like:
        [0, 0, 0]
        [0, 0, 1]
        [0, 0, 2]
        [0, 1, 0]
        etc
        0 = sell
        1 = hold
        2 = buy
        """
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # Calculate size of state
        self.state_dim = self.n_stock * 2 + 1
        self.reset()

    def reset(self):
        """
        Initialize a few more attributes and returns the initial state
        """
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment

        return self._get_obs()

    def step(self, action):
        """
         It performs an action in the environment and returns the next state and reward among other things
        """
        # Check the action that passed in exists in action space
        assert action in self.action_space
        # Get current value before performing the action
        pre_val = self._get_val()
        # Update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]
        # Perform the trade
        self._trade(action)
        # Get the new value after taking the action
        cur_val = self._get_val()
        # Reward is the increase in portfolio value
        reward = cur_val - pre_val
        # Done if I have run out of money
        done = self.cur_step == self.n_step - 1
        # Store the current value of the portfolio here
        info = {'cur_val': cur_val}
        # Conform to the Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """
        This function returns the state
        State: vector of size 7 (n_stock * 2 + 1):
            shares of stock 1 owned
            shares of stock 2 owned
            shares of stock 3 owned
            price of stock 1 (using daily close price)
            price of stock 2
            price of stock 3
            cash owned (can be used to purchase more stocks)
        """
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand

        return obs

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        """
        Index the action we want to perform
        0 = sell
        1 = hold
        2 = buy
        """
        action_vec = self.action_list[action]
        # Determine which stocks to buy or sell
        # Define two lists that store indices of stocks I want to sell or buy
        sell_index, buy_index = [], []
        # Loop through all the indices and actions in the enumerate(action_vec)
        for i, a in enumerate(action_vec):
            # append the index to the sell list if the action is sell
            if a == 0:
                sell_index.append(i)
            # append the index to the buy list if the action is buy
            if a == 2:
                buy_index.append(i)
        # Sell any stocks I want to sell
        # Then buy any stocks I want to buy
        if sell_index:
            # Note: to simplify the problem, when I sell, I will sell all shares of that stock
            # Loop through all sell_index
            for i in sell_index:
                # update the cash in hand by adding that stock price times that number of shares
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                # Change the stock owned to be 0
                self.stock_owned[i] = 0
        # Do the same thing for buying
        if buy_index:
            # Note: when buying, I will loop through each stock we want to buy,
            # and buy one share at a time until I run out of cash
            # Define a can_buy flag
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand >= self.stock_price[i]:
                        # Buy one share if cash in hand is greater than that stock price
                        self.stock_owned[i] += 1
                        # Deduct the share price from cash in hand
                        self.cash_in_hand -= self.stock_price[i]
                    # Else change the can_buy flag to False
                    else:
                        can_buy = False
