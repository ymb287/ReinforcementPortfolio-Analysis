import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, stock_data, initial_value=1000, trading_fee=0.001, reward_function=None):
        super(TradingEnv, self).__init__()
        
        # Store the stock data and initialize environment parameters
        self.stock_data = stock_data
        self.num_assets = stock_data.shape[1] + 1  # Number of assets (stocks + cash)
        self.num_stocks = stock_data.shape[1]  # Number of stocks only
        self.trading_fee = trading_fee  # Trading fee percentage
        self.initial_wealth = initial_value  # Initial portfolio value (corrected typo)
        self.portfolio_weights_end = np.array([1] + [0] * self.num_stocks, dtype=np.float32)
        self.reward_function = reward_function  # Reward function as a parameter
        self.dsr_A = 0
        self.dsr_B = 0

        # Initialize arrays to track returns and squared returns for each stock
        self.recent_returns = [[] for _ in range(self.num_stocks)]

        # Define action space: Actions range from 0 to 1 for each asset ####### I CHANGED FROM -1 to 0
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        
        # Define observation space: Asset prices, portfolio weights, and portfolio value
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(self.num_assets * 2 +5,),  # Prices, weights, and portfolio value
            dtype=np.float32
        )
        
        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        self.wealth_end = self.initial_wealth  # Initial wealth in cash
        self.current_step = 0  # Reset the current step to the beginning
        self.portfolio_weights_end = np.array([1] + [0] * self.num_stocks, dtype=np.float32)  # Start with all cash
        
        # Get the initial asset prices (including cash price which is always 1)
        self.current_prices_end = np.concatenate(([1.0], self.stock_data[self.current_step])).astype(np.float32)
        
        # Construct the initial state: balance, stock prices, and portfolio weights
        volatilities = np.zeros(self.num_stocks)  # Volatility is 0 at the beginning
        self.state = np.concatenate((self.current_prices_end, self.portfolio_weights_end, volatilities)).astype(np.float32)

        return self.state, {}

    def diff_sharpe_reward(self, wealth_previous, wealth_current, eta=0.01):
        dsr_A_pre = self.dsr_A
        dsr_B_pre = self.dsr_B
        return_portfolio = wealth_current / wealth_previous - 1
        return_squared = return_portfolio ** 2
        delta_A = (return_portfolio - dsr_A_pre)
        delta_B = (return_squared - dsr_B_pre)

        self.dsr_A = dsr_A_pre + eta*delta_A
        self.dsr_B = dsr_B_pre + eta*delta_B

        denom = dsr_B_pre - dsr_A_pre**2
        reward = 0 if denom == 0 else (dsr_B_pre * delta_A - 0.5 * dsr_A_pre * delta_B) / (denom**1.5)       
        return reward

    def calculate_stock_volatility(self):
        """Calculate 20-day historical volatility (standard deviation) for each stock."""
        volatilities = np.zeros(self.num_stocks)
        for stock_idx in range(self.num_stocks):
            returns = self.recent_returns[stock_idx]
            if len(returns) > 1: 
                volatilities[stock_idx] = np.std(returns, ddof=1)
            else:
                volatilities[stock_idx] = 0 
        return volatilities

    def step(self, action):
        """Execute one step in the environment with the given action."""
        wealth_previous_end = self.wealth_end
        stock_weights_previous_end = self.portfolio_weights_end[1:]
           
        # Update the portfolio weights according to action
        portfolio_weights = action
        stock_weights = portfolio_weights[1:]
        
        # Calculate trade volume and apply trading fee
        trade_volume = np.abs(stock_weights - stock_weights_previous_end)
        trading_cost = np.sum(trade_volume) * self.trading_fee

        # Deduct trading cost from wealth
        wealth = wealth_previous_end * (1 - trading_cost)

        # Get previous asset prices (including cash price which is always 1)
        price_previous_end = np.concatenate(([1.0], self.stock_data[self.current_step]))

        # Proceed to the next step
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1 
        self.current_prices_end = np.concatenate(([1.0], self.stock_data[self.current_step]))
        price_relative = self.current_prices_end / price_previous_end      
        change_weights = price_relative * portfolio_weights
        norming_weights = np.dot(price_relative, portfolio_weights)
        self.portfolio_weights_end = change_weights / norming_weights
        self.wealth_end = wealth * norming_weights

        # Calculate returns for each stock (ignoring cash)
        returns = (self.current_prices_end[1:] / price_previous_end[1:]) - 1

        # Update the buffer with the latest returns for each stock
        for i in range(self.num_stocks):
            self.recent_returns[i].append(returns[i])
            if len(self.recent_returns[i]) > 20:
                self.recent_returns[i].pop(0)


        # Calculate volatilities for each stock
        volatilities = self.calculate_stock_volatility()

        # Calculate reward using the reward function
        if self.reward_function == 'diff_sharpe_reward':
            wealth_end = int(self.wealth_end)
            reward = self.diff_sharpe_reward(wealth_previous_end, wealth_end)
        elif self.reward_function == 'portfolio_value':
            reward = self.wealth_end - wealth_previous_end
        else:
            raise ValueError('Reward function not recognized')

        # Update the state
        self.state = np.concatenate((self.current_prices_end, self.portfolio_weights_end, volatilities)).astype(np.float32)
        
        # Return step information
        return self.state, reward, done, False, {}

    def render(self):
        if self.current_step % 50 == 0:
            print(f'Step: {self.current_step}')
            print(f'Portfolio Value: {self.wealth_end:.2f}')
            
            # Print current prices and portfolio weights
            formatted_prices = [f'{price:.2f}' for price in self.current_prices_end]
            formatted_weights = [f'{weight:.2f}' for weight in self.portfolio_weights_end]

            print(f'Current Prices: {formatted_prices}')
            print(f'Current Weights: {formatted_weights}')


    def get_portfolio_value(self):
        # Calculate the current portfolio value (cash + value of stocks)
        return self.wealth_end

    def get_portfolio_weights(self):
        # Return the current portfolio weights
        return self.portfolio_weights_end

