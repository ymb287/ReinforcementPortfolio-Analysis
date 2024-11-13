import pandas as pd
import numpy as np
from arch import arch_model
import yfinance as yf
from sklearn.model_selection import train_test_split
import pickle
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm 
import os


# Step 1: Get the list of S&P 500 stocks from Wikipedia
def get_sp500_stocks():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    
    tickers = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text.strip()
        tickers.append(ticker)
    
    return tickers

# Step 2: Fetch stock data and calculate average daily trading volume
def fetch_stock_data(ticker, start_date="2000-01-01", end_date="2023-12-31"):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)
    
    # Filter to get the required columns: Adj Close and Volume
    data = stock_data['Volume'].dropna()
 
    # Calculate the average daily trading volume (ADTV) over the period
    adtv = data.mean()
    min_volume = data.min()
    
    # Return stock data with additional metrics (ADTV and Market Cap)
    return adtv, min_volume, stock_data


def filter_liquid_stocks_2(sp500_tickers, min_adtv=1_000_000, min_volume_threshold=500_000):
    liquid_stocks = []
    total_stocks = len(sp500_tickers)  # Total number of S&P 500 stocks
    passed_criteria = 0  # Counter for stocks that meet the criteria
    failed_criteria = 0  # Counter for stocks that do not meet the criteria
    failed_fetch = 0
    missing_data = 0

    for ticker in tqdm(sp500_tickers, desc="Processing stocks", unit="stock"):
        try:
            adtv, min_volume, stock_data = fetch_stock_data(ticker)


            # Check if the stock meets ADTV, market cap, and minimum trading volume criteria
            if adtv >= min_adtv and min_volume >= min_volume_threshold:
                if len(stock_data['Close']) < 6000:
                    missing_data += 1
                    print(f"Missing data for {ticker}")
                else:
                    liquid_stocks.append((ticker, adtv, min_volume))
                    passed_criteria += 1  # Increment counter for passed stocks
            else:
                failed_criteria += 1  # Increment counter for failed stocks
        except Exception as e:
            failed_fetch += 1  # Increment counter for failed stocks (error in fetching data)
            print(f"Error fetching data for {ticker}: {e}")

    
    # Summary
    print(f"\nTotal Stocks Processed: {total_stocks}")
    print(f"Stocks that met the criteria: {passed_criteria}")
    print(f"Stocks with missing data: {missing_data}")
    print(f"Stocks that did not meet the criteria or had errors: {failed_criteria}")
    print(f"Stocks with fetch errors: {failed_fetch}")

    # Return DataFrame of liquid stocks
    return pd.DataFrame(liquid_stocks, columns=['Ticker', 'ADTV', 'Min Volume'])



# Step 3: Filter liquid stocks and classify stocks
def filter_liquid_stocks(sp500_tickers, min_adtv=1_000_000, min_volume_threshold=500_000):
    liquid_stocks = []
    non_callable_stocks = []
    non_liquid_stocks = []
    not_long_enough_stocks = []

    for ticker in tqdm(sp500_tickers, desc="Processing stocks", unit="stock"):
        try:
            adtv, min_volume, stock_data = fetch_stock_data(ticker)

            # Check if the stock meets ADTV, market cap, and minimum trading volume criteria
            if adtv >= min_adtv and min_volume >= min_volume_threshold:
                if len(stock_data['Close']) < 6000:
                    not_long_enough_stocks.append(ticker)
                else:
                    liquid_stocks.append(ticker)
            else:
                non_liquid_stocks.append(ticker)
        except Exception as e:
            non_callable_stocks.append(ticker)

    # Return DataFrame of stocks categorized by criteria
    return {
        'Scraped Stocks': sp500_tickers,
        'Non-callable Stocks': non_callable_stocks,
        'Non-liquid Stocks': non_liquid_stocks,
        'Not Long Enough Stocks': not_long_enough_stocks,
        'Liquid Stocks': liquid_stocks
    }


############################################

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def create_log_return(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def save_data_csv(data, filename):
    data.to_csv(filename)

def load_data_csv(filename): 
    data = pd.read_csv(filename, index_col=0)
    return data

def stock_train_test_split(returns, split_date="2020-01-01"):
    # Ensure the index is a datetime index if not already
    if not pd.api.types.is_datetime64_any_dtype(returns.index):
        returns.index = pd.to_datetime(returns.index)

    # Split the data so that test_data starts on the split_date
    train_data = returns.loc[:split_date]  # Training data up to 2020-12-31
    test_data = returns.loc[split_date:]   # Test data starting from 2021-01-01

    return train_data, test_data


def run_garch(data, p, q, dist = 'normal', save = False):
    percent_log_return = data*100
    garch = arch_model(percent_log_return, p=p, q=q, dist=dist)
    res = garch.fit(disp="off")
    if save:
        save_garch_model([garch, res],  "saved_models/garch_and_fit_" + data.name + ".pkl")    
    return garch, res


def run_simulation(garch, garch_parameters, num_simulation_days):
    sim = garch.simulate(garch_parameters, num_simulation_days)
    return sim['data']

def simulate_many_episodes(data, p, q, dist='normal', num_days = 750, num_simulation_episodes = 50, test_size = 0.2, 
                            initial_price = 100, load_model = False, save = False):

    train_simulations = int(num_simulation_episodes * (1 - test_size))
    test_simulations = num_simulation_episodes - train_simulations

    sim_df_train = pd.DataFrame(index=range(num_days + 1), columns=range(train_simulations))
    sim_df_test = pd.DataFrame(index=range(num_days + 1), columns=range(test_simulations))

    if load_model:
        models = load_garch_model([garch, res],  "saved_models/garch_and_fit_" + data.name + ".pkl")
        garch, res = models[0], models[1]
    else:
        garch, res = run_garch(data, p, q, dist, save)

    for i in tqdm(range(num_simulation_episodes), desc="Simulating episodes", unit="episode"):
        simulation = run_simulation(garch, res.params, num_days)
        simulated_log_returns = simulation / 100
        simulated_returns = np.exp(simulated_log_returns) - 1
        simulated_prices = initial_price * (1 + simulated_returns).cumprod()
        simulated_prices_with_initial = pd.concat([pd.Series([initial_price]), pd.Series(simulated_prices)], ignore_index=True)

        # Split the simulations into training and testing sets
        if i < train_simulations:
            sim_df_train[i] = simulated_prices_with_initial
        else:
            sim_df_test[i - train_simulations] = simulated_prices_with_initial

        save_data_csv(sim_df_train, "data/sim_train_" + data.name + ".csv")
        save_data_csv(sim_df_test, "data/sim_test_" + data.name + ".csv")

    return sim_df_train, sim_df_test


def save_garch_model(fitted_models, filename):
    with open(filename, 'wb') as f:
        pickle.dump(fitted_models, f)

def load_garch_model(filename):
    with open(filename, 'rb') as f:
        fitted_models = pickle.load(f)
    return fitted_models


########################
def load_simulation_data(tickers):
    stock_data_dict = {}
    for ticker in tickers:
        # Load training and testing data for each ticker
        train_file = f'data/sim_train_{ticker}.csv'
        test_file = f'data/sim_test_{ticker}.csv'
        if os.path.exists(train_file) and os.path.exists(test_file):
            train_data = pd.read_csv(train_file, index_col=0)
            test_data = pd.read_csv(test_file, index_col=0)
            stock_data_dict[ticker] = {'train': train_data, 'test': test_data}
        else:
            print(f"File not found for ticker: {ticker}")
    return stock_data_dict


def get_combined_simulation(stock_data_dict, simulation_index, set_type='train'):
    combined_data = []
    for stock in stock_data_dict.keys():
        # Access either 'train' or 'test' data and get the column specified by simulation_index
        stock_data = stock_data_dict[stock][set_type]
        combined_data.append(stock_data.iloc[:, simulation_index].values)
    return np.array(combined_data).T



def load_simulation_data_2(tickers):
    stock_data_dict = {}
    for ticker in tickers:
        # Load training and testing data for each ticker
        train_file = f'data/2_sim_train_{ticker}.csv'
        test_file = f'data/2_sim_test_{ticker}.csv'
        if os.path.exists(train_file) and os.path.exists(test_file):
            train_data = pd.read_csv(train_file, index_col=0)
            test_data = pd.read_csv(test_file, index_col=0)
            stock_data_dict[ticker] = {'train': train_data, 'test': test_data}
        else:
            print(f"File not found for ticker: {ticker}")
    return stock_data_dict

