import plotly.express as px
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Set the default rendering mode to open in the browser
pio.renderers.default = "browser"

def plot_original_stock(tickers, start_date, end_date):
    for i in tickers:
        stock = yf.Ticker(i)
        data = stock.history(start="2000-01-01", end="2024-01-01")
        
        # Using Plotly to create the interactive plot
        fig = px.line(data, x=data.index, y='Close', title=f'Stock Price Movement for {i}')
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        fig.show()  # This will open the plot in your browser

def plot_simulations(sim_df, title="Simulations"):
    # Convert the simulation DataFrame to long format for Plotly
    sim_df_long = sim_df.reset_index().melt(id_vars='index', var_name='Simulation', value_name='Price')
    sim_df_long.rename(columns={'index': 'Days'}, inplace=True)
    
    # Using Plotly to plot multiple simulations
    fig = px.line(sim_df_long, x='Days', y='Price', color='Simulation', title=title)
    fig.update_layout(xaxis_title="Days", yaxis_title="Price")
    fig.show()  # This will open the plot in your browser

def plot_one_combined_simulation(combined_data, tickers, s_type='train'):
    df = pd.DataFrame(combined_data, columns=tickers)
    df['Days'] = df.index
    
    # Using Plotly to plot combined simulation
    fig = px.line(df, x='Days', y=tickers, title=f"Combined {s_type} Simulation for Selected Stocks")
    fig.update_layout(xaxis_title="Days", yaxis_title="Price")
    fig.show()  # This will open the plot in your browser
