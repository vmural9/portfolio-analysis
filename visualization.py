# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_portfolio_performance(portfolio_returns):
    cumulative_returns = (1 + portfolio_returns).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns, label='Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Portfolio Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_efficient_frontier(processed_data, num_portfolios=5000, risk_free_rate=0.01):
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
    
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = len(processed_data.keys())
    
    # Create lists instead of numpy arrays for collecting results
    returns = []
    volatilities = []
    sharpe_ratios = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        returns.append(portfolio_return)
        volatilities.append(portfolio_volatility)
        sharpe_ratios.append(sharpe_ratio)
    
    # Convert to numpy arrays for plotting
    returns = np.array(returns)
    volatilities = np.array(volatilities)
    sharpe_ratios = np.array(sharpe_ratios)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True)
    plt.show()

def plot_individual_assets(processed_data):
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
    
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean() * 252
    volatilities = daily_returns.std() * np.sqrt(252)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(volatilities, mean_returns, marker='o', s=100)
    for i, ticker in enumerate(mean_returns.index):
        plt.annotate(ticker, (volatilities.iloc[i], mean_returns.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Risk vs Return of Individual Assets')
    plt.grid(True)
    plt.show()

def plot_monte_carlo_results(simulation_results, initial_investment=10000):
    ending_values = simulation_results * initial_investment
    plt.figure(figsize=(10, 6))
    sns.histplot(ending_values, bins=50, kde=True)
    plt.title('Monte Carlo Simulation Results')
    plt.xlabel('Portfolio Ending Value')
    plt.ylabel('Frequency')
    plt.axvline(x=np.percentile(ending_values, 5), color='r', linestyle='--', label='5th Percentile')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_return_distribution(portfolio_returns):
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_returns, bins=50, kde=True)
    plt.title('Distribution of Portfolio Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
