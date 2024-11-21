# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import logging

"""
Module for creating portfolio analysis visualizations.

This module provides functions for plotting various aspects of portfolio
performance, including returns, efficient frontier, and Monte Carlo simulations.
"""

def plot_portfolio_performance(portfolio_returns):
    """
    Plots the cumulative performance of the portfolio over time.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns with datetime index.

    Notes:
        - Displays cumulative returns starting from an initial value of 1
        - Includes date axis and grid for better readability
    """
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
    """
    Plots the efficient frontier showing risk-return tradeoff of random portfolios.

    Args:
        processed_data (dict): Dictionary of processed stock data.
        num_portfolios (int, optional): Number of random portfolios to generate.
            Defaults to 5000.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.01.

    Raises:
        ValueError: If unable to generate efficient frontier due to invalid data.
    """
    logger = logging.getLogger('portfolio_analyzer')
    
    try:
        logger.debug(f"Generating efficient frontier with {num_portfolios} portfolios")
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
        logger.debug("Efficient frontier plot generated successfully")
        plt.show()
        
    except Exception as e:
        logger.error(f"Error generating efficient frontier plot: {str(e)}")
        raise

def plot_individual_assets(processed_data):
    """
    Creates a scatter plot of individual assets' risk-return characteristics.

    Args:
        processed_data (dict): Dictionary of processed stock data with ticker
            symbols as keys and pandas DataFrames as values.

    Notes:
        - Plots annualized return vs. annualized volatility for each asset
        - Labels each point with the corresponding ticker symbol
        - Assumes 252 trading days for annualization
    """
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
    """
    Creates a histogram of Monte Carlo simulation results.

    Args:
        simulation_results (numpy.ndarray): Array of simulation ending values.
        initial_investment (float, optional): Initial portfolio value.
            Defaults to 10000.

    Notes:
        Displays a histogram with KDE and marks the 5th percentile value.
    """
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
    """
    Plots the distribution of portfolio returns.

    Args:
        portfolio_returns (pandas.Series): Series of portfolio returns.

    Notes:
        Creates a histogram with kernel density estimation (KDE) of returns.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_returns, bins=50, kde=True)
    plt.title('Distribution of Portfolio Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
