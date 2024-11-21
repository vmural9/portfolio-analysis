# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import logging
from calculations import calculate_portfolio_stats, optimize_portfolio, generate_efficient_frontier

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

def plot_efficient_frontier(processed_data, current_weights=None, allow_short=False, constraints=None, target_return=None):
    """
    Plot the efficient frontier with current, optimal, and minimum variance portfolios.
    
    Args:
        processed_data (dict): Dictionary of processed stock data
        current_weights (dict, optional): Current portfolio weights
        allow_short (bool): Whether to allow short selling
        constraints (dict, optional): Dictionary with 'min' and 'max' allocation constraints
        target_return (float, optional): Target return for optimal portfolio
    """
    logger = logging.getLogger('portfolio_analyzer')
    
    try:
        # Generate efficient frontier points
        returns, volatilities, weights = generate_efficient_frontier(
            processed_data, allow_short=allow_short, constraints=constraints
        )
        
        # Plot efficient frontier
        plt.figure(figsize=(12, 8))
        plt.plot(volatilities, returns, 'b-', label='Efficient Frontier')
        
        # Plot current portfolio if provided
        if current_weights is not None:
            current_return, current_vol = calculate_portfolio_stats(
                np.array(list(current_weights.values())),
                processed_data
            )
            plt.plot(current_vol, current_return, 'r*', markersize=15, label='Current Portfolio')
        
        # Find and plot minimum variance portfolio
        min_var_weights, min_var_return, min_var_vol = optimize_portfolio(processed_data)
        plt.plot(min_var_vol, min_var_return, 'g*', markersize=15, label='Minimum Variance')
        
        # Plot optimal portfolio if target return is provided
        if target_return is not None:
            optimal_weights, opt_return, opt_vol = optimize_portfolio(
                processed_data, target_return=target_return, allow_short=allow_short
            )
            plt.plot(opt_vol, opt_return, 'y*', markersize=15, label='Optimal Portfolio')
        
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        logger.debug("Efficient frontier plot generated successfully")
        
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
