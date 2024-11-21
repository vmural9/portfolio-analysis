# calculations.py
import pandas as pd
import numpy as np
from data_fetching import fetch_stock_data
import logging
from scipy.optimize import minimize

"""
Module for performing portfolio analysis calculations.

This module provides functions for calculating various portfolio metrics,
including returns, volatility, Sharpe ratio, and Monte Carlo simulations.
"""

def calculate_portfolio_returns(processed_data, portfolio_weights):
    """
    Calculates daily returns for a portfolio given stock data and weights.

    Args:
        processed_data (dict): Dictionary of processed stock data with ticker
            symbols as keys and pandas DataFrames as values.
        portfolio_weights (dict): Dictionary of portfolio weights with ticker
            symbols as keys and weights as values.

    Returns:
        pandas.Series: Daily portfolio returns.

    Raises:
        ValueError: If portfolio weights don't sum to 1 or if data is missing.
    """
    logger = logging.getLogger('portfolio_analyzer')
    
    try:
        weights = pd.Series(portfolio_weights)
        price_data = pd.DataFrame()
        for ticker, data in processed_data.items():
            price_data[ticker] = data['Adj Close']
        daily_returns = price_data.pct_change().dropna()
        portfolio_returns = daily_returns.dot(weights)
        
        logger.debug(f"Calculated portfolio returns with shape: {portfolio_returns.shape}")
        return portfolio_returns
        
    except Exception as e:
        logger.error(f"Error calculating portfolio returns: {str(e)}")
        raise

def calculate_portfolio_volatility(portfolio_returns):
    """
    Calculates the annualized volatility of portfolio returns.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.

    Returns:
        float: Annualized portfolio volatility (standard deviation).

    Notes:
        Annualization assumes 252 trading days per year.
    """
    daily_volatility = portfolio_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.01):
    """
    Calculates the annualized Sharpe ratio for the portfolio.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.01 (1%).

    Returns:
        float: Annualized Sharpe ratio.

    Notes:
        - Sharpe ratio is calculated as (portfolio_return - risk_free_rate) / portfolio_volatility
        - Annualization assumes 252 trading days per year
    """
    daily_risk_free_rate = risk_free_rate / 252
    excess_returns = portfolio_returns - daily_risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)
    return annualized_sharpe_ratio

def calculate_beta(portfolio_returns, benchmark_returns):
    """
    Calculates the portfolio's beta relative to a benchmark.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.
        benchmark_returns (pandas.Series): Daily benchmark returns.

    Returns:
        float: Portfolio beta coefficient.

    Notes:
        Beta measures the portfolio's systematic risk relative to the market,
        where beta = 1 indicates same volatility as the market.
    """
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    benchmark_variance = benchmark_returns.var()
    beta = covariance / benchmark_variance
    return beta

def calculate_var(portfolio_returns, confidence_level=0.05):
    """
    Calculates the Value at Risk (VaR) for the portfolio.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.
        confidence_level (float, optional): Confidence level for VaR calculation.
            Defaults to 0.05 (95% confidence).

    Returns:
        float: Value at Risk at specified confidence level.

    Notes:
        VaR represents the maximum expected loss at the given confidence level.
    """
    var = np.percentile(portfolio_returns, 100 * confidence_level)
    return var

def monte_carlo_simulation(processed_data, portfolio_weights, num_simulations=1000, forecast_days=252):
    """
    Performs Monte Carlo simulation to forecast potential portfolio outcomes.

    Args:
        processed_data (dict): Dictionary of processed stock data.
        portfolio_weights (dict): Dictionary of portfolio weights.
        num_simulations (int, optional): Number of simulation runs. Defaults to 1000.
        forecast_days (int, optional): Number of days to forecast. Defaults to 252.

    Returns:
        numpy.ndarray: Array of simulation results representing final portfolio values.

    Raises:
        ValueError: If input data is invalid or missing.
    """
    logger = logging.getLogger('portfolio_analyzer')
    
    try:
        logger.info(f"Starting Monte Carlo simulation with {num_simulations} simulations")
        weights = np.array(list(portfolio_weights.values()))
        price_data = pd.DataFrame()
        for ticker, data in processed_data.items():
            price_data[ticker] = data['Adj Close']
            
        daily_returns = price_data.pct_change().dropna()
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        
        simulation_results = np.zeros(num_simulations)
        for i in range(num_simulations):
            if i % 200 == 0:  # Log progress every 200 simulations
                logger.debug(f"Completed {i} simulations")
                
            simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, forecast_days)
            portfolio_simulated_returns = np.cumprod(np.dot(simulated_returns, weights) + 1)
            simulation_results[i] = portfolio_simulated_returns[-1]
        
        logger.info("Monte Carlo simulation completed successfully")
        return simulation_results
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        raise

def calculate_portfolio_metrics(processed_data, portfolio_weights):
    """
    Calculates key portfolio metrics including returns, volatility, and risk measures.

    Args:
        processed_data (dict): Dictionary of processed stock data.
        portfolio_weights (dict): Dictionary of portfolio weights.

    Returns:
        dict: Dictionary containing calculated metrics:
            - Expected Return (float): Annualized expected return
            - Volatility (float): Annualized volatility
            - Sharpe Ratio (float): Risk-adjusted return measure
            - Beta (float): Portfolio beta relative to S&P 500
            - VaR (float): Value at Risk at 95% confidence level

    Raises:
        ValueError: If unable to calculate metrics due to invalid data.
    """
    metrics = {}
    portfolio_returns = calculate_portfolio_returns(processed_data, portfolio_weights)
    metrics['Expected Return'] = portfolio_returns.mean() * 252
    metrics['Volatility'] = calculate_portfolio_volatility(portfolio_returns)
    metrics['Sharpe Ratio'] = calculate_sharpe_ratio(portfolio_returns)
    
    # Fetch benchmark data
    benchmark_data = fetch_stock_data(['^GSPC'], portfolio_returns.index[0], portfolio_returns.index[-1])
    benchmark_returns = benchmark_data['^GSPC']['Adj Close'].pct_change().dropna()
    
    # Align dates
    aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    portfolio_returns_aligned = aligned_returns.iloc[:, 0]
    benchmark_returns_aligned = aligned_returns.iloc[:, 1]
    
    metrics['Beta'] = calculate_beta(portfolio_returns_aligned, benchmark_returns_aligned)
    metrics['VaR'] = calculate_var(portfolio_returns)
    
    return metrics

def calculate_portfolio_stats(weights, processed_data):
    """
    Calculate portfolio statistics (return, volatility).
    """
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
    
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return portfolio_return, portfolio_volatility

def optimize_portfolio(processed_data, target_return=None, allow_short=False, constraints=None):
    """
    Optimize portfolio weights to minimize variance for a given target return.
    
    Args:
        processed_data (dict): Dictionary of processed stock data
        target_return (float, optional): Target annual return. If None, finds minimum variance portfolio
        allow_short (bool): Whether to allow short selling
        constraints (dict, optional): Dictionary with 'min' and 'max' allocation constraints per asset
    
    Returns:
        tuple: (optimal weights, expected return, volatility)
    """
    # Prepare data
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
    
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = len(processed_data.keys())
    
    # Define constraints
    bounds = (-1, 1) if allow_short else (0, 1)
    if constraints:
        bounds = [(constraints.get('min', 0), constraints.get('max', 1)) for _ in range(num_assets)]
    else:
        bounds = [bounds for _ in range(num_assets)]
    
    # Constraint: weights sum to 1
    constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Add target return constraint if specified
    if target_return is not None:
        constraints_list.append({
            'type': 'eq',
            'fun': lambda x: np.sum(mean_returns * x) * 252 - target_return
        })
    
    # Objective function: minimize portfolio variance
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Optimize
    result = minimize(objective, initial_weights,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints_list)
    
    optimal_weights = result.x
    return_value, volatility = calculate_portfolio_stats(optimal_weights, processed_data)
    
    return optimal_weights, return_value, volatility
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    optimal_weights = result.x
    return_value, volatility = calculate_portfolio_stats(optimal_weights, mean_returns, cov_matrix)
    
    return optimal_weights, return_value, volatility

def generate_efficient_frontier(processed_data, num_points=100, allow_short=False, constraints=None):
    """
    Generate efficient frontier points.
    
    Returns:
        tuple: (returns, volatilities, weights_list)
    """
    # Get minimum variance portfolio
    min_weights, min_ret, min_vol = optimize_portfolio(processed_data, allow_short=allow_short, constraints=constraints)
    
    # Get maximum return portfolio (by optimizing for return instead of variance)
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
    
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    
    # Find portfolio with maximum return
    num_assets = len(processed_data.keys())
    bounds = (-1, 1) if allow_short else (0, 1)
    if constraints:
        bounds = [(constraints.get('min', 0), constraints.get('max', 1)) for _ in range(num_assets)]
    else:
        bounds = [bounds for _ in range(num_assets)]
    
    # For maximum return, we'll maximize the negative of return (since minimize is our only option)
    def objective(weights):
        return -np.sum(mean_returns * weights) * 252
    
    constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    result = minimize(objective, 
                     np.array([1/num_assets] * num_assets),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints_list)
    
    max_weights = result.x
    max_ret, max_vol = calculate_portfolio_stats(max_weights, processed_data)
    
    # Generate points between minimum variance and maximum return
    target_returns = np.linspace(min_ret, max_ret, num_points)
    efficient_portfolios = []
    
    for target in target_returns:
        try:
            weights, ret, vol = optimize_portfolio(
                processed_data, 
                target_return=target,
                allow_short=allow_short,
                constraints=constraints
            )
            efficient_portfolios.append((ret, vol, weights))
        except:
            continue
    
    returns = [p[0] for p in efficient_portfolios]
    volatilities = [p[1] for p in efficient_portfolios]
    weights_list = [p[2] for p in efficient_portfolios]
    
    return returns, volatilities, weights_list
