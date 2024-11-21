# calculations.py
import pandas as pd
import numpy as np
from data_fetching import fetch_stock_data
from scipy.optimize import minimize

def calculate_portfolio_returns(processed_data, portfolio_weights):
    weights = pd.Series(portfolio_weights)
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
    daily_returns = price_data.pct_change().dropna()
    portfolio_returns = daily_returns.dot(weights)
    return portfolio_returns

def calculate_portfolio_volatility(portfolio_returns):
    daily_volatility = portfolio_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.01):
    daily_risk_free_rate = risk_free_rate / 252
    excess_returns = portfolio_returns - daily_risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)
    return annualized_sharpe_ratio

def calculate_beta(portfolio_returns, benchmark_returns):
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    benchmark_variance = benchmark_returns.var()
    beta = covariance / benchmark_variance
    return beta

def calculate_var(portfolio_returns, confidence_level=0.05):
    var = np.percentile(portfolio_returns, 100 * confidence_level)
    return var

def monte_carlo_simulation(processed_data, portfolio_weights, num_simulations=1000, forecast_days=252):
    weights = np.array(list(portfolio_weights.values()))
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    simulation_results = np.zeros(num_simulations)
    for i in range(num_simulations):
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, forecast_days)
        portfolio_simulated_returns = np.cumprod(np.dot(simulated_returns, weights) + 1)
        simulation_results[i] = portfolio_simulated_returns[-1]
    
    return simulation_results

def calculate_portfolio_metrics(processed_data, portfolio_weights):
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

def minimize_variance(processed_data, target_return, allow_short=False):
    # Create empty DataFrame first
    price_data = pd.DataFrame()
    
    # Add columns one by one
    for ticker, data in processed_data.items():
        price_data[ticker] = data['Adj Close']
        
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean().values * 252  # Annualized returns
    cov_matrix = daily_returns.cov().values * 252     # Annualized covariance
    
    num_assets = len(mean_returns)
    args = (cov_matrix,)

    # Initial guess as numpy array
    init_guess = np.array([1. / num_assets] * num_assets)

    # Constraints with ravel() to ensure 1D
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns.ravel()) - target_return}
    )

    # Bounds
    if allow_short:
        bounds = [(-1.0, 1.0) for _ in range(num_assets)]
    else:
        bounds = [(0.0, 1.0) for _ in range(num_assets)]

    # Objective function to minimize (portfolio variance)
    def portfolio_variance(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # Optimize
    result = minimize(portfolio_variance, init_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise Exception("Optimization failed: " + result.message)

    # Get the optimal weights
    optimal_weights = result.x

    # Calculate optimal portfolio metrics
    portfolio_return = np.dot(optimal_weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

    return optimal_weights, portfolio_return, portfolio_volatility

