# main.py
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
from calculations import calculate_portfolio_metrics, monte_carlo_simulation, calculate_portfolio_returns, minimize_variance
from visualization import (
    plot_portfolio_performance,
    plot_efficient_frontier,
    plot_individual_assets,
    plot_monte_carlo_results,
    plot_return_distribution,
    plot_monte_carlo_results,
    plot_efficient_frontier_with_optimal
)

def main():
    # User inputs
    portfolio = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
    start_date = '2020-01-01'
    end_date = '2023-10-01'

    # Fetch data
    raw_data = fetch_stock_data(list(portfolio.keys()), start_date, end_date)

    # Process data
    processed_data = process_stock_data(raw_data)
    # print(processed_data)

    # Calculate metrics
    metrics = calculate_portfolio_metrics(processed_data, portfolio)
    portfolio_returns = calculate_portfolio_returns(processed_data, portfolio)
    print("Portfolio Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Use the current portfolio's return as a baseline for target return
    current_return = metrics['Expected Return']
    target_return = current_return * 0.9  # Try for 90% of current return
    allow_short = False

    # Flag to track optimization success
    optimization_succeeded = False
    
    try:
        optimal_weights, opt_return, opt_volatility = minimize_variance(
            processed_data, target_return, allow_short
        )
        print("\nOptimal Portfolio Weights:")
        for ticker, weight in zip(portfolio.keys(), optimal_weights):
            print(f"{ticker}: {weight:.4f}")
        print(f"Optimal Portfolio Return: {opt_return:.4f}")
        print(f"Optimal Portfolio Volatility: {opt_volatility:.4f}")
        optimization_succeeded = True
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Visualizations
    plot_portfolio_performance(portfolio_returns)
    plot_individual_assets(processed_data)
    plot_efficient_frontier(processed_data)
    
    if optimization_succeeded:
        plot_efficient_frontier_with_optimal(
            processed_data, optimal_weights, opt_return, opt_volatility, allow_short
        )
    
    plot_return_distribution(portfolio_returns)

    # Monte Carlo Simulation
    simulation_results = monte_carlo_simulation(processed_data, portfolio)
    plot_monte_carlo_results(simulation_results)

if __name__ == '__main__':
    main()
