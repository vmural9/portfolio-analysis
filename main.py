# main.py
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
from calculations import calculate_portfolio_metrics, monte_carlo_simulation, calculate_portfolio_returns
from visualization import (
    plot_portfolio_performance,
    plot_efficient_frontier,
    plot_individual_assets,
    plot_monte_carlo_results,
    plot_return_distribution
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

    # Visualizations
    plot_portfolio_performance(portfolio_returns)
    plot_individual_assets(processed_data)
    plot_efficient_frontier(processed_data)
    plot_return_distribution(portfolio_returns)

    # Monte Carlo Simulation
    simulation_results = monte_carlo_simulation(processed_data, portfolio)
    plot_monte_carlo_results(simulation_results)

if __name__ == '__main__':
    main()
