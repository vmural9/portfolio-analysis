# main.py
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
from calculations import (
    calculate_portfolio_metrics,
    monte_carlo_simulation,
    calculate_portfolio_returns,
    optimize_portfolio
)
from visualization import (
    plot_portfolio_performance,
    plot_efficient_frontier,
    plot_individual_assets,
    plot_monte_carlo_results,
    plot_return_distribution
)
from logger_config import setup_logger

"""
Main module for portfolio analysis application.

This module orchestrates the portfolio analysis process by combining
data fetching, processing, calculations, and visualization components.
It serves as the entry point for the application.
"""

def main():
    """
    Main function to run the portfolio analysis.

    The function performs the following steps:
    1. Fetches historical stock data
    2. Processes and cleans the data
    3. Calculates portfolio metrics
    4. Generates visualizations
    5. Runs Monte Carlo simulations

    Raises:
        Exception: If any step in the analysis process fails.
    """
    # Setup logger
    logger = setup_logger()
    logger.info("Starting portfolio analysis")

    try:
        # User inputs
        portfolio = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        start_date = '2020-01-01'
        end_date = '2023-10-01'
        
        logger.info(f"Analyzing portfolio: {portfolio}")
        logger.info(f"Time period: {start_date} to {end_date}")

        # Fetch data
        logger.debug("Fetching stock data...")
        raw_data = fetch_stock_data(list(portfolio.keys()), start_date, end_date)

        # Process data
        logger.debug("Processing stock data...")
        processed_data = process_stock_data(raw_data)

        # Calculate metrics
        logger.debug("Calculating portfolio metrics...")
        metrics = calculate_portfolio_metrics(processed_data, portfolio)
        portfolio_returns = calculate_portfolio_returns(processed_data, portfolio)
        
        logger.info("Portfolio Metrics:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        # Visualizations
        logger.debug("Generating visualizations...")
        plot_portfolio_performance(portfolio_returns)
        plot_individual_assets(processed_data)
        plot_efficient_frontier(
            processed_data, 
            current_weights=portfolio,
            allow_short=False,
            target_return=metrics['Expected Return'] * 0.9
        )
        plot_return_distribution(portfolio_returns)

        # Monte Carlo Simulation
        logger.debug("Running Monte Carlo simulation...")
        simulation_results = monte_carlo_simulation(processed_data, portfolio)
        plot_monte_carlo_results(simulation_results)
        
        # Hardcoded values (previously in values.txt)
        target_return = metrics['Expected Return'] * 0.9  # 90% of current return
        allow_short = False

        # Calculate optimal portfolio
        optimal_weights, opt_return, opt_vol = optimize_portfolio(
            processed_data,
            target_return=target_return,
            allow_short=allow_short
        )
        
        # Log optimization results
        logger.info("\nOptimal Portfolio Results:")
        for ticker, weight in zip(portfolio.keys(), optimal_weights):
            logger.info(f"{ticker} weight: {weight:.4f}")
        logger.info(f"Optimal Portfolio Return: {opt_return:.4f}")
        logger.info(f"Optimal Portfolio Volatility: {opt_vol:.4f}")
        
        logger.info("Portfolio analysis completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during portfolio analysis: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
