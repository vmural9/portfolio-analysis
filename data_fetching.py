# data_fetching.py
import yfinance as yf

"""
Module for fetching stock data from various financial data sources.

This module provides functionality to retrieve historical stock data
using the yfinance API.
"""

def fetch_stock_data(ticker_list, start_date, end_date):
    """
    Fetches historical stock data for given tickers within a date range.

    Args:
        ticker_list (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        dict: Dictionary with ticker symbols as keys and pandas DataFrames containing
            stock data as values.

    Raises:
        yfinance.YFinanceError: If there's an error fetching data from Yahoo Finance.
    """
    data = {}
    for ticker in ticker_list:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data
    return data
