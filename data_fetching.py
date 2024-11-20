# data_fetching.py
import yfinance as yf

def fetch_stock_data(ticker_list, start_date, end_date):
    data = {}
    for ticker in ticker_list:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data
    return data
