import yfinance as yf
import time
from datetime import datetime


class StockPriceMonitor:
    def __init__(self, ticker_symbol, interval=60, max_retries=3):
        self.ticker_symbol = ticker_symbol.upper()
        self.interval = interval
        self.max_retries = max_retries

    def run(self):
        print(
            f"Checking the stock price for {self.ticker_symbol} every {self.interval} seconds."
        )

        while True:
            try:
                ticker_price = None
                for attempt in range(self.max_retries):
                    try:
                        ticker_data = yf.Ticker(self.ticker_symbol)
                        hist = ticker_data.history(
                            period="1d", interval="1m", prepost=True
                        )

                        if not hist.empty:
                            ticker_price = hist["Close"].iloc[-1]
                        break
                    except Exception as retry_error:
                        print(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Retry {attempt + 1}/{self.max_retries} failed: {retry_error}"
                        )
                        if attempt + 1 == self.max_retries:
                            raise retry_error
                        time.sleep(2)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if ticker_price is not None:
                    print(
                        f"{current_time} - The current stock price of {self.ticker_symbol} is: ${ticker_price:.2f}"
                    )
                else:
                    print(
                        f"{current_time} - Could not retrieve the stock price for {self.ticker_symbol}."
                    )

            except Exception as e:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - An error occurred: {e}"
                )
                print(
                    "Please check your network or Yahoo Finance API status. See https://curl.se/libcurl/c/libcurl-errors.html for details."
                )

            time.sleep(self.interval)


if __name__ == "__main__":
    ticker_symbol = input("Enter the ticker symbol: ")
    monitor = StockPriceMonitor(ticker_symbol)
    monitor.run()
