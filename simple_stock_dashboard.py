import pandas as pd
import ta  # Technical analysis library

# Tickers of companies -> can be expanded
companies = ["ORCL", "GOOGL", "AAPL", "MSFT"]


def load_stock_data(company):
    try:
        data = pd.read_csv(
            f"Stock-Analysis/Financial Data/HistoricalData_{company}.csv"
        )

        # Removing dollar sign
        for col in ["Close/Last", "Open", "High", "Low"]:
            if col in data.columns:
                data[col] = data[col].str.replace("$", "").astype(float)

        # Close/Last -> Close
        data = data.rename(columns={"Close/Last": "Close"})

        return data
    except:
        print(f"Could not load data for {company}")
        return None


def calculate_indicators(data, company):
    if data is None or len(data) < 20:  # 20 days window
        return None

    # Price Change
    first_price = data["Close"].iloc[0]
    last_price = data["Close"].iloc[-1]
    change_pct = ((last_price - first_price) / first_price) * 100

    # Technical Indicators: SMA, EMA
    data["SMA_20"] = ta.trend.sma_indicator(data["Close"], window=20)
    data["EMA_20"] = ta.trend.ema_indicator(data["Close"], window=20)

    # Get the latest values
    current_sma = data["SMA_20"].iloc[-1]
    current_ema = data["EMA_20"].iloc[-1]

    # Determine trend
    trend = "↑" if last_price > current_ema else "↓"

    return {
        "Company": company,
        "Price": f"${last_price:.2f}",
        "Change": f"{change_pct:.1f}%",
        "SMA(20)": f"${current_sma:.2f}",
        "EMA(20)": f"${current_ema:.2f}",
        "Trend": trend,
    }


def print_results(results):
    print("\nSTOCK ANALYSIS REPORT")
    print("=" * 70)
    print(
        "{:<6} | {:>10} | {:>8} | {:>10} | {:>10} | Trend".format(
            "Company", "Price", "Change", "SMA(20)", "EMA(20)"
        )
    )
    print("-" * 70)

    for result in results:
        if result is not None:
            print(
                "{:<6} | {:>10} | {:>8} | {:>10} | {:>10} | {:>5}".format(
                    result["Company"],
                    result["Price"],
                    result["Change"],
                    result["SMA(20)"],
                    result["EMA(20)"],
                    result["Trend"],
                )
            )

    print("\nTrend Guide: ↑ = Price above EMA(20), ↓ = Price below EMA(20)")


def main():
    all_results = []

    for company in companies:
        data = load_stock_data(company)
        company_result = calculate_indicators(data, company)
        all_results.append(company_result)

    print_results([r for r in all_results if r is not None])


# Run the program
if __name__ == "__main__":
    main()
