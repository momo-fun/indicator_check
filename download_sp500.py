"""
Download daily price data for all S&P 500 constituents.

Tickers are scraped from Wikipedia at runtime:
  https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

Output: price_data/<TICKER>.csv  (one file per ticker, 5 years of daily data)
"""

import os
import time
import pandas as pd
import yfinance as yf

WIKI_URL    = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "price_data"))
PERIOD      = "5y"
INTERVAL    = "1d"
SLEEP_SEC   = 0.3   # courtesy delay between downloads


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia."""
    tables = pd.read_html(WIKI_URL, attrs={"id": "constituents"})
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # yfinance uses '-' where Wikipedia uses '.' (e.g. BRK.B → BRK-B)
    return [t.replace(".", "-") for t in tickers]


def download_all(tickers: list[str]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    total = len(tickers)
    failed = []

    for i, ticker in enumerate(tickers, 1):
        csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        print(f"[{i:>3}/{total}] {ticker:<8}", end="  ", flush=True)
        try:
            df = yf.download(
                ticker,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                raise ValueError("empty result")
            df.to_csv(csv_path)
            print(f"OK  ({len(df)} bars)")
        except Exception as exc:
            print(f"FAILED  ({exc.__class__.__name__}: {exc})")
            failed.append(ticker)

        time.sleep(SLEEP_SEC)

    print(f"\nDone. {total - len(failed)}/{total} tickers saved to {DATA_DIR}/")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    print(f"Fetching S&P 500 ticker list from Wikipedia…")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers.\n")
    download_all(tickers)
