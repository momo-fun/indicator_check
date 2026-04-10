"""
Download daily price data for all S&P 500 constituents.

Ticker list priority:
  1. sp500.csv in the same directory (cached from a previous run)
  2. Scraped live from Wikipedia

Output: price_data/<TICKER>.csv  (one file per ticker, 5 years of daily data)
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf

WIKI_URL    = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "price_data"))
PERIOD      = "5y"
INTERVAL    = "1d"
SLEEP_SEC   = 0.3   # courtesy delay between downloads

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; sp500-downloader/1.0)"
}


def get_sp500_tickers() -> list[str]:
    """Return up to 500 S&P 500 tickers.

    Loads from sp500.csv if it exists alongside this script, otherwise
    fetches and parses the Wikipedia constituents table.
    """
    cached_sp500_path = os.path.join(os.path.dirname(__file__), "sp500.csv")
    if os.path.exists(cached_sp500_path):
        print(f"Loading ticker list from cache: {cached_sp500_path}")
        sp500_df = pd.read_csv(cached_sp500_path)
    else:
        print(f"Fetching ticker list from Wikipedia…")
        r = requests.get(WIKI_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        sp500_df = tables[0]

    tickers = sp500_df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return tickers[:500]


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
