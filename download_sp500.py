#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:58:39 2026

@author: momoc
"""

import os
import pandas as pd
import requests
import yfinance as yf

# 1. Setup environment
os.makedirs("price_data", exist_ok=True)
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

# 2. Get the Ticker List
cached_sp500_path = "sp500.csv"

if os.path.exists(cached_sp500_path):
    print("Loading tickers from cache...")
    sp500_df = pd.read_csv(cached_sp500_path)
else:
    print("Fetching tickers from Wikipedia...")
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    sp500_df = tables[0]
    # Save the list locally for next time
    sp500_df.to_csv(cached_sp500_path, index=False)

# Clean tickers (yfinance uses '-' instead of '.' for classes, e.g., BRK.B -> BRK-B)
tickers = sp500_df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()

# 3. Download Data
print(f"Starting download for {len(tickers)} tickers...")

for t in tickers:
    try:
        data = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
        
        if not data.empty:
            data.to_csv(f"price_data/{t}.csv")
            print(f"Successfully downloaded: {t}")
        else:
            print(f"No data found for: {t}")
            
    except Exception as e:
        print(f"Could not download {t}: {e}")

print("Done! Check the 'price_data' folder.")
