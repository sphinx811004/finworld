import os
import json
import asyncio
import yfinance as yf
import pandas as pd
from finworld.factor.alpha158 import Alpha158

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    assets_name = "exp"
    source = "yfinance"
    level = "1day"
    symbols = ["AAPL", "AMZN", "GOOGL", "MSFT"]
    start_date = "2015-05-01"
    end_date = "2025-05-02"

    exp_dir = os.path.join(root, "datasets", assets_name)
    price_tag = f"{assets_name}_{source}_price_{level}"
    feature_tag = f"{assets_name}_{source}_feature_{level}"

    price_dir = os.path.join(exp_dir, price_tag)
    feature_dir = os.path.join(exp_dir, feature_tag)
    os.makedirs(price_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    print(f"Downloading historical data for {symbols} from {start_date} to {end_date}...")
    raw_df = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False, progress=False)

    # Extract clean OHLCV per symbol and ensure matching index
    symbol_dfs = {}
    common_index = raw_df.index
    for s in symbols:
        df_s = pd.DataFrame(index=common_index)
        df_s["open"] = raw_df["Open"][s]
        df_s["high"] = raw_df["High"][s]
        df_s["low"] = raw_df["Low"][s]
        df_s["close"] = raw_df["Close"][s]
        df_s["volume"] = raw_df["Volume"][s]
        df_s = df_s.dropna()
        symbol_dfs[s] = df_s

    # Find common valid dates across all symbols
    valid_dates = None
    for s in symbols:
        if valid_dates is None:
            valid_dates = set(symbol_dfs[s].index)
        else:
            valid_dates = valid_dates.intersection(set(symbol_dfs[s].index))

    valid_dates = sorted(list(valid_dates))
    print(f"Found {len(valid_dates)} common trading dates across all {len(symbols)} symbols.")

    # Save cleaned price data
    for s in symbols:
        df_s = symbol_dfs[s].loc[valid_dates].copy()
        df_s.insert(0, "timestamp", df_s.index.strftime("%Y-%m-%d"))
        df_s = df_s.reset_index(drop=True)
        price_file = os.path.join(price_dir, f"{s}.jsonl")
        df_s.to_json(price_file, orient="records", lines=True)
        print(f"Saved price data for {s}: {len(df_s)} rows to {price_file}")
        symbol_dfs[s] = df_s

    # Compute Alpha158 features for each symbol
    alpha = Alpha158()
    for s in symbols:
        print(f"Computing Alpha158 factors for {s}...")
        df_s = symbol_dfs[s].copy()
        res = asyncio.run(alpha.run(df_s))
        factors_df = res["factors_df"]

        if pd.api.types.is_datetime64_any_dtype(factors_df["timestamp"]):
            factors_df["timestamp"] = factors_df["timestamp"].dt.strftime("%Y-%m-%d")

        feature_file = os.path.join(feature_dir, f"{s}.jsonl")
        factors_df.to_json(feature_file, orient="records", lines=True)
        print(f"Saved feature data for {s}: {len(factors_df.columns)-1} factors to {feature_file}")

    # Build meta_info.json
    asset_list_path = os.path.join(root, "configs", "_asset_list_", "exp.json")
    with open(asset_list_path) as f:
        full_info = json.load(f)

    symbols_info = {}
    for s in symbols:
        symbols_info[s] = full_info.get(s, {"symbol": s, "exchange": "NASDAQ Global Select"})

    meta = {
        "symbols": symbols,
        "symbols_info": symbols_info,
        "tags": [price_tag, feature_tag]
    }

    meta_path = os.path.join(exp_dir, "meta_info.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully updated {meta_path} for {len(symbols)} portfolio assets!")

if __name__ == "__main__":
    main()
