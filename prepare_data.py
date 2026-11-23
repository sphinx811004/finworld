import os
import json
import asyncio
import pandas as pd
from finworld.factor.alpha158 import Alpha158
from finworld.log import logger

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(root, "datasets", "exp")
    price_tag = "AAPL_yfinance_price_1day"
    feature_tag = "AAPL_yfinance_feature_1day"
    
    price_dir = os.path.join(exp_dir, price_tag)
    feature_dir = os.path.join(exp_dir, feature_tag)
    os.makedirs(price_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    
    price_file = os.path.join(price_dir, "AAPL.jsonl")
    feature_file = os.path.join(feature_dir, "AAPL.jsonl")
    
    print(f"Reading price data from {price_file}...")
    df = pd.read_json(price_file, lines=True)
    
    # Standardize columns to timestamp, open, high, low, close, volume
    price_cols = ["open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp")
    df = df[["timestamp"] + price_cols].reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    
    # Write cleaned price file
    df.to_json(price_file, orient="records", lines=True)
    print(f"Saved cleaned price data: {len(df)} rows, columns: {list(df.columns)}")
    
    # Compute Alpha158 features
    print("Computing Alpha158 factors...")
    alpha = Alpha158()
    res = asyncio.run(alpha.run(df.copy()))
    factors_df = res["factors_df"]
    print(f"Computed {len(factors_df.columns) - 1} factors across {len(factors_df)} rows")
    
    # Ensure factors_df timestamp matches format
    if pd.api.types.is_datetime64_any_dtype(factors_df["timestamp"]):
        factors_df["timestamp"] = factors_df["timestamp"].dt.strftime("%Y-%m-%d")
    
    factors_df.to_json(feature_file, orient="records", lines=True)
    print(f"Saved feature data to {feature_file}")
    
    # Load symbol info from configs/_asset_list_/exp.json
    asset_list_path = os.path.join(root, "configs", "_asset_list_", "exp.json")
    with open(asset_list_path) as f:
        asset_info_all = json.load(f)
    
    aapl_info = asset_info_all.get("AAPL", {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "exchange": "NASDAQ Global Select",
        "currency": "USD"
    })
    
    # Update meta_info.json
    meta = {
        "symbols": ["AAPL"],
        "symbols_info": {
            "AAPL": aapl_info
        },
        "tags": [
            price_tag,
            feature_tag
        ]
    }
    
    meta_path = os.path.join(exp_dir, "meta_info.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {meta_path} successfully!")

if __name__ == "__main__":
    main()
