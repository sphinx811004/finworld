# download_yf.py  (save in C:\Users\praty\FinWorld, run: python download_yf.py)
import json, os
import yfinance as yf

SYMBOL = "AAPL"
SOURCE = "yfinance"
LEVEL = "1day"
START, END = "1995-05-01", "2025-05-02"   # adjust to match the config
ROOT = os.path.join("datasets", "exp")

tag = f"{SYMBOL}_{SOURCE}_price_{LEVEL}"
out_dir = os.path.join(ROOT, tag)
os.makedirs(out_dir, exist_ok=True)

df = yf.download(SYMBOL, start=START, end=END, auto_adjust=False, progress=False)
if hasattr(df.columns, "levels"):          # newer yfinance returns MultiIndex columns
    df.columns = df.columns.get_level_values(0)
df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
df = df[["open", "high", "low", "close", "volume", "adj_close"]].dropna()
df.insert(0, "timestamp", df.index.strftime("%Y-%m-%d"))

path = os.path.join(out_dir, f"{SYMBOL}.jsonl")
df.to_json(path, orient="records", lines=True)
print("wrote", len(df), "rows to", path)

meta_path = os.path.join(ROOT, "meta_info.json")
meta = {"symbols_info": {}, "tags": []}
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
meta.setdefault("symbols_info", {})[SYMBOL] = {"symbol": SYMBOL, "name": "Apple Inc."}
tags = meta.setdefault("tags", [])
if tag not in tags:
    tags.append(tag)
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("meta_info.json updated, tags:", meta["tags"])