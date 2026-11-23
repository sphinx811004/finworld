import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
from tensordict import TensorDict

# Add workspace to path
root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root)

from finworld.config import config
from finworld.registry import DATASET, AGENT

def run_meta_prediction(capital: float = 100000.0):
    config_path = "configs/rl_portfolio/meta_ppo/exp_meta_ppo_portfolio.py"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--cfg-options", default=None)
    args = parser.parse_args([])

    config.init_config(config_path, args)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = DATASET.build(config.dataset)
    symbols = dataset.symbols

    # Load MetaPPO agent
    agent = AGENT.build(config.agent).to(device)

    # Checkpoint
    ckpts = sorted(glob.glob(os.path.join(config.checkpoint_path, "*.pth")))
    if not ckpts:
        ckpts = sorted(glob.glob(os.path.join(config.exp_path, "checkpoint", "*.pth")))
    
    if not ckpts:
        print(f"Error: No checkpoint found in {config.checkpoint_path} or {config.exp_path}/checkpoint.")
        return

    latest_ckpt = ckpts[-1]
    agent.load_state_dict(torch.load(latest_ckpt, map_location=device))
    agent.eval()

    # Build latest market tensor
    window = config.history_timestamps # 64
    feature_list = []
    prices = []
    
    for sym in symbols:
        asset_feat = dataset.assets_data[sym]["features"].iloc[-window:].values
        feature_list.append(asset_feat)
        latest_price = float(dataset.assets_data[sym]["original_prices"].iloc[-1]["close"])
        prices.append(latest_price)

    dense_features = np.stack(feature_list, axis=1) # (window, num_assets, feat_dim)
    dense_tensor = torch.tensor(dense_features, dtype=torch.float32).unsqueeze(0).to(device)
    times = dataset.assets_data[symbols[0]]["times"].iloc[-window:].values
    sparse_tensor = torch.tensor(times, dtype=torch.int32).unsqueeze(0).to(device)

    cashes = torch.ones((1, window), dtype=torch.float32).to(device) * float(config.initial_amount)
    positions = torch.zeros((1, window, len(symbols)), dtype=torch.float32).to(device)
    actions = torch.zeros((1, window, 1 + len(symbols)), dtype=torch.float32).to(device)
    actions[:, :, 0] = 0.5 # 50% cash baseline
    rets = torch.zeros((1, window), dtype=torch.float32).to(device)

    # Estimate proxy step returns over the window from price close
    close_prices = np.stack([dataset.assets_data[sym]["original_prices"].iloc[-window:]["close"].values for sym in symbols], axis=1)
    basket_returns = np.diff(close_prices.mean(axis=1)) / close_prices.mean(axis=1)[:-1]
    rets[0, 1:] = torch.tensor(basket_returns, dtype=torch.float32).to(device)

    input_tensor = TensorDict({
        "dense": dense_tensor,
        "sparse": sparse_tensor,
        "cashes": cashes,
        "positions": positions,
        "actions": actions,
        "rets": rets,
    }, batch_size=1)

    with torch.no_grad():
        # 1. Market Context Encoder inference
        z = agent.extract_context(input_tensor) # (1, context_dim)
        z_norm = float(torch.norm(z).item())
        
        # 2. Context-Conditioned Actor inference
        logits = agent.actor(input_tensor, z)
        alpha = F.softplus(logits) + 1e-3
        expected_weights = (alpha / alpha.sum(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()

    latest_date = str(dataset.assets_data[symbols[0]]["features"].index[-1])[:10]

    # Context regime stats
    mean_ret = float(rets.mean().item()) * 252 * 100 # annualized %
    vol_ret = float(rets.std().item()) * np.sqrt(252) * 100 # annualized %

    print("\n" + "="*75)
    print(" CONTEXT-BASED META-RL PORTFOLIO: LATEST ADAPTIVE ALLOCATION")
    print("="*75)
    print(f"Date:                {latest_date}")
    print(f"Portfolio Capital:   ${capital:,.2f}")
    print(f"Model Checkpoint:    {os.path.basename(latest_ckpt)}")
    print(f"Inferred Regime:     Trend={mean_ret:+.1f}% ARR | Volatility={vol_ret:.1f}% | z_norm={z_norm:.2f}")
    print("="*75)

    print(f"{'Asset':<10} | {'Price ($)':<12} | {'Target Weight':<15} | {'Allocation ($)':<16} | {'Target Shares':<12}")
    print("-" * 75)

    cash_weight = float(expected_weights[0])
    cash_alloc = capital * cash_weight
    print(f"{'CASH':<10} | {'$1.00':<12} | {cash_weight*100:>13.2f}% | ${cash_alloc:>14,.2f} | {'-':<12}")

    for idx, sym in enumerate(symbols):
        w = float(expected_weights[idx + 1])
        alloc = capital * w
        p = prices[idx]
        shares = (alloc / p) if p > 0 else 0
        p_str = f"${p:.2f}" if p > 0 else "N/A"
        print(f"{sym:<10} | {p_str:<12} | {w*100:>13.2f}% | ${alloc:>14,.2f} | {shares:>11.2f}")

    print("-" * 75)
    print(f"{'TOTAL':<10} | {'':<12} | {expected_weights.sum()*100:>13.2f}% | ${capital:>14,.2f} |")
    print("="*75 + "\n")

if __name__ == "__main__":
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 100000.0
    run_meta_prediction(cap)
