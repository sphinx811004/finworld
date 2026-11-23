import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_portfolio_performance():
    records_path = "workdir/exp_portfolio_ppo/valid_records.jsonl"
    if not os.path.exists(records_path):
        print(f"Error: {records_path} does not exist.")
        return

    records = []
    with open(records_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Drop the last record if it resets to initial value (terminal "done" record)
    if len(records) > 2 and records[-1].get("value") == records[0].get("value"):
        records = records[:-1]

    symbols = ["AAPL", "AMZN", "GOOGL", "MSFT"]
    asset_labels = ["Cash"] + symbols
    n = len(records)
    
    # Extract values, prices, positions, actions, cash
    values = []
    cashes = []
    all_actions = []
    all_prices = []
    
    for r in records:
        values.append(r.get("value", np.nan))
        cashes.append(r.get("cash", np.nan))
        act = r.get("action")
        if isinstance(act, list) and len(act) == 5:
            all_actions.append(act)
        else:
            all_actions.append([np.nan]*5)
        p = r.get("price")
        if isinstance(p, list) and len(p) == 4:
            all_prices.append(p)
        else:
            all_prices.append([np.nan]*4)
    
    values = np.array(values, dtype=np.float64)
    cashes = np.array(cashes, dtype=np.float64)
    actions = np.array(all_actions, dtype=np.float64)  # shape (n, 5): Cash, AAPL, AMZN, GOOGL, MSFT
    prices = np.array(all_prices, dtype=np.float64)     # shape (n, 4): AAPL, AMZN, GOOGL, MSFT

    # Compute x-axis as trading days
    days = np.arange(n)

    initial_val = values[0]
    final_val = values[-1]
    total_ret = ((final_val - initial_val) / initial_val) * 100.0
    profit = final_val - initial_val

    # Normalize weights to sum to 100%
    row_sums = actions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_weights = (actions / row_sums) * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), gridspec_kw={'height_ratios': [2, 2, 2]})
    plt.subplots_adjust(hspace=0.30)

    # ==== Subplot 1: Portfolio Value ====
    ax1 = axes[0]
    ax1.plot(days, values, color="#1f77b4", linewidth=2.0, label=f"PPO Portfolio (Return: {total_ret:+.2f}%)")
    ax1.axhline(initial_val, color="gray", linestyle="--", alpha=0.7, label=f"Initial Capital (${initial_val:,.0f})")
    ax1.set_title(
        f"Multi-Asset PPO Portfolio Optimization (Validation Period)\n"
        f"Final Value: ${final_val:,.2f} | Net Profit: ${profit:+,.2f} ({total_ret:+.2f}%)",
        fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_xlim(0, n-1)

    # ==== Subplot 2: Dynamic Asset Allocation Weights (Stacked Area) ====
    ax2 = axes[1]
    palette = ["#7f7f7f", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    ax2.stackplot(days, norm_weights.T, labels=asset_labels, colors=palette, alpha=0.85)
    ax2.set_title("Dynamic Asset Allocation Weights (% of Portfolio)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Allocation (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter('{x:.0f}%')
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", ncol=5, frameon=True, fontsize=10)
    ax2.set_xlim(0, n-1)

    # ==== Subplot 3: Normalized Performance (Base 100) ====
    ax3 = axes[2]
    # Portfolio
    norm_port = (values / initial_val) * 100.0
    ax3.plot(days, norm_port, color="#1f77b4", linewidth=2.0, label="PPO Dynamic Portfolio")
    # Individual assets (buy & hold from initial price)
    colors_bh = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for idx, sym in enumerate(symbols):
        p_col = prices[:, idx]
        if np.isnan(p_col[0]) or p_col[0] == 0:
            continue
        norm_s = (p_col / p_col[0]) * 100.0
        ax3.plot(days, norm_s, linestyle="--", alpha=0.7, color=colors_bh[idx], label=f"{sym} (Buy & Hold)")
    ax3.set_title("Relative Performance: PPO Portfolio vs. Individual Assets (Base = 100)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Normalized Growth", fontsize=11)
    ax3.set_xlabel("Trading Day (Validation Period)", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", ncol=3, frameon=True, fontsize=9)
    ax3.set_xlim(0, n-1)

    out_dir = "workdir/exp_portfolio_ppo/plot"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "portfolio_performance.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Portfolio performance plot saved to: {out_file}")

    # Print summary stats
    print(f"\n--- Validation Summary ---")
    print(f"Initial Capital:  ${initial_val:,.2f}")
    print(f"Final Value:      ${final_val:,.2f}")
    print(f"Net Profit:       ${profit:+,.2f}")
    print(f"Total Return:     {total_ret:+.2f}%")
    
    # Average allocation
    avg_weights = np.nanmean(norm_weights, axis=0)
    print(f"\nAverage Allocation:")
    for i, label in enumerate(asset_labels):
        print(f"  {label:>6s}: {avg_weights[i]:.1f}%")

if __name__ == "__main__":
    plot_portfolio_performance()
