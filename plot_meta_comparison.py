import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_records(records_path):
    if not os.path.exists(records_path):
        return None
    records = []
    with open(records_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if len(records) > 2 and records[-1].get("value") == records[0].get("value"):
        records = records[:-1]
    return records

def plot_comparison():
    base_path = "workdir/exp_portfolio_ppo/valid_records.jsonl"
    meta_path = "workdir/exp_portfolio_meta_ppo/valid_records.jsonl"
    
    base_records = load_records(base_path)
    meta_records = load_records(meta_path)
    
    if meta_records is None:
        print(f"Waiting for {meta_path} to be generated...")
        return

    symbols = ["AAPL", "AMZN", "GOOGL", "MSFT"]
    asset_labels = ["Cash"] + symbols
    
    meta_vals = np.array([r.get("value", np.nan) for r in meta_records], dtype=np.float64)
    meta_initial = meta_vals[0]
    meta_final = meta_vals[-1]
    meta_ret = ((meta_final - meta_initial) / meta_initial) * 100.0
    
    days = np.arange(len(meta_vals))
    
    # Meta actions (norm weights)
    meta_actions = np.array([r.get("action", [0]*5) for r in meta_records], dtype=np.float64)
    row_sums = meta_actions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    meta_weights = (meta_actions / row_sums) * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), gridspec_kw={'height_ratios': [2.2, 1.8, 2.0]})
    plt.subplots_adjust(hspace=0.30)
    
    # ==== Subplot 1: Portfolio Value Growth ====
    ax1 = axes[0]
    if base_records is not None:
        base_vals = np.array([r.get("value", np.nan) for r in base_records], dtype=np.float64)
        base_ret = ((base_vals[-1] - base_vals[0]) / base_vals[0]) * 100.0
        ax1.plot(days[:len(base_vals)], base_vals, color="#7f7f7f", linestyle="--", linewidth=2.0, label=f"Baseline PPO (+{base_ret:.2f}%)")
    
    ax1.plot(days, meta_vals, color="#2ca02c", linewidth=2.5, label=f"Context Meta-RL (+{meta_ret:.2f}%)")
    ax1.axhline(meta_initial, color="gray", linestyle=":", alpha=0.6, label="Initial Capital ($100,000)")
    ax1.set_title(
        f"Portfolio Optimization: Context-Based Meta-RL vs. Baseline PPO\n"
        f"Meta-RL Final Value: ${meta_final:,.2f} | Net Return: +{meta_ret:.2f}%",
        fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.yaxis.set_major_formatter('${x:,.0f}')
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.set_xlim(0, len(days)-1)
    
    # ==== Subplot 2: Drawdown Comparison ====
    ax2 = axes[1]
    meta_peaks = np.maximum.accumulate(meta_vals)
    meta_dd = (meta_vals - meta_peaks) / meta_peaks * 100.0
    
    if base_records is not None:
        base_peaks = np.maximum.accumulate(base_vals)
        base_dd = (base_vals - base_peaks) / base_peaks * 100.0
        ax2.plot(days[:len(base_vals)], base_dd, color="#7f7f7f", linestyle="--", alpha=0.7, label=f"Baseline PPO (Max DD: {base_dd.min():.2f}%)")
        
    ax2.plot(days, meta_dd, color="#d62728", linewidth=1.8, label=f"Meta-RL (Max DD: {meta_dd.min():.2f}%)")
    ax2.fill_between(days, meta_dd, 0, color="#d62728", alpha=0.15)
    ax2.set_title("Drawdown Profile (Downside Risk Management)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.yaxis.set_major_formatter('{x:.1f}%')
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.set_xlim(0, len(days)-1)
    
    # ==== Subplot 3: Meta-RL Dynamic Allocation Weights ====
    ax3 = axes[2]
    palette = ["#7f7f7f", "#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd"]
    ax3.stackplot(days, meta_weights.T, labels=asset_labels, colors=palette, alpha=0.85)
    ax3.set_title("Meta-RL Context-Adaptive Allocation Weights (% of Portfolio)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Allocation (%)", fontsize=11)
    ax3.set_xlabel("Trading Day (Validation Period)", fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.yaxis.set_major_formatter('{x:.0f}%')
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="lower left", ncol=5, frameon=True, fontsize=10)
    ax3.set_xlim(0, len(days)-1)
    
    out_dir = "workdir/exp_portfolio_meta_ppo/plot"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "meta_vs_baseline_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {out_file}")
    
    # Print comparison table
    print("\n" + "="*60)
    print(" PERFORMANCE COMPARISON: BASELINE PPO vs. CONTEXT META-RL")
    print("="*60)
    if base_records is not None:
        print(f"Baseline PPO:  Return: +{base_ret:.2f}% | Max DD: {base_dd.min():.2f}%")
    print(f"Context Meta-RL: Return: +{meta_ret:.2f}% | Max DD: {meta_dd.min():.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    plot_comparison()
