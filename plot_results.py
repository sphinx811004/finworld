import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot trading performance from valid_records.jsonl")
    parser.add_argument("--records", default="workdir/AAPL_trading_ppo/valid_records.jsonl", help="Path to valid_records.jsonl")
    parser.add_argument("--output", default="workdir/AAPL_trading_ppo/plot/trading_performance.png", help="Path to save the plot")
    args = parser.parse_args()

    if not os.path.exists(args.records):
        print(f"File not found: {args.records}")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_json(args.records, lines=True)

    # Trim any environment reset rows at the end
    if len(df) > 2 and df["value"].iloc[-1] == df["cash"].iloc[0]:
        df = df.iloc[:-2].copy()

    initial_cash = df["cash"].iloc[0]
    final_value = df["value"].iloc[-1]
    total_profit_dollar = final_value - initial_cash
    total_return_pct = (total_profit_dollar / initial_cash) * 100

    initial_price = df["price"].iloc[0]
    final_price = df["price"].iloc[-1]
    shares = initial_cash / initial_price
    benchmark_value = shares * df["price"]
    benchmark_return_pct = ((final_price - initial_price) / initial_price) * 100

    # Max Drawdown calculation
    cum_max = np.maximum.accumulate(df["value"])
    drawdown = (df["value"] - cum_max) / cum_max * 100
    max_drawdown = np.min(drawdown)

    buys = df[df["action_label"] == "BUY"]
    sells = df[df["action_label"] == "SELL"]
    holds = df[df["action_label"] == "HOLD"]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})

    # Top chart: Equity Curve vs Buy & Hold Benchmark
    ax1.plot(df.index, df["value"], label=f"FinWorld PPO Strategy (+{total_return_pct:.1f}%)", color="#1f77b4", linewidth=2.2)
    ax1.plot(df.index, benchmark_value, label=f"Buy & Hold Benchmark (+{benchmark_return_pct:.1f}%)", color="#7f7f7f", linestyle="--", linewidth=1.5, alpha=0.8)

    # Plot Buy/Sell signals on the equity curve
    ax1.scatter(buys.index, df.loc[buys.index, "value"], marker="^", color="#2ca02c", s=60, label=f"BUY Signal ({len(buys)})", zorder=5)
    ax1.scatter(sells.index, df.loc[sells.index, "value"], marker="v", color="#d62728", s=60, label=f"SELL Signal ({len(sells)})", zorder=5)

    ax1.set_title("FinWorld PPO Trading Performance Evaluation (AAPL 2020 - 2025)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Portfolio Value ($ USD)", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Info summary box on top chart
    stats_text = (
        f"Initial Capital:  ${initial_cash:,.0f}\n"
        f"Final Value:      ${final_value:,.0f}\n"
        f"Strategy Profit:  +${total_profit_dollar:,.0f} (+{total_return_pct:.2f}%)\n"
        f"Max Drawdown:     {max_drawdown:.2f}%\n"
        f"Trades Taken:     {len(buys)} Buys, {len(sells)} Sells"
    )
    ax1.text(0.98, 0.05, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9))

    # Bottom chart: Asset Close Price with Execution Points
    ax2.plot(df.index, df["price"], label="AAPL Close Price ($)", color="#ff7f0e", linewidth=1.5)
    ax2.scatter(buys.index, df.loc[buys.index, "price"], marker="^", color="#2ca02c", s=50, zorder=5)
    ax2.scatter(sells.index, df.loc[sells.index, "price"], marker="v", color="#d62728", s=50, zorder=5)

    ax2.set_xlabel("Trading Days (Validation Horizon)", fontsize=11)
    ax2.set_ylabel("AAPL Price ($)", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Chart saved successfully: {args.output}")

if __name__ == "__main__":
    main()
