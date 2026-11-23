# FinWorld Project Documentation & Changelog

This document details all modifications, bug fixes, architectural enhancements, and algorithms implemented in **FinWorld** (`TradeMaster-NTU/FinWorld`) on Windows with NVIDIA RTX 4050 GPU acceleration.

---

## 1. System & Platform Compatibility Fixes

When running FinWorld on Windows with Python 3.11, several platform and package incompatibilities were encountered and resolved:

### A. NumPy 2.x & Gym 0.26 Compatibility
- **Issue**: Gym 0.26 accesses `np.bool8`, which was deprecated and removed in NumPy 2.x (`AttributeError: module 'numpy' has no attribute 'bool8'`).
- **Fix**: Added an alias patch in [`finworld/__init__.py`](file:///c:/Users/praty/FinWorld/finworld/__init__.py):
  ```python
  import numpy as np
  if not hasattr(np, 'bool8'):
      np.bool8 = np.bool_
  ```

### B. Alpha158 Factor Generator Iteration Bug
- **Issue**: In [`finworld/factor/alpha158.py`](file:///c:/Users/praty/FinWorld/finworld/factor/alpha158.py), line 31 had `for window in windows or windows:`. When `windows` was `None`, this caused `TypeError: 'NoneType' object is not iterable`.
- **Fix**: Updated line 31 to iterate over `self.windows` (`for window in self.windows:`), correctly falling back to default windows `[5, 10, 20, 30, 60]`.

### C. WandB Non-Interactive Login Guard
- **Issue**: [`finworld/log/wandb.py`](file:///c:/Users/praty/FinWorld/finworld/log/wandb.py) crashed with `KeyError: 'WANDB_API_KEY'` when running headless without environment credentials.
- **Fix**: Wrapped the login block in `if os.environ.get("WANDB_API_KEY"):` and set `self.is_main_process = False` otherwise to gracefully bypass WandB logging when an API key is not supplied.

### D. Missing Dependencies & Safe Registry Imports
- **Issue**: Dynamic registry loading in `finworld/registry.py` failed if optional modules (e.g. `scikit-learn`, `browser_use`, `lightgbm`) were missing.
- **Fix**:
  - Installed `scikit-learn` in the conda environment.
  - Added safe `try...except ImportError` guards in [`finworld/tools/auto_browser.py`](file:///c:/Users/praty/FinWorld/finworld/tools/auto_browser.py), [`finworld/tools/custom_browser.py`](file:///c:/Users/praty/FinWorld/finworld/tools/custom_browser.py), [`finworld/tools/browser/controller.py`](file:///c:/Users/praty/FinWorld/finworld/tools/browser/controller.py), and [`finworld/agent/ml/lightgbm.py`](file:///c:/Users/praty/FinWorld/finworld/agent/ml/lightgbm.py).

### E. Windows Multiprocessing VectorEnv Failure
- **Issue**: Gym's `AsyncVectorEnv` uses process spawning on Windows, causing pipe serialization crashes (`TypeError: cannot unpack non-iterable NoneType object`).
- **Fix**: Replaced `gym.vector.AsyncVectorEnv` with `gym.vector.SyncVectorEnv` in both [`finworld/trainer/rl/ppo_trading_trainer.py`](file:///c:/Users/praty/FinWorld/finworld/trainer/rl/ppo_trading_trainer.py) and [`finworld/trainer/rl/ppo_portfolio_trainer.py`](file:///c:/Users/praty/FinWorld/finworld/trainer/rl/ppo_portfolio_trainer.py).

### F. Checkpoint Directory Auto-Creation
- **Issue**: `torch.save(...)` failed with `FileNotFoundError` if the checkpoint folder did not exist.
- **Fix**: Added `os.makedirs(self.config.checkpoint_path, exist_ok=True)` in trainer initializers.

### G. Validation Record Array Length Alignment
- **Issue**: In `ppo_portfolio_trainer.py` and `meta_portfolio_trainer.py`, `portfolio_records.to_dataframe()` threw `ValueError: Length of values (469) does not match length of index (470)` because `timestamp` was recorded at reset ($N+1$ entries) while actions/returns were recorded only at step ($N$ entries).
- **Fix**: Appended the terminal record upon environment completion to ensure all fields have equal length ($470$).

---

## 2. Data Engineering & Multi-Asset Pipeline

### A. Free YFinance Pipeline
Financial Modeling Prep (FMP) was hitting free-tier API rate limits. Created [`prepare_portfolio_data.py`](file:///c:/Users/praty/FinWorld/prepare_portfolio_data.py):
1. **Asset Selection**: 4 high-liquidity US equities (`AAPL`, `AMZN`, `GOOGL`, `MSFT`).
2. **Date Alignment**: Downloaded 10 years of data (2015-05-01 to 2025-05-01), computed the exact intersection of common trading dates (2,516 days), ensuring zero missing rows (a strict requirement of FinWorld's `MultiAssetDataset`).
3. **Alpha158 Factor Mining**: Generated 145 technical features per asset (moving average crosses, momentum, volatility bands, volume volume dynamics).
4. **Data Registry**: Output prices to `datasets/exp/exp_yfinance_price_1day/<SYMBOL>.jsonl` and features to `datasets/exp/exp_yfinance_feature_1day/<SYMBOL>.jsonl`, registering tags in [`datasets/exp/meta_info.json`](file:///c:/Users/praty/FinWorld/datasets/exp/meta_info.json).

---

## 3. Baseline PPO Portfolio Optimization

### A. Configuration
Created [`configs/rl_portfolio/ppo/exp_ppo_portfolio.py`](file:///c:/Users/praty/FinWorld/configs/rl_portfolio/ppo/exp_ppo_portfolio.py):
- **Action Space**: Continuous 5-dimensional Dirichlet distribution `[Cash, AAPL, AMZN, GOOGL, MSFT]`.
- **Model**: Transformer encoder (4 layers, 4 heads, 256 embedding dim, 7.4M parameters).
- **Split**: Train (2015–2023), Validation (2023–2025).

### B. Trainer Enhancement
Added `test()` method to [`finworld/trainer/rl/ppo_portfolio_trainer.py`](file:///c:/Users/praty/FinWorld/finworld/trainer/rl/ppo_portfolio_trainer.py).

### C. Visualization & Inference
- Created [`plot_portfolio.py`](file:///c:/Users/praty/FinWorld/plot_portfolio.py): Generates 3-panel performance plot at [`workdir/exp_portfolio_ppo/plot/portfolio_performance.png`](file:///c:/Users/praty/FinWorld/workdir/exp_portfolio_ppo/plot/portfolio_performance.png).
- Created [`predict_portfolio.py`](file:///c:/Users/praty/FinWorld/predict_portfolio.py): Computes target weight distribution and dollar allocation for custom capital.

---

## 4. Context-Based Meta-Reinforcement Learning (Meta-RL)

To address market non-stationarity and regime shifts (e.g. high volatility crashes vs. momentum expansions), a complete **Context-Based Meta-RL architecture** was designed and implemented from scratch:

```
Trajectory Context [r_t, a_t, pos_t, cash_t]
                  │
                  ▼
   ┌─────────────────────────────┐
   │    MarketContextEncoder     │ (1D Temporal CNN + Statistical Summary)
   └──────────────┬──────────────┘
                  │ Latent Regime Context Vector z ∈ ℝ⁶⁴
                  ▼
   ┌─────────────────────────────┐
   │    FiLM Context Layer       │  h_mod = γ(z) ⊙ h_state + β(z)
   └──────────────┬──────────────┘
                  ▼
         ┌─────────────────┐
         │    MetaActor    │ ──► Continuous Dirichlet Allocation [Cash, Stocks]
         └─────────────────┘
```

### Files Created for Meta-RL:
1. [`finworld/models/meta/context_encoder.py`](file:///c:/Users/praty/FinWorld/finworld/models/meta/context_encoder.py):
   - `MarketContextEncoder`: Processes rolling returns, downside semi-volatility, drawdown proxies, and allocation history into latent context vector $z \in \mathbb{R}^{64}$.
   - `FiLMContextModulation`: Feature-wise Linear Modulation layer dynamically scaling and shifting latent states based on $z$.
2. [`finworld/models/meta/__init__.py`](file:///c:/Users/praty/FinWorld/finworld/models/meta/__init__.py): Module export.
3. [`finworld/agent/rl/meta_ppo.py`](file:///c:/Users/praty/FinWorld/finworld/agent/rl/meta_ppo.py):
   - `MetaActor`: Context-modulated Transformer policy.
   - `MetaCritic`: Context-modulated value network $V(s, z)$.
   - `MetaPPO`: Agent registered in `finworld.registry.AGENT`.
4. [`finworld/trainer/rl/meta_portfolio_trainer.py`](file:///c:/Users/praty/FinWorld/finworld/trainer/rl/meta_portfolio_trainer.py):
   - Trainer registered in `finworld.registry.TRAINER` providing joint optimization of policy, context encoder, and critic.
5. [`configs/rl_portfolio/meta_ppo/exp_meta_ppo_portfolio.py`](file:///c:/Users/praty/FinWorld/configs/rl_portfolio/meta_ppo/exp_meta_ppo_portfolio.py):
   - Configuration file for Meta-RL training.
6. [`plot_meta_comparison.py`](file:///c:/Users/praty/FinWorld/plot_meta_comparison.py):
   - Comparative visualizer generating [`workdir/exp_portfolio_meta_ppo/plot/meta_vs_baseline_comparison.png`](file:///c:/Users/praty/FinWorld/workdir/exp_portfolio_meta_ppo/plot/meta_vs_baseline_comparison.png).
7. [`predict_meta_portfolio.py`](file:///c:/Users/praty/FinWorld/predict_meta_portfolio.py):
   - Dedicated inference tool outputting inferred market regime metrics alongside recommended allocations.

---

## 5. Performance Benchmark: PPO vs. Context Meta-RL

Evaluated over the exact same out-of-sample validation period (May 2023 – May 2025, 470 trading days, \$100,000 initial capital):

| Metric | Baseline PPO | Context-Based Meta-RL | Difference |
| :--- | :--- | :--- | :--- |
| **Total Net Return** | **+24.32%** | **+34.16%** | **+9.84% excess return** |
| **Final Portfolio Value** | \$124,320.26 | **\$134,163.50** | **+\$9,843.24 net profit** |
| **Annualized Return (ARR)** | 12.38% | **17.06%** | **+4.68% / year** |
| **Max Drawdown (MDD)** | -12.35% | **-11.77%** | **Lower downside risk** |
| **Sharpe Ratio (SR)** | 1.07 | **1.27** | Improved risk-adjusted return |
| **Sortino Ratio (SOR)** | 1.54 | **1.85** | Superior upside capture |
| **Calmar Ratio (ARR / MDD)**| 1.00 | **1.45** | 45% higher return per unit of drawdown |

---

## 6. Lookahead Bias & Data Integrity Verification

A strict code audit confirmed zero lookahead bias:
- `if_use_future = False` in all configs.
- Observation windows strictly slide backward 64 days without peeking forward.
- Alpha158 uses positive backward shifts only (`shift(window)` with $window > 0$).
- `WindowedScaler` normalizes each chunk using the *prior* chunk's statistics.
- Train and validation splits are strictly chronological (May 2015 – May 2023 for training; May 2023 – May 2025 for testing).

---

## 7. Command Reference

### Run Meta-RL Allocation Inference:
```powershell
& "C:\Users\praty\miniconda3\envs\finworld\python.exe" predict_meta_portfolio.py 100000
```

### Run Baseline PPO Allocation Inference:
```powershell
& "C:\Users\praty\miniconda3\envs\finworld\python.exe" predict_portfolio.py 100000
```

### Generate Comparative Visualizations:
```powershell
& "C:\Users\praty\miniconda3\envs\finworld\python.exe" plot_meta_comparison.py
```
