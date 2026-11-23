import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tensordict import TensorDict
from finworld.config import config
from finworld.registry import AGENT, DATASET

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained FinWorld PPO model")
    parser.add_argument("--config", default="configs/rl_trading/ppo/AAPL_ppo_trading.py", help="Config file")
    parser.add_argument("--checkpoint", default="workdir/AAPL_trading_ppo/checkpoint/00000002.pth", help="Model checkpoint path")
    args = parser.parse_args()

    config.init_config(args.config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    agent = AGENT.build(config.agent).to(device)

    if not os.path.exists(args.checkpoint):
        # Pick the latest available checkpoint in checkpoint dir
        ckpt_dir = config.checkpoint_path
        ckpts = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        args.checkpoint = ckpts[-1]

    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(checkpoint)
    agent.eval()

    print("Loading latest market data...")
    dataset = DATASET.build(config.dataset)
    window = config.history_timestamps # 64

    features = dataset.asset_data["features"].iloc[-window:].values
    times = dataset.asset_data["times"].iloc[-window:].values
    dates = dataset.asset_data["features"].index[-window:]
    latest_date = dates[-1]

    cashes = np.ones((window,), dtype=np.float32) * float(config.initial_amount)
    positions = np.zeros((window,), dtype=np.float32)
    actions = np.ones((window,), dtype=np.int32)
    rets = np.zeros((window,), dtype=np.float32)

    input_tensor = TensorDict({
        "dense": torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device),
        "sparse": torch.tensor(times, dtype=torch.int32).unsqueeze(0).to(device),
        "cashes": torch.tensor(cashes, dtype=torch.float32).unsqueeze(0).to(device),
        "positions": torch.tensor(positions, dtype=torch.int32).unsqueeze(0).to(device),
        "actions": torch.tensor(actions, dtype=torch.int32).unsqueeze(0).to(device),
        "rets": torch.tensor(rets, dtype=torch.float32).unsqueeze(0).to(device),
    }, batch_size=1).to(device)

    with torch.no_grad():
        logits = agent.actor(input_tensor)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        action_id = int(torch.argmax(logits, dim=-1).item())

    action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}

    print("\n" + "="*50)
    print(f" Asset:               {config.symbol}")
    print(f" Latest Data Date:    {latest_date}")
    print(f" Action Probabilities:")
    print(f"   BUY  :  {probs[0]*100:6.2f}%")
    print(f"   HOLD :  {probs[1]*100:6.2f}%")
    print(f"   SELL :  {probs[2]*100:6.2f}%")
    print("-" * 50)
    print(f" Recommended Action:  >>> {action_names[action_id]} <<<")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
