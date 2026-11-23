import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

class MarketContextEncoder(nn.Module):
    """
    Context-Based Meta-RL Market Context Encoder.
    
    Extracts a latent market regime representation z from recent trajectory statistics:
    - Step returns (mean, volatility, downside risk, momentum)
    - Action history (recent asset weights)
    - Position/Cash exposures
    
    The latent regime vector z dynamically conditions the Meta-Actor and Meta-Critic.
    """
    def __init__(self,
                 num_assets: int = 4,
                 history_timestamps: int = 64,
                 context_dim: int = 64,
                 hidden_dim: int = 128):
        super(MarketContextEncoder, self).__init__()
        
        self.num_assets = num_assets
        self.history_timestamps = history_timestamps
        self.context_dim = context_dim
        
        # Action dim = 1 (cash) + num_assets
        self.action_dim = 1 + num_assets
        
        # Per-timestamp features: return (1) + cash (1) + positions (num_assets) + actions (1 + num_assets)
        self.per_step_dim = 1 + 1 + num_assets + self.action_dim
        
        # 1D Temporal Convolution feature extractor for recent sequence
        self.temporal_net = nn.Sequential(
            nn.Conv1d(in_channels=self.per_step_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Statistical summary head: rolling metrics over the window
        # (mean_ret, vol_ret, downside_vol, max_dd, cash_ratio, top_asset_weight) -> 6 summary metrics
        self.stat_fc = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Projection to latent regime embedding z
        self.proj_z = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim)
        )
        
    def _compute_statistical_features(self, rets: torch.Tensor, cashes: torch.Tensor, actions: torch.Tensor):
        """
        Compute invariant statistical properties of the market regime:
        - Mean return
        - Return volatility
        - Downside semi-volatility
        - Max drawdown proxy
        - Mean cash weight
        - Mean equity concentration
        """
        # rets: (B, T)
        mean_ret = rets.mean(dim=-1, keepdim=True)
        vol_ret = rets.std(dim=-1, keepdim=True) + 1e-6
        
        # Downside semi-volatility
        downside = torch.clamp(rets - mean_ret, max=0.0)
        downside_vol = torch.sqrt((downside ** 2).mean(dim=-1, keepdim=True) + 1e-6)
        
        # Cumulative return & max drawdown proxy
        cum_ret = torch.cumsum(rets, dim=-1)
        running_max = torch.cummax(cum_ret, dim=-1)[0]
        dd = (running_max - cum_ret).max(dim=-1, keepdim=True)[0]
        
        # Cash & equity behavior from actions (B, T, action_dim)
        cash_ratio = actions[..., 0].mean(dim=-1, keepdim=True)
        equity_weights = actions[..., 1:]
        max_equity_conc = equity_weights.max(dim=-1)[0].mean(dim=-1, keepdim=True)
        
        stat_feats = torch.cat([mean_ret, vol_ret, downside_vol, dd, cash_ratio, max_equity_conc], dim=-1)
        return stat_feats

    def forward(self, x: TensorDict) -> torch.Tensor:
        """
        Args:
            x (TensorDict): contains 'rets', 'cashes', 'positions', 'actions'
        Returns:
            z (torch.Tensor): Latent market regime context vector of shape (B, context_dim)
        """
        rets = x["rets"] # (B, T)
        cashes = x["cashes"] # (B, T)
        positions = x["positions"] # (B, T, num_assets)
        actions = x["actions"] # (B, T, 1 + num_assets)
        
        batch_size = rets.shape[0]
        
        # Normalize cash relative to initial if needed
        norm_cashes = cashes / (cashes.max(dim=-1, keepdim=True)[0] + 1e-6)
        
        # Normalize positions relative to sum
        pos_sum = positions.sum(dim=-1, keepdim=True) + 1e-6
        norm_positions = positions / pos_sum
        
        # Construct sequence features: (B, T, per_step_dim)
        seq_features = torch.cat([
            rets.unsqueeze(-1),
            norm_cashes.unsqueeze(-1),
            norm_positions,
            actions
        ], dim=-1)
        
        # (B, per_step_dim, T) for Conv1D
        conv_input = seq_features.transpose(1, 2)
        h_temp = self.temporal_net(conv_input).squeeze(-1) # (B, hidden_dim)
        
        # Statistical features
        stat_feats = self._compute_statistical_features(rets, cashes, actions) # (B, 6)
        h_stat = self.stat_fc(stat_feats) # (B, hidden_dim // 2)
        
        # Fused regime representation
        fused = torch.cat([h_temp, h_stat], dim=-1)
        z = self.proj_z(fused) # (B, context_dim)
        
        return z

class FiLMContextModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioned on market regime z.
    Modulates representation h:
        h_modulated = gamma(z) * h + beta(z)
    """
    def __init__(self, context_dim: int, feature_dim: int):
        super(FiLMContextModulation, self).__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.Sigmoid() # Scale factor between 0 and 2
        )
        self.beta_net = nn.Sequential(
            nn.Linear(context_dim, feature_dim)
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: (B, context_dim), h: (B, feature_dim)
        gamma = self.gamma_net(z) * 2.0
        beta = self.beta_net(z)
        
        h_mod = gamma * h + beta
        return self.layer_norm(h_mod)
