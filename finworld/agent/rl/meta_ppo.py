import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from einops import rearrange
from diffusers.utils.accelerate_utils import apply_forward_hook

from finworld.registry import AGENT, EMBED, ENCODER
from finworld.models.base import Model
from finworld.models.modules.mean_pool import MeanPool1D
from finworld.models.modules.attention_pool import AttentionPool1D
from finworld.models.embed import DenseLinearEmbed
from finworld.models.embed.position import SinCosPosition2DEmbed, SinCosPosition1DEmbed
from finworld.models.meta.context_encoder import MarketContextEncoder, FiLMContextModulation
from finworld.task import TaskType

class MetaActor(Model):
    """
    Actor network conditioned on latent market regime context z.
    """
    def __init__(self,
                 task_type: str,
                 embed_config: dict,
                 encoder_config: dict,
                 action_dim: int,
                 output_dim: int,
                 context_dim: int = 64,
                 pool_type: str = 'avg',
                 **kwargs):
        super(MetaActor, self).__init__(**kwargs)
        
        self.task_type = TaskType.from_string(task_type)
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        self.embed = EMBED.build(embed_config)
        
        if self.task_type == TaskType.TRADING:
            self.pos_embed = SinCosPosition1DEmbed(
                num_positions=embed_config["history_timestamps"] // embed_config["patch_timestamps"],
                embed_dim=embed_config["output_dim"],
                num_prefix=0
            )
        elif self.task_type == TaskType.PORTFOLIO:
            self.pos_embed = SinCosPosition2DEmbed(
                num_time=embed_config["history_timestamps"] // embed_config["patch_timestamps"],
                num_space=embed_config["num_assets"],
                embed_dim=embed_config["output_dim"],
                num_prefix=0
            )
            
        self.encoder = ENCODER.build(encoder_config)
        
        if pool_type == 'avg':
            self.pool = MeanPool1D(dim=1, keepdim=False)
        else:
            self.pool = AttentionPool1D(
                in_features=encoder_config['output_dim'],
                out_features=output_dim,
                embed_dim=encoder_config['latent_dim'],
                num_heads=encoder_config['num_heads']
            )
            
        # Context-based FiLM modulation
        self.context_modulation = FiLMContextModulation(
            context_dim=context_dim,
            feature_dim=encoder_config['output_dim']
        )
        
        # Policy output head
        self.decoder = nn.Sequential(
            nn.Linear(encoder_config['output_dim'], encoder_config['output_dim'] // 2),
            nn.GELU(),
            nn.Linear(encoder_config['output_dim'] // 2, output_dim)
        )
        
        self.initialize_weights()

    @apply_forward_hook
    def encode(self, x: TensorDict):
        h = self.embed(x)
        pos = self.pos_embed(h)
        h = h + pos
        h, _, _ = self.encoder(h)
        return h

    def forward(self, x: TensorDict, z: torch.Tensor):
        batch_size = x.batch_size
        
        if len(batch_size) == 2:
            x = TensorDict(
                {key: rearrange(val, 'b e ... -> (b e) ...') for key, val in x.items()},
                batch_size=batch_size[0] * batch_size[1]
            )
            z = rearrange(z, 'b e ... -> (b e) ...')
            
        h = self.encode(x)
        h = self.pool(h)
        
        # Modulate by market regime context z
        h_mod = self.context_modulation(h, z)
        logits = self.decoder(h_mod)
        
        if len(batch_size) == 2:
            logits = rearrange(logits, '(b e) ... -> b e ...', b=batch_size[0])
            
        return logits

class MetaCritic(Model):
    """
    Critic network estimating state-value V(s, z) conditioned on market regime z.
    """
    def __init__(self,
                 task_type: str,
                 embed_config: dict,
                 encoder_config: dict,
                 action_dim: int,
                 output_dim: int,
                 context_dim: int = 64,
                 pool_type: str = 'avg',
                 **kwargs):
        super(MetaCritic, self).__init__(**kwargs)
        
        self.task_type = TaskType.from_string(task_type)
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        self.embed = EMBED.build(embed_config)
        
        if self.task_type == TaskType.TRADING:
            self.pos_embed = SinCosPosition1DEmbed(
                num_positions=embed_config["history_timestamps"] // embed_config["patch_timestamps"],
                embed_dim=embed_config["output_dim"],
                num_prefix=0
            )
        elif self.task_type == TaskType.PORTFOLIO:
            self.pos_embed = SinCosPosition2DEmbed(
                num_time=embed_config["history_timestamps"] // embed_config["patch_timestamps"],
                num_space=embed_config["num_assets"],
                embed_dim=embed_config["output_dim"],
                num_prefix=0
            )
            
        self.encoder = ENCODER.build(encoder_config)
        self.pool = MeanPool1D(dim=1, keepdim=False)
        
        self.context_modulation = FiLMContextModulation(
            context_dim=context_dim,
            feature_dim=encoder_config['output_dim']
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoder_config['output_dim'], encoder_config['output_dim'] // 2),
            nn.GELU(),
            nn.Linear(encoder_config['output_dim'] // 2, output_dim)
        )
        
        self.initialize_weights()

    @apply_forward_hook
    def encode(self, x: TensorDict):
        h = self.embed(x)
        pos = self.pos_embed(h)
        h = h + pos
        h, _, _ = self.encoder(h)
        return h

    def forward(self, x: TensorDict, z: torch.Tensor):
        batch_size = x.batch_size
        
        if len(batch_size) == 2:
            x = TensorDict(
                {key: rearrange(val, 'b e ... -> (b e) ...') for key, val in x.items()},
                batch_size=batch_size[0] * batch_size[1]
            )
            z = rearrange(z, 'b e ... -> (b e) ...')
            
        h = self.encode(x)
        h = self.pool(h)
        
        h_mod = self.context_modulation(h, z)
        val = self.decoder(h_mod).squeeze(-1)
        
        if len(batch_size) == 2:
            val = rearrange(val, '(b e) ... -> b e ...', b=batch_size[0])
            
        return val

@AGENT.register_module(force=True)
class MetaPPO(nn.Module):
    """
    Context-Based Meta-RL PPO Agent for Portfolio Optimization.
    """
    def __init__(self,
                 *args,
                 task_type: str,
                 embed_config: dict,
                 encoder_config: dict,
                 action_dim: int,
                 actor_output_dim: int,
                 critic_output_dim: int,
                 context_dim: int = 64,
                 **kwargs):
        super(MetaPPO, self).__init__()
        
        self.task_type = TaskType.from_string(task_type)
        self.action_dim = action_dim
        self.actor_output_dim = actor_output_dim
        self.critic_output_dim = critic_output_dim
        self.context_dim = context_dim
        
        num_assets = embed_config.get("num_assets", action_dim - 1)
        history_timestamps = embed_config.get("history_timestamps", 64)
        
        # 1. Market Context Encoder
        self.context_encoder = MarketContextEncoder(
            num_assets=num_assets,
            history_timestamps=history_timestamps,
            context_dim=context_dim
        )
        
        # 2. Context-Conditioned Actor
        self.actor = MetaActor(
            task_type=task_type,
            embed_config=embed_config,
            encoder_config=encoder_config,
            action_dim=action_dim,
            output_dim=actor_output_dim,
            context_dim=context_dim
        )
        
        # 3. Context-Conditioned Critic
        self.critic = MetaCritic(
            task_type=task_type,
            embed_config=embed_config,
            encoder_config=encoder_config,
            action_dim=action_dim,
            output_dim=critic_output_dim,
            context_dim=context_dim
        )

    def extract_context(self, x: TensorDict) -> torch.Tensor:
        """Extract latent market regime context z from trajectory state"""
        batch_size = x.batch_size
        if len(batch_size) == 2:
            flat_x = TensorDict(
                {k: rearrange(v, 'b e ... -> (b e) ...') for k, v in x.items()},
                batch_size=batch_size[0] * batch_size[1]
            )
            z = self.context_encoder(flat_x)
            z = rearrange(z, '(b e) ... -> b e ...', b=batch_size[0])
            return z
        return self.context_encoder(x)

    def get_action(self, x: TensorDict):
        z = self.extract_context(x)
        logits = self.actor(x, z)
        
        if self.task_type == TaskType.TRADING:
            dis = Categorical(logits=logits)
            return dis.sample()
        elif self.task_type == TaskType.PORTFOLIO:
            alpha = F.softplus(logits) + 1e-3
            dist = Dirichlet(alpha)
            return dist.rsample()

    def get_value(self, x: TensorDict, a: torch.Tensor = None):
        z = self.extract_context(x)
        return self.critic(x, z)

    def get_action_and_value(self, x: TensorDict, a: torch.Tensor = None):
        batch_size = x.batch_size
        z = self.extract_context(x)
        
        logits = self.actor(x, z)
        value = self.critic(x, z)
        
        if self.task_type == TaskType.TRADING:
            dis = Categorical(logits=logits)
            if a is None:
                a = dis.sample()
            logprob = dis.log_prob(a)
            entropy = dis.entropy()
            return a, logprob, entropy, value
        elif self.task_type == TaskType.PORTFOLIO:
            alpha = F.softplus(logits) + 1e-3
            dist = Dirichlet(alpha)
            if a is None:
                a = dist.rsample()
            # Clamp action slightly inside (0, 1) to avoid log(0) numerical instability
            eps = 1e-6
            safe_a = torch.clamp(a, min=eps, max=1.0 - eps)
            safe_a = safe_a / safe_a.sum(dim=-1, keepdim=True)
            logprob = dist.log_prob(safe_a)
            entropy = dist.entropy()
            return a, logprob, entropy, value
