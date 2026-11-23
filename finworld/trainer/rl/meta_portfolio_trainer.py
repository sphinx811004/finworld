import torch
from torch import nn
import gym
import torch.optim as optim
import time
import numpy as np
import os
from copy import deepcopy
from tensordict import TensorDict
from einops import rearrange

from finworld.registry import TRAINER, ENVIRONMENT
from finworld.environment import make_env
from finworld.utils import build_storage, PortfolioRecords
from finworld.log import logger

@TRAINER.register_module(force=True)
class MetaPortfolioTrainer():
    """
    Context-Based Meta-RL Portfolio Trainer.
    Optimizes a MetaPPO policy with latent market regime context conditioning.
    """
    def __init__(self,
                 config = None,
                 dataset = None,
                 agent = None,
                 metrics = None,
                 device = None,
                 dtype = None,
                 **kwargs):
        self.config = config
        self.dataset = dataset

        train_environment_config = deepcopy(config.train_environment)
        train_environment_config.update({"dataset": dataset})
        self.train_environment = ENVIRONMENT.build(train_environment_config)

        valid_environment_config = deepcopy(config.valid_environment)
        valid_environment_config.update({"dataset": dataset})
        self.valid_environment = ENVIRONMENT.build(valid_environment_config)

        test_environment_config = deepcopy(config.test_environment)
        test_environment_config.update({"dataset": dataset})
        self.test_environment = ENVIRONMENT.build(test_environment_config)

        env_name = self.train_environment.__class__.__name__

        self.train_environments = gym.vector.SyncVectorEnv([
            make_env(env_name, env_params=dict(env=deepcopy(self.train_environment),
                                                transition_shape=config.transition_shape, seed=config.seed + i))
            for i in range(config.num_envs)
        ])

        self.valid_environments = gym.vector.SyncVectorEnv([
            make_env(env_name, env_params=dict(env=deepcopy(self.valid_environment),
                                                transition_shape=config.transition_shape, seed=config.seed + i))
            for i in range(1)
        ])

        self.test_environments = gym.vector.SyncVectorEnv([
            make_env(env_name, env_params=dict(env=deepcopy(self.test_environment),
                                                transition_shape=config.transition_shape, seed=config.seed + i))
            for i in range(1)
        ])

        self.metrics = metrics
        self.agent = agent

        # Optimize Actor and MarketContextEncoder together
        actor_and_context_params = list(agent.actor.parameters())
        if hasattr(agent, "context_encoder"):
            actor_and_context_params += list(agent.context_encoder.parameters())

        self.policy_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, actor_and_context_params),
            lr=config.policy_learning_rate, eps=1e-5, weight_decay=1e-4
        )
        self.value_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, list(agent.critic.parameters())),
            lr=config.value_learning_rate, eps=1e-5
        )

        self.device = device
        self.dtype = dtype

        self.global_step = 0
        self.check_index = 0
        self.save_index = 0

        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            os.makedirs(self.config.checkpoint_path, exist_ok=True)

        self.storage = self.set_storage()
        self.use_bc = getattr(self.config, "use_bc", False)

    def __str__(self):
        return f"MetaPortfolioTrainer(task_type={self.config.task_type})"

    def __repr__(self):
        return self.__str__()

    def set_storage(self):
        transition = self.config.transition
        transition_shape = self.config.transition_shape

        storage = TensorDict({}, batch_size=self.config.num_steps).to(self.device)
        for name in transition:
            assert name in transition_shape
            shape = (self.config.num_steps, *transition_shape[name]["shape"])
            dtype_str = transition_shape[name]["type"]
            dtype = torch.float32 if dtype_str == "float32" else torch.int32
            storage[name] = torch.zeros(shape, dtype=dtype).to(self.device)
        return storage

    def flatten_storage(self, storage):
        flat_storage = TensorDict({}, batch_size=self.config.batch_size).to(self.device)
        for key in storage.keys():
            flat_storage[key] = rearrange(storage[key], 's e ... -> (s e) ...')
        return flat_storage

    def explore_environment(self,
                            next_obs: TensorDict,
                            next_done: torch.Tensor,
                            portfolio_records: PortfolioRecords,
                            info: dict):
        for step in range(0, self.config.num_steps):
            self.global_step += 1 * self.config.num_envs

            with torch.no_grad():
                input_tensor = TensorDict(
                    {
                        "dense": next_obs["features"],
                        "sparse": next_obs["times"].to(torch.int32),
                        "cashes": next_obs["policy_portfolio_cashes"],
                        "positions": next_obs["policy_portfolio_positions"].to(torch.float32),
                        "actions": next_obs["policy_portfolio_actions"].to(torch.float32),
                        "rets": next_obs["policy_portfolio_rets"],
                    }, batch_size=next_obs.batch_size
                ).to(next_obs.device)

                action, logprob, _, value = self.agent.get_action_and_value(input_tensor)
                self.storage["training_values"][step] = value.flatten()

            for key, val in next_obs.items():
                self.storage[key][step] = val
            self.storage["training_dones"][step] = next_done
            self.storage["training_actions"][step] = action
            self.storage["training_logprobs"][step] = logprob

            # Step environment
            next_obs_raw, reward, done, truncated, info = self.train_environments.step(action.cpu().numpy())

            portfolio_records.add(
                dict(
                    timestamp=info["timestamp"][0],
                    price=info["price"][0],
                    cash=info["cash"][0],
                    position=info["position"][0],
                    value=info["value"][0],
                    action=info["action"][0],
                    ret=info["ret"][0],
                    total_profit=info["total_profit"][0],
                ),
            )

            self.storage["training_rewards"][step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs = TensorDict({key: torch.Tensor(v) for key, v in next_obs_raw.items()}, batch_size=self.config.num_envs).to(self.device)
            next_done = torch.Tensor(done).to(self.device)

            if "final_info" in info:
                for info_item in info["final_info"]:
                    if info_item is not None:
                        logger.info(f"| global_step={self.global_step}, total_return={info_item['total_return']:.4f}, total_profit = {info_item['total_profit']:.4f}")
                        logger.log_metric({
                            "charts/total_return": info_item["total_return"],
                            "charts/total_profit": info_item["total_profit"],
                        })

        # Bootstrap value if not done
        with torch.no_grad():
            last_tensor = TensorDict(
                {
                    "dense": next_obs["features"],
                    "sparse": next_obs["times"].to(torch.int32),
                    "cashes": next_obs["policy_portfolio_cashes"],
                    "positions": next_obs["policy_portfolio_positions"].to(torch.float32),
                    "actions": next_obs["policy_portfolio_actions"].to(torch.float32),
                    "rets": next_obs["policy_portfolio_rets"],
                }, batch_size=next_obs.batch_size
            ).to(next_obs.device)
            next_value = self.agent.get_value(last_tensor).reshape(1, -1)
            advantages = torch.zeros_like(self.storage["training_rewards"]).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.storage["training_dones"][t + 1]
                    nextvalues = self.storage["training_values"][t + 1]
                delta = self.storage["training_rewards"][t] + self.config.gamma * nextvalues * nextnonterminal - self.storage["training_values"][t]
                advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.storage["training_values"]

        self.storage["training_advantages"] = advantages
        self.storage["training_returns"] = returns

        return next_obs, next_done, portfolio_records, info

    def update_value(self, flat_storage, b_inds, info):
        value_update_steps = info["value_update_steps"]
        v_loss = info["v_loss"]

        self.value_optimizer.zero_grad()
        for start in range(0, self.config.batch_size, self.config.value_minibatch_size):
            value_update_steps += 1
            end = start + self.config.value_minibatch_size
            mb_inds = b_inds[start:end]

            input_tensor = TensorDict({
                "dense": flat_storage["features"][mb_inds],
                "sparse": flat_storage["times"][mb_inds].to(torch.int32),
                "cashes": flat_storage["policy_portfolio_cashes"][mb_inds],
                "positions": flat_storage["policy_portfolio_positions"][mb_inds].to(torch.float32),
                "actions": flat_storage["policy_portfolio_actions"][mb_inds].to(torch.float32),
                "rets": flat_storage["policy_portfolio_rets"][mb_inds],
            }, batch_size=self.config.value_minibatch_size).to(self.device)

            newvalue = self.agent.get_value(input_tensor)
            v_loss = 0.5 * ((newvalue - flat_storage["training_returns"][mb_inds]) ** 2).mean()
            loss = v_loss / (self.config.batch_size // self.config.value_minibatch_size)
            loss.backward()

        nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()
        info["v_loss"] = v_loss
        info["value_update_steps"] = value_update_steps
        return info

    def update_policy(self, flat_storage, b_inds, info):
        policy_update_steps = info["policy_update_steps"]
        clipfracs = info["clipfracs"]
        total_approx_kl = info["total_approx_kl"]

        self.policy_optimizer.zero_grad()
        for start in range(0, self.config.batch_size, self.config.policy_minibatch_size):
            policy_update_steps += 1
            end = start + self.config.policy_minibatch_size
            mb_inds = b_inds[start:end]

            input_tensor = TensorDict({
                "dense": flat_storage["features"][mb_inds],
                "sparse": flat_storage["times"][mb_inds].to(torch.int32),
                "cashes": flat_storage["policy_portfolio_cashes"][mb_inds],
                "positions": flat_storage["policy_portfolio_positions"][mb_inds].to(torch.float32),
                "actions": flat_storage["policy_portfolio_actions"][mb_inds].to(torch.float32),
                "rets": flat_storage["policy_portfolio_rets"][mb_inds],
            }, batch_size=self.config.policy_minibatch_size).to(self.device)

            input_actions = flat_storage["training_actions"][mb_inds].to(torch.float32)
            _, newlogprob, entropy, _ = self.agent.get_action_and_value(input_tensor, input_actions)

            logratio = newlogprob - flat_storage["training_logprobs"][mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                total_approx_kl += approx_kl / self.config.gradient_checkpointing_steps
                clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

            mb_advantages = flat_storage["training_advantages"][mb_inds]
            if self.config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            loss = pg_loss - self.config.ent_coef * entropy_loss
            loss = loss / (self.config.batch_size // self.config.policy_minibatch_size)
            loss.backward()

        # Clip norms for Actor and ContextEncoder
        trainable_params = list(self.agent.actor.parameters()) + list(self.agent.context_encoder.parameters())
        nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
        self.policy_optimizer.step()

        info["pg_loss"] = pg_loss
        info["entropy_loss"] = entropy_loss
        info["total_approx_kl"] = total_approx_kl
        info["clipfracs"] = clipfracs
        info["policy_update_steps"] = policy_update_steps
        return info

    def train(self):
        start_time = time.time()
        obs_raw, info = self.train_environments.reset()
        next_obs = TensorDict({key: torch.Tensor(v) for key, v in obs_raw.items()}, batch_size=self.config.num_envs).to(self.device)
        next_done = torch.zeros(self.config.num_envs).to(self.device)
        num_updates = self.config.total_steps // self.config.batch_size

        for update in range(1, num_updates + 1):
            portfolio_records = PortfolioRecords()
            next_obs, next_done, portfolio_records, info = self.explore_environment(
                next_obs=next_obs, next_done=next_done, portfolio_records=portfolio_records, info=info
            )

            flat_storage = self.flatten_storage(self.storage)
            b_inds = np.arange(self.config.batch_size)

            info_update = {
                "value_update_steps": 0, "v_loss": torch.tensor(0.0),
                "policy_update_steps": 0, "clipfracs": [], "kl_explode": False,
                "pg_loss": torch.tensor(0.0), "entropy_loss": torch.tensor(0.0),
                "total_approx_kl": 0, "logprob": torch.tensor(0.0),
            }

            for epoch in range(self.config.update_epochs):
                np.random.shuffle(b_inds)
                info_update = self.update_value(flat_storage, b_inds, info_update)
                info_update = self.update_policy(flat_storage, b_inds, info_update)

            sps = self.global_step / (time.time() - start_time + 1e-6)
            logger.info(f"| Step: {self.global_step}/{self.config.total_steps} | SPS: {sps:.1f} | PG Loss: {info_update['pg_loss']:.4f} | V Loss: {info_update['v_loss']:.4f}")

            if self.global_step % self.config.check_steps >= self.check_index:
                self.valid()
                self.check_index += 1

            if self.global_step % self.config.save_steps >= self.save_index:
                torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(self.global_step // self.config.save_steps)))
                self.save_index += 1

        self.valid()
        torch.save(self.agent.state_dict(), os.path.join(self.config.checkpoint_path, "{:08d}.pth".format(self.global_step // self.config.save_steps + 1)))

        self.train_environments.close()
        self.valid_environments.close()
        self.test_environments.close()

    def valid(self):
        mode = "valid"
        portfolio_records = PortfolioRecords()
        state, info = self.valid_environments.reset()

        portfolio_records.add(
            dict(
                timestamp=info["timestamp"][0], price=info["price"][0], cash=info["cash"][0],
                position=info["position"][0], value=info["value"][0],
            ),
        )

        next_obs = TensorDict({key: torch.Tensor(value) for key, value in state.items()}, batch_size=1).to(self.device)

        while True:
            with torch.no_grad():
                input_tensor = TensorDict(
                    {
                        "dense": next_obs["features"],
                        "sparse": next_obs["times"].to(torch.int32),
                        "cashes": next_obs["policy_portfolio_cashes"],
                        "positions": next_obs["policy_portfolio_positions"].to(torch.float32),
                        "actions": next_obs["policy_portfolio_actions"].to(torch.float32),
                        "rets": next_obs["policy_portfolio_rets"],
                    }, batch_size=next_obs.batch_size
                ).to(next_obs.device)

                action = self.agent.get_action(input_tensor)

            next_obs, reward, done, truncated, info = self.valid_environments.step(action.cpu().numpy())

            portfolio_records.add(
                dict(
                    action=info["action"][0], ret=info["ret"][0], total_profit=info["total_profit"][0],
                    timestamp=info["timestamp"][0], price=info["price"][0], cash=info["cash"][0],
                    position=info["position"][0], value=info["value"][0],
                ),
            )

            next_obs = TensorDict({key: torch.Tensor(value) for key, value in next_obs.items()}, batch_size=1).to(self.device)

            if "final_info" in info:
                for info_item in info["final_info"]:
                    if info_item is not None:
                        logger.info(f"| Valid Result: total_return={info_item['total_return']:.4f}, total_profit = {info_item['total_profit']:.4f}")
                break

        portfolio_records.add(
            dict(
                action=info["action"][0],
                ret=info["ret"][0],
                total_profit=info["total_profit"][0],
            )
        )

        rets = np.array(portfolio_records.data["ret"], dtype=np.float32)
        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            metrics[metric_name] = metric_fn(ret=rets)

        metrics = {f"{mode}/{k}": v for k, v in metrics.items()}
        logger.log_metric(metrics)
        metrics_string = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"| Valid Metrics: {metrics_string}")

        records_df = portfolio_records.to_dataframe()
        records_df.to_json(os.path.join(self.config.exp_path, f"{mode}_records.jsonl"), orient="records", lines=True)

    def test(self):
        return self.valid()
