import copy
import json
import os.path
import typing as t
from logging import getLogger

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from sac.dataclasses import TrainArguments, Experiment
from sac.agent import SacAgent
from sac.utils import Memory, Network

logger = getLogger(__name__)


class SacTrainer:
    """
    The code is based on the ideas of the paper https://arxiv.org/pdf/1910.07207
    """

    def __init__(self, train_args: TrainArguments):
        self.train_args = train_args

    def train(
        self,
        agent: SacAgent,
        env: gym.Env,
        history_file_path: t.Optional[str] = None
    ) -> t.Tuple[t.List[t.List[float]], t.List[float], t.List[t.Dict[str, t.Any]]]:
        agent.reset()
        memory = Memory(agent.device)
        experiment = self._prepare_experiment(agent, env)

        history, policy_losses = [], []
        q_value_losses = [[] for _ in range(self.train_args.num_q_networks)]

        for i_episode in range(self.train_args.num_episodes):
            logger.info(f"Episode №{i_episode + 1}")
            if experiment.train_alpha:
                logger.info(f"Alpha = {experiment.log_alpha.exp().item():.2f}")

            curr_state = env.reset()
            trajectory = {"rewards": []}

            for i_trajectory_step in range(self.train_args.max_trajectory_length):
                action, next_state, reward, done = self._take_env_step(
                    agent, env, curr_state
                )
                trajectory["rewards"].append(reward)
                memory.add_sample(
                    curr_state.tolist(), action, reward, next_state.tolist(), done
                )
                curr_state = next_state

                if len(memory) < self.train_args.batch_size:
                    continue
                batch_q_value_losses, batch_policy_loss = self._train_on_batch(
                    experiment,
                    memory.get_batch(self.train_args.batch_size),
                )
                for batch_q_value_loss, network_q_value_losses in zip(
                    batch_q_value_losses, q_value_losses
                ):
                    network_q_value_losses.append(batch_q_value_loss)
                policy_losses.append(batch_policy_loss)

                if done:
                    break

            logger.info(f"Total reward: {sum(trajectory['rewards']):.2f}")
            history.append(trajectory)

        if history_file_path is not None:
            self._save_history(history, history_file_path)
        return q_value_losses, policy_losses, history

    def _prepare_experiment(
        self, agent: SacAgent, env: gym.Env
    ) -> Experiment:
        q_networks = [
            Network(
                [
                    env.observation_space.shape[0],
                    *self.train_args.q_network_hidden_sizes,
                    env.action_space.n
                ]
            ) for _ in range(self.train_args.num_q_networks)
        ]
        target_q_networks = [copy.deepcopy(network) for network in q_networks]
        q_network_optimizers = [
            Adam(q_network.parameters(), lr=self.train_args.q_network_lr)
            for q_network in q_networks
        ]
        policy_network_optimizer = Adam(
            agent.network.parameters(), lr=self.train_args.policy_network_lr
        )

        if self.train_args.alpha is None:
            log_alpha = torch.tensor(0.0, requires_grad=True, device=agent.device)
            alpha_optimizer = Adam([log_alpha], lr=self.train_args.alpha_lr)
            target_entropy = - torch.log(
                torch.tensor(1.0 / env.action_space.n, device=agent.device)
            ) * self.train_args.max_entropy_ratio
            logger.info(
                f"Use trained alpha: target_entropy = {target_entropy:.2f}"
            )
        else:
            log_alpha = torch.log(
                torch.tensor(
                    self.train_args.alpha, requires_grad=False, device=agent.device
                )
            )
            alpha_optimizer = None
            target_entropy = None
            logger.info(f"Use fixed alpha = {self.train_args.alpha}")

        return Experiment(
            agent,
            env,
            q_networks,
            target_q_networks,
            log_alpha,
            policy_network_optimizer,
            q_network_optimizers,
            alpha_optimizer,
            target_entropy,
            train_alpha=self.train_args.alpha is None,
        )

    def _take_env_step(
        self, agent: SacAgent, env: gym.Env, curr_state: np.ndarray
    ) -> t.Tuple[int, np.ndarray, float, bool]:
        action = agent.sample_actions(
            torch.from_numpy(curr_state).to(agent.device).unsqueeze(0)
        ).item()
        next_state, reward, done, _ = env.step(action)
        return action, next_state, reward, done

    def _train_on_batch(
        self,
        exp: Experiment,
        batch: t.Tuple[torch.Tensor, ...],
    ) -> t.Tuple[t.List[float], float]:
        states, actions, rewards, next_states, dones = batch
        target_q_values = self._calc_target_q_values(
            exp, next_states, rewards, dones
        )
        q_values_losses = self._update_q_networks(
            exp, states, actions, target_q_values
        )

        policy_log_probs = exp.agent.calc_log_probs(states, mode="train")
        policy_loss = self._update_policy_network(
            exp, states, policy_log_probs
        )
        if exp.train_alpha:
            self._update_alpha(exp, policy_log_probs)

        self._update_target_q_networks(exp)

        return q_values_losses, policy_loss

    def _calc_target_q_values(
        self,
        exp: Experiment,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # next_states = [batch_size, state_dim]
        # rewards = [batch_size]
        # dones = [batch_size]

        min_next_q_values = self._calc_min_q_values(exp, next_states)
        # min_next_q_values = [batch_size, num_actions]
        policy_log_probs = exp.agent.calc_log_probs(next_states, mode="eval")
        # policy_log_probs = [batch_size, num_actions]
        policy_probs = torch.exp(policy_log_probs)
        # policy_probs = [batch_size, num_actions]

        expectation_over_policy = (
            policy_probs * (
                min_next_q_values - exp.log_alpha.exp().detach() * policy_log_probs
            )
        ).sum(dim=1)
        # expectation_over_policy = [batch_size]

        return (
            rewards + self.train_args.gamma * (1 - dones) * expectation_over_policy
        ).detach()

    def _update_q_networks(
        self,
        exp: Experiment,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_q_values: torch.Tensor,
    ) -> t.List[float]:
        # states = [batch_size, state_dim]
        # actions = [batch_size]
        # target_q_values = [batch_size]

        losses = []
        for q_network, optimizer in zip(exp.q_networks, exp.q_network_optimizers):
            pred_q_values = q_network(states)
            # pred_q_values = [batch_size, num_actions]
            pred_q_values = torch.gather(
                pred_q_values, dim=1, index=actions.unsqueeze(1)
            ).squeeze(1)
            # pred_q_values = [batch_size]
            loss = F.mse_loss(pred_q_values, target_q_values)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses

    def _update_target_q_networks(self, exp: Experiment):
        for target_q_network, q_network in zip(exp.target_q_networks, exp.q_networks):
            param_name2param = dict(q_network.named_parameters())
            for param_name, target_param in target_q_network.named_parameters():
                param = param_name2param[param_name]
                target_param.data.copy_(
                    (1 - self.train_args.q_network_update_tau) * target_param.data + (
                        self.train_args.q_network_update_tau * param.data
                    )
                )

    def _update_policy_network(
        self,
        exp: Experiment,
        states: torch.Tensor,
        policy_log_probs: torch.Tensor,
    ) -> float:
        # states = [batch_size, state_dim]
        # policy_log_probs = [batch_size, num_actions]
        policy_probs = torch.exp(policy_log_probs)
        # policy_probs = [batch_size, num_actions]

        min_q_values = self._calc_min_q_values(exp, states)
        # min_q_values = [batch_size, num_actions]

        losses = (
            policy_probs * (
                exp.log_alpha.exp().detach() * policy_log_probs - min_q_values
            )
        ).sum(dim=1)
        # losses = [batch_size]
        loss = losses.mean(dim=0)

        exp.policy_network_optimizer.zero_grad()
        loss.backward()
        exp.policy_network_optimizer.step()

        return loss.item()

    def _update_alpha(
        self,
        exp: Experiment,
        policy_log_probs: torch.Tensor,
    ):
        # policy_log_probs = [batch_size, num_actions]
        policy_log_probs = policy_log_probs.detach()
        policy_probs = torch.exp(policy_log_probs)

        entropy = - (policy_probs * policy_log_probs).sum(dim=1)
        logger.debug(f"Mean entropy: {entropy.mean(dim=0).item():.2f}")
        loss = (exp.log_alpha.exp() * (entropy - exp.target_entropy)).mean(dim=0)

        exp.alpha_optimizer.zero_grad()
        loss.backward()
        exp.alpha_optimizer.step()

    def _calc_min_q_values(
        self,
        exp: Experiment,
        states: torch.Tensor,
    ) -> torch.Tensor:
        # states = [batch_size, state_dim]
        # actions = [batch_size, num_actions]
        q_values_per_network = []
        for q_network in exp.q_networks:
            q_network.eval()
            with torch.no_grad():
                q_values = q_network(states)
                # q_values = [batch_size, num_actions]
            q_values_per_network.append(q_values.unsqueeze(0))
        min_q_values = torch.min(
            torch.concatenate(q_values_per_network, dim=0), dim=0
        ).values
        # min_q_values = [batch_size, num_actions]
        return min_q_values

    @staticmethod
    def _save_history(history: t.Sequence[t.Dict[str, t.Any]], history_file_path: str):
        dir_path = os.path.dirname(history_file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(history_file_path, "w", encoding="utf-8") as history_file:
            for trajectory_history in history:
                history_file.write(json.dumps(trajectory_history) + "\n")
