import logging
import time
import argparse

import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
N_STATES = 500
N_ACTIONS = 6


class CrossEntropyAgent:
    def __init__(self, n_states, n_actions, is_stochastic_env):
        self.n_states = n_states
        self.n_actions = n_actions
        self.is_stochastic_env = is_stochastic_env
        self.model = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.deterministic_model = None

    def get_action(self, state):
        if self.is_stochastic_env and self.deterministic_model is None:
            raise RuntimeError("Determine model before calling get_action")
        if self.is_stochastic_env:
            model = self.deterministic_model
        else:
            model = self.model
        action = np.random.choice(self.n_actions, p=model[state])
        return action

    def fit(self, trajectories, q_param, smoothing_lambda):
        new_model = np.zeros((self.n_states, self.n_actions))
        elite_trajectories = self.extract_elite_trajectories(trajectories, q_param)

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state, action] += 1

        row_sums = new_model.sum(axis=1)
        new_model[row_sums == 0, :] = self.model[row_sums == 0, :]
        row_sums = new_model.sum(axis=1, keepdims=True)
        new_model /= row_sums

        new_model = smoothing_lambda * new_model + (1 - smoothing_lambda) * self.model

        self.reset_model(new_model)

    @staticmethod
    def extract_elite_trajectories(trajectories, q_param):
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        quantile = np.quantile(total_rewards, q=q_param)

        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory["rewards"])
            if total_reward >= quantile:
                elite_trajectories.append(trajectory)

        return elite_trajectories

    def save_model(self, file_path):
        np.save(file_path, self.model)

    def load_model(self, file_path):
        model = np.load(file_path)
        if model.shape != (self.n_states, self.n_actions):
            raise Exception("Wrong shape of model")
        self.reset_model(model)

    def reset_model(self, new_model=None):
        if new_model is None:
            new_model = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.model = new_model
        self.deterministic_model = None

    def determine_model(self):
        deterministic_model = np.zeros_like(self.model)
        n_states = self.model.shape[0]
        thresholds = np.random.rand(n_states, 1)
        actions_indices = (self.model.cumsum(axis=1) > thresholds).argmax(axis=1)
        deterministic_model[np.arange(n_states), actions_indices] = 1.0
        self.deterministic_model = deterministic_model


def get_trajectory(env, agent, max_length=1000, visualize=False):
    trajectory = {"states": [], "actions": [], "rewards": []}
    state = env.reset()
    if visualize:
        env.render()
        time.sleep(0.5)

    for _ in range(max_length):
        trajectory["states"].append(state)

        action = agent.get_action(state)
        trajectory["actions"].append(action)

        state, reward, done, _ = env.step(action)
        trajectory["rewards"].append(reward)

        if visualize:
            env.render()
            time.sleep(0.5)

        if done:
            break

    return trajectory


def train_agent(
    env,
    agent,
    q_param,
    n_iterations,
    n_trajectories,
    n_deterministic_models,
    smoothing_lambda,
):
    if not agent.is_stochastic_env and n_deterministic_models > 1:
        raise ValueError("n_deterministic_models > 1 in deterministic environment")

    logger.info(
        f"Training agent with the following hyperparameters: q_param = {q_param}; "
        f"n_trajectories = {n_trajectories}; n_iterations = {n_iterations}; "
        f"n_deterministic_models = {n_deterministic_models}; "
        f"smoothing_lambda = {smoothing_lambda}"
    )
    agent.reset_model()

    average_rewards = []
    for i_iteration in range(n_iterations):
        i_model2trajectories = []
        for _ in tqdm(range(n_deterministic_models), desc="Determine models"):
            if agent.is_stochastic_env:
                agent.determine_model()
            i_model2trajectories.append(
                [get_trajectory(env, agent) for _ in range(n_trajectories)]
            )

        i_model2average_reward = []
        for model_trajectories in i_model2trajectories:
            i_model2average_reward.append(
                np.mean(
                    [np.sum(trajectory["rewards"]) for trajectory in model_trajectories]
                )
            )
        logger.info(f"Min model average reward: {min(i_model2average_reward):.2f}")
        logger.info(f"Max model average reward: {max(i_model2average_reward):.2f}")

        average_rewards.append(np.mean(i_model2average_reward))
        logger.info(
            f"Iteration: {i_iteration + 1}; Average reward: {average_rewards[-1]:.2f}"
        )

        threshold_reward = np.quantile(i_model2average_reward, q_param)
        trajectories = []
        for model_trajectories, model_reward in zip(
                i_model2trajectories, i_model2average_reward
        ):
            if model_reward >= threshold_reward:
                trajectories.extend(model_trajectories)

        if agent.is_stochastic_env:
            agent.fit(trajectories, q_param=0.0, smoothing_lambda=smoothing_lambda)
        else:
            agent.fit(trajectories, q_param=q_param, smoothing_lambda=smoothing_lambda)

    return average_rewards


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "run"])
    parser.add_argument("--model_file_path", required=False, default="model.npy")
    args = parser.parse_args()

    env = gym.make("Taxi-v3")

    if args.mode == "train":
        n_iterations = 50
        deterministic_average_rewards = train_agent(
            env,
            CrossEntropyAgent(N_STATES, N_ACTIONS, is_stochastic_env=False),
            0.3,
            n_iterations,
            1000,
            n_deterministic_models=1,
            smoothing_lambda=1.0
        )
        agent = CrossEntropyAgent(N_STATES, N_ACTIONS, is_stochastic_env=True)
        stochastic_average_rewards = train_agent(
            env,
            agent,
            0.7,
            n_iterations,
            10,
            n_deterministic_models=10000,
            smoothing_lambda=0.9
        )
        agent.save_model("model.npy")

        iterations = np.arange(n_iterations) + 1
        plt.plot(
            iterations, deterministic_average_rewards, label="deterministic env"
        )
        plt.plot(
            iterations, stochastic_average_rewards, label="stochastic env"
        )
        plt.xlabel("Iteration")
        plt.ylabel("Average reward")
        plt.legend()
        plt.show()
    elif args.mode == "run":
        env.reset(seed=int(time.time()))
        agent = CrossEntropyAgent(N_STATES, N_ACTIONS, is_stochastic_env=True)
        agent.load_model(args.model_file_path)
        trajectory = get_trajectory(env, agent, max_length=50, visualize=True)
        logger.info(f"total reward: {np.sum(trajectory['rewards'])}")
    else:
        raise NotImplementedError
