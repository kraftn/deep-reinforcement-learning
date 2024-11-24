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
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = np.ones((self.n_states, self.n_actions)) / self.n_actions

    def get_action(self, state):
        action = np.random.choice(self.n_actions, p=self.model[state])
        return action

    def fit(self, trajectories, q_param, smoothing_lambda=0.0):
        new_model = np.zeros((self.n_states, self.n_actions))
        elite_trajectories = self.extract_elite_trajectories(trajectories, q_param)

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state, action] += 1

        new_model += smoothing_lambda

        for state in range(self.n_states):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model

    @staticmethod
    def extract_elite_trajectories(trajectories, q_param):
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        quantile = np.quantile(total_rewards, q=q_param)

        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory["rewards"])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        return elite_trajectories

    def save_model(self, file_path):
        np.save(file_path, self.model)

    def load_model(self, file_path):
        model = np.load(file_path)
        if model.shape != (self.n_states, self.n_actions):
            raise Exception("Wrong shape of model")
        self.model = model

    def reset_model(self):
        self.model = np.ones((self.n_states, self.n_actions)) / self.n_actions


class PolicySmoothingCrossEntropyAgent(CrossEntropyAgent):
    def fit(self, trajectories, q_param, smoothing_lambda=1.0):
        old_model = self.model.copy()
        super().fit(trajectories, q_param)
        self.model = smoothing_lambda * self.model + (1 - smoothing_lambda) * old_model


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


def train_agent(env, agent, q_param, n_iterations, n_trajectories, smoothing_lambda):
    logger.info(
        f"Training agent with the following hyperparameters: q_param = {q_param}; "
        f"n_trajectories = {n_trajectories}; smoothing_lambda = {smoothing_lambda}; "
        f"n_iterations = {n_iterations}"
    )
    agent.reset_model()

    average_rewards = []
    for _ in tqdm(range(n_iterations), desc="Training agent"):
        # policy evaluation
        trajectories = [get_trajectory(env, agent) for _ in range(n_trajectories)]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        average_rewards.append(np.mean(total_rewards))
        # policy improvement
        agent.fit(trajectories, q_param, smoothing_lambda)

    return average_rewards


def search_params(env, agent, params_grid, n_iterations):
    logger.info("Searching hyperparameters")

    params_combinations = [dict()]
    for param_name, param_values in params_grid.items():
        updated_combinations = []
        for param_value in param_values:
            for combination in params_combinations:
                combination = combination.copy()
                combination[param_name] = param_value
                updated_combinations.append(combination)
        params_combinations = updated_combinations

    params2average_rewards = []
    for params_combination in params_combinations:
        average_rewards = train_agent(
            env,
            agent,
            n_iterations=n_iterations,
            **params_combination
        )
        params2average_rewards.append((params_combination, average_rewards))

    best_params, average_rewards = max(
        params2average_rewards, key=lambda item: np.max(item[1])
    )
    logger.info(f"Max average reward = {np.max(average_rewards)}")
    best_n_iterations = np.argmax(average_rewards) + 1
    best_params["n_iterations"] = best_n_iterations

    best_params_str = "; ".join(
        f"{param_name} = {param_value}"
        for param_name, param_value in best_params.items()
    )
    logger.info(f"The best hyperparameters: {best_params_str}")

    return best_params


def make_plots(env, agent, params_grid, best_params):
    logger.info("Making plots")

    for param_name, param_values in params_grid.items():
        for param_value in param_values:
            params = best_params.copy()
            params[param_name] = param_value
            average_rewards = train_agent(
                env, agent, **params
            )

            plt.plot(
                np.arange(len(average_rewards)) + 1,
                average_rewards,
                label=f"{param_name} = {param_value}"
            )

        plt.xlabel("Iteration")
        plt.ylabel("Average reward")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["search", "compare"])
    args = parser.parse_args()

    env = gym.make("Taxi-v3")

    if args.mode == "search":
        params_grid = {
            "q_param": [0.3],
            "n_trajectories": [1000],
            "smoothing_lambda": None
        }

        for agent, smoothing_lambda in zip(
            [
                CrossEntropyAgent(N_STATES, N_ACTIONS),
                PolicySmoothingCrossEntropyAgent(N_STATES, N_ACTIONS)
            ],
            [np.logspace(-2, 1, 5), np.linspace(0.5, 0.95, 5)]
        ):
            params_grid["smoothing_lambda"] = smoothing_lambda
            best_params = search_params(env, agent, params_grid, n_iterations=35)
            make_plots(env, agent, {"smoothing_lambda": smoothing_lambda}, best_params)
    elif args.mode == "compare":
        n_iterations = 35
        average_rewards = train_agent(
            env, CrossEntropyAgent(N_STATES, N_ACTIONS), 0.3, n_iterations, 1000, 0.0
        )
        iterations = np.arange(n_iterations) + 1

        for settings in [
            {
                "smoothing_type": "laplace smoothing",
                "agent": CrossEntropyAgent(N_STATES, N_ACTIONS),
                "smoothing_lambda": 0.05
            },
            {
                "smoothing_type": "policy smoothing",
                "agent": PolicySmoothingCrossEntropyAgent(N_STATES, N_ACTIONS),
                "smoothing_lambda": 0.9
            }
        ]:
            smooth_average_rewards = train_agent(
                env,
                settings["agent"],
                0.3,
                n_iterations,
                1000,
                settings["smoothing_lambda"]
            )
            plt.plot(iterations, average_rewards, label="without smoothing")
            plt.plot(
                iterations, smooth_average_rewards, label=settings["smoothing_type"]
            )
            plt.xlabel("Iteration")
            plt.ylabel("Average reward")
            plt.legend()
            plt.show()
    else:
        raise NotImplementedError
