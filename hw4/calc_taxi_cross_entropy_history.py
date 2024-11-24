import argparse
import os.path

import gym

from hw1.practice1_1 import CrossEntropyAgent, train_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_files_dir", default="histories")
    args = parser.parse_args()

    env = gym.make("Taxi-v3")
    agent = CrossEntropyAgent(n_states=500, n_actions=6)
    n_iterations = 35
    n_trajectories = 1000
    n_episodes = n_iterations * n_trajectories

    history_file_path = os.path.join(
        args.history_files_dir, "Taxi-v3_CrossEntropyAgent.jsonl"
    )
    train_agent(
        env, agent,
        q_param=0.3,
        n_iterations=n_iterations,
        n_trajectories=n_trajectories,
        history_file_path=history_file_path,
    )
