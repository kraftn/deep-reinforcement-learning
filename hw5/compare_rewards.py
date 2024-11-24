import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def read_rewards(file_path: str) -> np.ndarray:
    total_rewards = []
    with open(file_path, "r", encoding="utf-8") as history_file:
        for line in history_file:
            trajectory = json.loads(line)
            total_rewards.append(sum(trajectory["rewards"]))
    return np.array(total_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_names", nargs="+")
    parser.add_argument("--directory", default="histories")
    parser.add_argument("--window_size", default=101, type=int)
    parser.add_argument("--max_episodes_num", default=None, type=int)
    args = parser.parse_args()
    if args.window_size % 2 == 0:
        raise RuntimeError("Even window_size was specified")

    file_name2total_rewards = dict()
    for file_name in args.file_names:
        file_path = os.path.join(args.directory, file_name)
        total_rewards = read_rewards(file_path)
        if args.max_episodes_num is not None:
            total_rewards = total_rewards[:args.max_episodes_num]
        file_name2total_rewards[file_name] = total_rewards

    plt.figure(figsize=(10, 7.5))
    for file_name, total_rewards in file_name2total_rewards.items():
        mean_total_rewards = np.lib.stride_tricks.sliding_window_view(
            total_rewards, window_shape=args.window_size,
        ).mean(axis=1)
        trajectory_indices = np.arange(
            args.window_size // 2, total_rewards.shape[0] - args.window_size // 2
        )
        plt.plot(trajectory_indices, mean_total_rewards, label=file_name)
    plt.legend()
    plt.show()
