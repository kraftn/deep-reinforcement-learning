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
    parser.add_argument("--num_trajectories_per_group", default=100, type=int)
    args = parser.parse_args()

    file_name2total_rewards = dict()
    for file_name in args.file_names:
        file_path = os.path.join(args.directory, file_name)
        file_name2total_rewards[file_name] = read_rewards(file_path)

    plt.figure(figsize=(10, 7.5))
    for file_name, total_rewards in file_name2total_rewards.items():
        if len(total_rewards) % args.num_trajectories_per_group != 0:
            raise RuntimeError(
                "Number of trajectories isn't divisible by num_trajectories_per_group"
            )
        n_points = len(total_rewards) // args.num_trajectories_per_group

        trajectory_indices = [
            args.num_trajectories_per_group * i_point
            + args.num_trajectories_per_group // 2
            for i_point in range(n_points)
        ]
        total_rewards = np.mean(
            total_rewards.reshape((-1, args.num_trajectories_per_group)),
            axis=1
        )

        plt.plot(trajectory_indices, total_rewards, label=file_name)
    plt.legend()
    plt.show()
