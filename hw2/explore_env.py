import argparse
import time

import gym


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("--max_length", default=1000, type=int)
    parser.add_argument("--time_interval", default=2, type=int)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    state = env.reset(seed=int(time.time()))
    print(f"State: {state}")
    env.render()

    action, total_reward = None, 0.0
    for i_step in range(args.max_length):
        if i_step % args.time_interval == 0:
            action = int(input())

        state, reward, done, _ = env.step(action)
        print(f"State: {state}")
        total_reward += reward

        env.render()

        if done:
            break

    print(f"Total reward: {total_reward:.2f}")
