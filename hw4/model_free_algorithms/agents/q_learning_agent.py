import typing as t

import numpy as np

from model_free_algorithms.agents.base_agents import TemporalDifferenceAgent


class QLearningAgent(TemporalDifferenceAgent):
    def _process_episode(
        self, i_episode: int, max_trajectory_length: int
    ) -> t.Dict[str, list]:
        state = self.env.reset()
        epsilon = self.calc_epsilon(i_episode, self.n_episodes)
        trajectory = {"states": [], "actions": [], "rewards": []}

        for i_step in range(max_trajectory_length):
            action_dist = self.get_action_dist(state, epsilon)
            action = np.random.choice(self.env.num_actions, p=action_dist)
            next_state, reward, done, _ = self.env.step(action)
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            q_values_update_step = (
                reward + self.gamma * self.q_values[next_state].max()
                - self.q_values[state, action]
            )
            self.q_values[state, action] += self.alpha * q_values_update_step
            state = next_state
            if done:
                break

        return trajectory
