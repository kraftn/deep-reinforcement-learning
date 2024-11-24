import typing as t
from logging import getLogger

import gym.core
import numpy as np

logger = getLogger(__name__)


class EnvWrapper:
    def __init__(
        self,
        env: gym.Env,
        min_state: np.ndarray,
        max_state: np.ndarray,
        discretization_grid: t.Sequence[int]
    ):
        self.env = env
        self.min_state = min_state
        self.max_state = max_state
        self.codebooks = np.full(
            (max(discretization_grid), min_state.shape[0]), fill_value=np.inf
        )
        for i_dim, (min_state_element, max_state_element, precision) in enumerate(
            zip(min_state, max_state, discretization_grid)
        ):
            self.codebooks[:precision, i_dim] = np.linspace(
                min_state_element, max_state_element, precision
            )

    def reset(self, seed: t.Optional[int] = None) -> int:
        state = self.env.reset(seed=seed)
        return self._discretize_state(state)

    def step(
        self, action: gym.core.ActType
    ) -> t.Tuple[int, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        return self._discretize_state(state), reward, done, info

    def render(self):
        self.env.render()

    @property
    def num_states(self) -> int:
        mask = ~np.isinf(self.codebooks)
        return np.prod(mask.sum(axis=0)).item()

    @property
    def num_actions(self) -> int:
        return self.env.action_space.n

    def _discretize_state(self, state: t.Union[int, np.ndarray]) -> int:
        if isinstance(state, int):
            state = np.array([state])
        if not np.all((self.min_state <= state) & (state <= self.max_state)):
            logger.debug(f"State {state.tolist()} is out of bounds")

        codes = np.abs(state - self.codebooks).argmin(axis=0)
        codebook_lengths = (~np.isinf(self.codebooks)).sum(axis=0)
        factors = np.cumprod(np.concatenate([[1], codebook_lengths[:-1]]))

        return (codes * factors).sum().item()
