"""Random agent that selects actions uniformly at random."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects random actions with weighted probabilities."""

    # Button order: ["a", "b", "start", "select", "up", "down", "left", "right"]
    # Reduce start/select probability to avoid menu spam
    DEFAULT_WEIGHTS = [1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0]

    def __init__(
        self,
        num_actions: int = 8,
        seed: int | None = None,
        weights: list[float] | None = None,
    ) -> None:
        """
        Initialize the random agent.

        Args:
            num_actions: Number of possible actions (8 for Game Boy)
            seed: Random seed for reproducibility
            weights: Optional probability weights for each action
        """
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

        # Normalize weights to probabilities
        w = np.array(weights if weights else self.DEFAULT_WEIGHTS[:num_actions])
        self.probabilities = w / w.sum()

    def act(self, observation: NDArray[np.uint8], info: dict[str, Any]) -> int:
        """Select a random action based on weighted probabilities."""
        return self.rng.choice(self.num_actions, p=self.probabilities)

    def reset(self) -> None:
        """Reset is a no-op for random agent."""
        pass
