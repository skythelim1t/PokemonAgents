"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def act(self, observation: NDArray[np.uint8], info: dict[str, Any]) -> int:
        """
        Choose an action based on the current observation.

        Args:
            observation: Current screen image (H, W, 3)
            info: Additional game state information

        Returns:
            Action index (0-7 for Game Boy buttons)
        """
        pass

    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        pass

    def update(
        self,
        observation: NDArray[np.uint8],
        action: int,
        reward: float,
        next_observation: NDArray[np.uint8],
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """
        Update the agent after taking an action (for learning agents).

        Args:
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended
            info: Additional information
        """
        pass
