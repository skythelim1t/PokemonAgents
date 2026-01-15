"""PPO agent wrapper for use with spectator UI."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.agents.base import BaseAgent


class PPOAgent(BaseAgent):
    """Wrapper around a trained PPO model for spectator UI."""

    def __init__(self, model_path: Path, deterministic: bool = True) -> None:
        """
        Initialize the PPO agent.

        Args:
            model_path: Path to the saved model (.zip file)
            deterministic: Use deterministic actions (default True)
        """
        from stable_baselines3 import PPO

        self.model = PPO.load(str(model_path))
        self.deterministic = deterministic

    def act(self, observation: NDArray[np.uint8], info: dict) -> int:
        """Select action based on observation."""
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(action)

    def reset(self) -> None:
        """Reset agent state (no-op for PPO)."""
        pass
