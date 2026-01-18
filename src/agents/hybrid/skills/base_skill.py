"""
Abstract base class for RL skills used by the orchestrator.
"""
from abc import ABC, abstractmethod
from typing import Optional

from stable_baselines3 import PPO

# Try to import RecurrentPPO
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False


class BaseSkill(ABC):
    """
    Abstract base class for RL skills.

    Each skill wraps a trained RL model and provides:
    - Termination conditions
    - Applicability checks
    - Parameter configuration
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.name = self.__class__.__name__.lower().replace("skill", "")
        self.lstm_states = None

        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        """Load trained model. Try RecurrentPPO first, fall back to PPO."""
        if HAS_RECURRENT:
            try:
                self.model = RecurrentPPO.load(model_path)
                print(f"Loaded RecurrentPPO model for {self.name}")
                return
            except Exception:
                pass

        try:
            self.model = PPO.load(model_path)
            print(f"Loaded PPO model for {self.name}")
        except Exception as e:
            print(f"Warning: Could not load model for {self.name}: {e}")
            self.model = None

    def predict(self, observation, deterministic: bool = True):
        """Get action from model."""
        if self.model is None:
            raise RuntimeError(f"No model loaded for skill: {self.name}")

        action, self.lstm_states = self.model.predict(
            observation,
            state=self.lstm_states,
            deterministic=deterministic
        )
        return action

    def reset_states(self):
        """Reset LSTM states for new skill invocation."""
        self.lstm_states = None

    @abstractmethod
    def configure(self, **params):
        """Configure skill with parameters from orchestrator."""
        pass

    @abstractmethod
    def check_termination(self, game_info: dict) -> tuple[bool, bool, Optional[str]]:
        """
        Check if skill should terminate.

        Args:
            game_info: Info dict from env._get_info()

        Returns:
            (done, success, interrupt_reason)
            interrupt_reason: "battle", "timeout", etc. or None
        """
        pass

    @abstractmethod
    def is_applicable(self, game_info: dict) -> bool:
        """Check if this skill can be used in current game state."""
        pass
