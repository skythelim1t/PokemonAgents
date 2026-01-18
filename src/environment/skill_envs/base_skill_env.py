"""
Base class for skill-specific environments.
Inherits from PokemonRedEnv and overrides reward/termination logic.
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environment.pokemon_env import PokemonRedEnv


class BaseSkillEnv(PokemonRedEnv, ABC):
    """
    Abstract base for skill training environments.

    Inherits all emulator handling, observation/action spaces from PokemonRedEnv.
    Subclasses override reward calculation and termination conditions.
    """

    def __init__(
        self,
        rom_path: str,
        init_state: str | None = None,
        skill_max_steps: int = 1000,
        **kwargs
    ):
        # Set a high max_steps in parent; we'll handle termination ourselves
        super().__init__(
            rom_path=rom_path,
            init_state=init_state,
            max_steps=999999,  # Effectively infinite; skill handles termination
            **kwargs
        )
        self.skill_max_steps = skill_max_steps
        self.skill_steps = 0
        self._skill_prev_state: dict = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset environment and skill-specific state."""
        obs, info = super().reset(seed=seed, options=options)
        self.skill_steps = 0
        self._skill_prev_state = self._get_skill_state()
        self._on_skill_reset()
        return obs, info

    def step(self, action: int) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        """Execute action with skill-specific reward and termination."""
        # Get previous state
        prev_state = self._skill_prev_state

        # Execute action using parent's step (handles emulator, button press)
        # We ignore parent's reward; calculate our own
        obs, _, terminated, truncated, info = super().step(action)

        # Get current state
        curr_state = self._get_skill_state()
        self._skill_prev_state = curr_state

        # Calculate skill-specific reward (override parent's reward)
        reward = self._calculate_skill_reward(prev_state, curr_state, action)

        # Check skill-specific termination
        self.skill_steps += 1
        skill_done, skill_success = self._check_skill_termination(curr_state)

        if skill_done:
            terminated = True
            info["skill_success"] = skill_success

        if self.skill_steps >= self.skill_max_steps:
            truncated = True
            info["skill_success"] = False
            info["skill_timeout"] = True

        info["skill_steps"] = self.skill_steps

        return obs, reward, terminated, truncated, info

    def _get_skill_state(self) -> dict:
        """
        Get current game state relevant to skills.
        Uses existing game_state from PokemonRedEnv.
        """
        if self.game_state is None:
            return {}

        x, y = self.game_state.get_player_position()
        party_hp, party_max_hp = self.game_state.get_total_party_hp()

        return {
            "map_id": self.game_state.get_map_id(),
            "player_x": x,
            "player_y": y,
            "in_battle": self.game_state.is_in_battle(),
            "party_hp": party_hp,
            "party_max_hp": party_max_hp,
            "party_hp_fraction": party_hp / party_max_hp if party_max_hp > 0 else 0,
            "party_size": self.game_state.get_party_count(),
            "party_alive": self.game_state.get_party_alive_count(),
            "badges": self.game_state.get_badge_count(),
            "in_menu": self.game_state.is_in_menu_or_dialogue(),
        }

    @abstractmethod
    def _calculate_skill_reward(
        self,
        prev_state: dict,
        curr_state: dict,
        action: int
    ) -> float:
        """Calculate skill-specific reward. Override in subclass."""
        pass

    @abstractmethod
    def _check_skill_termination(self, state: dict) -> tuple[bool, bool]:
        """
        Check if skill should terminate.
        Returns: (done, success)
        """
        pass

    def _on_skill_reset(self):
        """Hook for subclass-specific reset logic. Override if needed."""
        pass
