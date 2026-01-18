"""
PC management skill - handles Pokemon PC deposit/withdraw.
"""
from typing import Optional

from .base_skill import BaseSkill


class PCManageSkill(BaseSkill):
    """Skill for managing Pokemon PC (deposit/withdraw)."""

    POKEMON_CENTER_MAPS = {41, 58, 64, 68, 81, 89, 133, 141, 154, 171}

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path)
        self.action = "deposit"  # "deposit" or "withdraw"
        self.max_steps = 300
        self.current_steps = 0
        self.initial_party_size = 0

    def configure(
        self,
        action: str = "deposit",
        max_steps: int = 300,
        **kwargs
    ):
        """Configure PC management action."""
        self.action = action
        self.max_steps = max_steps
        self.current_steps = 0
        self.initial_party_size = 0
        self.reset_states()

    def check_termination(self, game_info: dict) -> tuple[bool, bool, Optional[str]]:
        """
        Terminate when:
        - Party composition changed correctly (success)
        - Left Pokemon Center (failure)
        - Max steps exceeded (failure)
        """
        self.current_steps += 1

        # Capture initial party size on first check
        if self.current_steps == 1:
            self.initial_party_size = game_info.get("party_count", 0)

        current_party_size = game_info.get("party_count", 0)

        # Success: correct action completed
        if self.action == "deposit":
            if current_party_size < self.initial_party_size:
                return True, True, None
        elif self.action == "withdraw":
            if current_party_size > self.initial_party_size:
                return True, True, None

        # Failure: left Pokemon Center
        current_map = game_info.get("map_id", 0)
        if current_map not in self.POKEMON_CENTER_MAPS:
            return True, False, "left_building"

        # Failure: timeout
        if self.current_steps >= self.max_steps:
            return True, False, "timeout"

        return False, False, None

    def is_applicable(self, game_info: dict) -> bool:
        """Can use PC if in Pokemon Center and not in battle."""
        if game_info.get("in_battle", False):
            return False

        current_map = game_info.get("map_id", 0)
        return current_map in self.POKEMON_CENTER_MAPS
