"""
Heal skill - handles healing at Pokemon Centers.
"""
from typing import Optional

from .base_skill import BaseSkill


class HealSkill(BaseSkill):
    """Skill for healing Pokemon at Pokemon Centers."""

    POKEMON_CENTER_MAPS = {41, 58, 64, 68, 81, 89, 133, 141, 154, 171}

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path)
        self.max_steps = 500
        self.current_steps = 0
        self.initial_hp_fraction = 1.0

    def configure(
        self,
        max_steps: int = 500,
        **kwargs
    ):
        """Configure heal skill."""
        self.max_steps = max_steps
        self.current_steps = 0
        self.initial_hp_fraction = 1.0
        self.reset_states()

    def check_termination(self, game_info: dict) -> tuple[bool, bool, Optional[str]]:
        """
        Terminate when:
        - Party fully healed (success)
        - Battle started (interrupt)
        - Max steps exceeded (failure)
        """
        self.current_steps += 1

        # Capture initial HP on first check
        if self.current_steps == 1:
            party_hp = game_info.get("party_hp", [])
            party_max_hp = game_info.get("party_max_hp", [])
            if party_max_hp and sum(party_max_hp) > 0:
                self.initial_hp_fraction = sum(party_hp) / sum(party_max_hp)

        # Interrupt: battle started
        if game_info.get("in_battle", False):
            return True, False, "battle"

        # Success: party fully healed
        party_hp = game_info.get("party_hp", [])
        party_max_hp = game_info.get("party_max_hp", [])
        if party_hp and party_max_hp:
            current_hp = sum(party_hp)
            max_hp = sum(party_max_hp)
            if max_hp > 0 and current_hp >= max_hp:
                return True, True, None

        # Failure: timeout
        if self.current_steps >= self.max_steps:
            return True, False, "timeout"

        return False, False, None

    def is_applicable(self, game_info: dict) -> bool:
        """Can heal if not in battle and party needs healing."""
        if game_info.get("in_battle", False):
            return False

        # Check if healing is needed
        party_hp = game_info.get("party_hp", [])
        party_max_hp = game_info.get("party_max_hp", [])
        if party_hp and party_max_hp:
            current_hp = sum(party_hp)
            max_hp = sum(party_max_hp)
            if max_hp > 0:
                return current_hp < max_hp

        return False
