"""
Battle skill - handles Pokemon battles.
"""
from typing import Optional

from .base_skill import BaseSkill


class BattleSkill(BaseSkill):
    """Skill for handling Pokemon battles."""

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path)
        self.strategy = "aggressive"
        self.max_turns = 100
        self.current_turns = 0

    def configure(
        self,
        strategy: str = "aggressive",
        max_turns: int = 100,
        **kwargs
    ):
        """Configure battle strategy."""
        self.strategy = strategy
        self.max_turns = max_turns
        self.current_turns = 0
        self.reset_states()

    def check_termination(self, game_info: dict) -> tuple[bool, bool, Optional[str]]:
        """
        Terminate when:
        - Battle ended (check win/loss)
        - Max turns exceeded
        """
        self.current_turns += 1

        # Battle ended
        if not game_info.get("in_battle", False):
            # Won if party is still alive
            success = game_info.get("party_alive", 0) > 0
            return True, success, None

        # Timeout
        if self.current_turns >= self.max_turns:
            return True, False, "timeout"

        return False, False, None

    def is_applicable(self, game_info: dict) -> bool:
        """Can only use battle skill when in battle."""
        return game_info.get("in_battle", False)
