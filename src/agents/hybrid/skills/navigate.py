"""
Navigation skill - moves agent to target map/location.
"""
from typing import Optional

from .base_skill import BaseSkill


class NavigateSkill(BaseSkill):
    """Skill for navigating between maps/locations."""

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path)
        self.target_map_ids: list[int] = []
        self.target_x: int | None = None
        self.target_y: int | None = None
        self.max_steps = 2000
        self.current_steps = 0

    def configure(
        self,
        target_map_ids: list[int] | None = None,
        target_x: int | None = None,
        target_y: int | None = None,
        max_steps: int = 2000,
        **kwargs
    ):
        """Configure navigation target."""
        self.target_map_ids = target_map_ids or []
        self.target_x = target_x
        self.target_y = target_y
        self.max_steps = max_steps
        self.current_steps = 0
        self.reset_states()

    def check_termination(self, game_info: dict) -> tuple[bool, bool, Optional[str]]:
        """
        Terminate when:
        - Reached target map (success)
        - Battle started (interrupt)
        - Max steps exceeded (failure)
        """
        self.current_steps += 1

        # Interrupt: battle started
        if game_info.get("in_battle", False):
            return True, False, "battle"

        # Success: reached target map
        current_map = game_info.get("map_id")
        if current_map in self.target_map_ids:
            # If specific coordinates requested, check those too
            if self.target_x is not None and self.target_y is not None:
                pos = game_info.get("position", (0, 0))
                if pos[0] == self.target_x and pos[1] == self.target_y:
                    return True, True, None
            else:
                return True, True, None

        # Failure: timeout
        if self.current_steps >= self.max_steps:
            return True, False, "timeout"

        return False, False, None

    def is_applicable(self, game_info: dict) -> bool:
        """Can navigate if not in battle."""
        return not game_info.get("in_battle", False)
