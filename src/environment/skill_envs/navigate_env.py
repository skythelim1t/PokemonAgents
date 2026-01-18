"""
Environment for training navigation skill.
Rewards movement toward target locations, penalizes battles.
"""
from .base_skill_env import BaseSkillEnv


class NavigateEnv(BaseSkillEnv):
    """
    Train agent to navigate between maps/locations.

    Args:
        target_maps: List of map IDs that count as success
        avoid_battles: If True, penalize entering battles
    """

    def __init__(
        self,
        rom_path: str,
        init_state: str | None = None,
        target_maps: list[int] | None = None,
        avoid_battles: bool = True,
        skill_max_steps: int = 2000,
        **kwargs
    ):
        super().__init__(
            rom_path=rom_path,
            init_state=init_state,
            skill_max_steps=skill_max_steps,
            **kwargs
        )
        self.target_maps = target_maps or []
        self.avoid_battles = avoid_battles
        self.nav_visited_coords: set[tuple[int, int, int]] = set()

    def _on_skill_reset(self):
        """Reset navigation-specific tracking."""
        self.nav_visited_coords = set()
        # Add starting position
        state = self._get_skill_state()
        self.nav_visited_coords.add((state["map_id"], state["player_x"], state["player_y"]))

    def _calculate_skill_reward(
        self,
        prev_state: dict,
        curr_state: dict,
        action: int
    ) -> float:
        """
        Navigation rewards:
        - New tile: +0.005
        - Reached target map: +5.0
        - Entered battle (if avoid_battles): -0.5
        """
        reward = 0.0

        # New tile bonus
        coord = (curr_state["map_id"], curr_state["player_x"], curr_state["player_y"])
        if coord not in self.nav_visited_coords:
            self.nav_visited_coords.add(coord)
            reward += 0.005

        # Target map reached
        if curr_state["map_id"] in self.target_maps:
            reward += 5.0

        # Battle penalty (navigation should avoid battles)
        if self.avoid_battles:
            if curr_state["in_battle"] and not prev_state["in_battle"]:
                reward -= 0.5

        return reward

    def _check_skill_termination(self, state: dict) -> tuple[bool, bool]:
        """
        Terminate when:
        - Reached target map (success)
        - Battle started (pause - orchestrator handles battle)
        """
        # Success: reached target
        if state["map_id"] in self.target_maps:
            return True, True

        # Pause: battle started (not failure, just interrupt)
        if state["in_battle"]:
            return True, False

        return False, False
