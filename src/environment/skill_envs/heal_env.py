"""
Environment for training heal skill.
Rewards finding Pokemon Center and healing party via Nurse Joy.
"""
from .base_skill_env import BaseSkillEnv


class HealEnv(BaseSkillEnv):
    """
    Train agent to navigate to Pokemon Center and heal at Nurse Joy.

    Should start from states with damaged party.
    """

    # Map IDs of Pokemon Centers (from Pokemon Red)
    POKEMON_CENTER_MAPS = {
        41,   # Viridian City Pokemon Center
        58,   # Pewter City Pokemon Center
        64,   # Cerulean City Pokemon Center
        68,   # Lavender Town Pokemon Center
        81,   # Vermilion City Pokemon Center
        89,   # Celadon City Pokemon Center
        133,  # Fuchsia City Pokemon Center
        141,  # Cinnabar Island Pokemon Center
        154,  # Saffron City Pokemon Center
        171,  # Indigo Plateau Pokemon Center
    }

    def __init__(
        self,
        rom_path: str,
        init_state: str | None = None,
        skill_max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            rom_path=rom_path,
            init_state=init_state,
            skill_max_steps=skill_max_steps,
            **kwargs
        )
        self.initial_hp_fraction = 0.0
        self.heal_visited_coords: set[tuple[int, int, int]] = set()

    def _on_skill_reset(self):
        """Track initial HP for comparison."""
        state = self._get_skill_state()
        self.initial_hp_fraction = state["party_hp_fraction"]
        self.heal_visited_coords = set()

    def _get_skill_state(self) -> dict:
        """Extended state for heal tracking."""
        state = super()._get_skill_state()
        state["in_pokemon_center"] = state["map_id"] in self.POKEMON_CENTER_MAPS
        return state

    def _calculate_skill_reward(
        self,
        prev_state: dict,
        curr_state: dict,
        action: int
    ) -> float:
        """
        Heal rewards:
        - HP restored: +1.0 x hp_gained
        - Full heal: +3.0
        - Entered Pokemon Center: +0.5
        - New tile: +0.002 (small exploration bonus)
        - Step penalty: -0.001 (encourage efficiency)
        """
        reward = -0.001  # Small step penalty

        # HP restored
        hp_gained = curr_state["party_hp_fraction"] - prev_state["party_hp_fraction"]
        if hp_gained > 0:
            reward += 1.0 * hp_gained
            # Full heal bonus
            if curr_state["party_hp_fraction"] >= 0.99:
                reward += 3.0

        # Entered Pokemon Center
        if curr_state["in_pokemon_center"] and not prev_state["in_pokemon_center"]:
            reward += 0.5

        # Small exploration bonus
        coord = (curr_state["map_id"], curr_state["player_x"], curr_state["player_y"])
        if coord not in self.heal_visited_coords:
            self.heal_visited_coords.add(coord)
            reward += 0.002

        return reward

    def _check_skill_termination(self, state: dict) -> tuple[bool, bool]:
        """
        Terminate when:
        - Fully healed (success)
        - Battle started (pause)
        """
        # Success: fully healed
        if state["party_hp_fraction"] >= 0.99:
            return True, True

        # Pause: battle started
        if state["in_battle"]:
            return True, False

        return False, False
