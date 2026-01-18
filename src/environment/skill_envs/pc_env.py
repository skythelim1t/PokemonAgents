"""
Environment for training PC management skill.
Starts inside Pokemon Center, rewards successful deposit/withdraw.
"""
from .base_skill_env import BaseSkillEnv


class PCEnv(BaseSkillEnv):
    """
    Train agent to use Pokemon PC for deposit/withdraw.

    Should be initialized with save state inside Pokemon Center,
    with Pokemon in party and in PC boxes.
    """

    POKEMON_CENTER_MAPS = {41, 58, 64, 68, 81, 89, 133, 141, 154, 171}

    def __init__(
        self,
        rom_path: str,
        init_state: str | None = None,
        action: str = "deposit",  # "deposit" or "withdraw"
        skill_max_steps: int = 300,
        **kwargs
    ):
        super().__init__(
            rom_path=rom_path,
            init_state=init_state,
            skill_max_steps=skill_max_steps,
            **kwargs
        )
        self.pc_action = action
        self.initial_party_size = 0

    def _on_skill_reset(self):
        """Capture initial party state."""
        state = self._get_skill_state()
        self.initial_party_size = state["party_size"]

    def _get_skill_state(self) -> dict:
        """Extended state for PC tracking."""
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
        PC rewards:
        - Correct action completed: +5.0
        - Wrong action (deposited when should withdraw): -2.0
        - Left Pokemon Center: -3.0
        - Step penalty: -0.01 (encourage efficiency)
        """
        reward = -0.01  # Step penalty

        # Check for party size changes
        prev_size = prev_state["party_size"]
        curr_size = curr_state["party_size"]

        if self.pc_action == "deposit":
            if curr_size < prev_size:
                reward += 5.0  # Successfully deposited
            elif curr_size > prev_size:
                reward -= 2.0  # Wrong direction

        elif self.pc_action == "withdraw":
            if curr_size > prev_size:
                reward += 5.0  # Successfully withdrew
            elif curr_size < prev_size:
                reward -= 2.0  # Wrong direction

        # Left Pokemon Center without completing
        if not curr_state["in_pokemon_center"] and prev_state["in_pokemon_center"]:
            reward -= 3.0

        return reward

    def _check_skill_termination(self, state: dict) -> tuple[bool, bool]:
        """
        Terminate when:
        - Party composition changed correctly (success)
        - Left Pokemon Center (failure)
        """
        curr_size = state["party_size"]

        if self.pc_action == "deposit":
            if curr_size < self.initial_party_size:
                return True, True  # Success
        elif self.pc_action == "withdraw":
            if curr_size > self.initial_party_size:
                return True, True  # Success

        # Left building = failure
        if not state["in_pokemon_center"]:
            return True, False

        return False, False
