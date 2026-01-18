"""
Environment for training battle skill.
Starts from battle state, rewards winning efficiently.
"""
from .base_skill_env import BaseSkillEnv


class BattleEnv(BaseSkillEnv):
    """
    Train agent to win Pokemon battles.

    Should be initialized with save states that start IN a battle.
    """

    def __init__(
        self,
        rom_path: str,
        init_state: str | None = None,
        skill_max_steps: int = 500,
        **kwargs
    ):
        super().__init__(
            rom_path=rom_path,
            init_state=init_state,
            skill_max_steps=skill_max_steps,
            **kwargs
        )
        self.initial_party_hp_fraction = 0.0
        self.prev_enemy_hp_fraction = 1.0

    def _on_skill_reset(self):
        """Capture initial HP values."""
        state = self._get_skill_state()
        self.initial_party_hp_fraction = state["party_hp_fraction"]
        self.prev_enemy_hp_fraction = 1.0

    def _get_skill_state(self) -> dict:
        """Extended state for battle tracking."""
        state = super()._get_skill_state()

        # Add enemy HP info
        if self.game_state is not None and state["in_battle"]:
            enemy_hp, enemy_max_hp = self.game_state.get_enemy_hp()
            state["enemy_hp"] = enemy_hp
            state["enemy_max_hp"] = enemy_max_hp
            state["enemy_hp_fraction"] = enemy_hp / enemy_max_hp if enemy_max_hp > 0 else 0
        else:
            state["enemy_hp"] = 0
            state["enemy_max_hp"] = 0
            state["enemy_hp_fraction"] = 0

        return state

    def _calculate_skill_reward(
        self,
        prev_state: dict,
        curr_state: dict,
        action: int
    ) -> float:
        """
        Battle rewards:
        - Damage dealt: +0.5 x (damage / enemy_max_hp)
        - Enemy KO: +2.0
        - OHKO: +0.5
        - No damage taken: +0.5 (on win)
        - Battle won: +3.0
        - Battle lost: -5.0
        """
        reward = 0.0

        # Only calculate battle rewards while in battle
        if prev_state["in_battle"]:
            prev_enemy_hp = self.prev_enemy_hp_fraction
            curr_enemy_hp = curr_state["enemy_hp_fraction"]

            # Damage dealt
            damage_dealt = prev_enemy_hp - curr_enemy_hp
            if damage_dealt > 0:
                reward += 0.5 * damage_dealt
                # KO bonus
                if curr_enemy_hp <= 0:
                    reward += 2.0
                    # OHKO bonus
                    if prev_enemy_hp >= 0.95:
                        reward += 0.5

            self.prev_enemy_hp_fraction = curr_enemy_hp

        # Battle outcome (battle just ended)
        if prev_state["in_battle"] and not curr_state["in_battle"]:
            # Check if we won (party still alive) or lost
            if curr_state["party_alive"] > 0:
                reward += 3.0  # Won
                # No damage bonus
                if curr_state["party_hp_fraction"] >= self.initial_party_hp_fraction - 0.01:
                    reward += 0.5
            else:
                reward -= 5.0  # Lost (whiteout)

        return reward

    def _check_skill_termination(self, state: dict) -> tuple[bool, bool]:
        """Terminate when battle ends."""
        # Check if battle ended
        if not state["in_battle"]:
            # Success if party is still alive
            success = state["party_alive"] > 0
            return True, success

        return False, False
