"""
Skill Router - loads and executes RL skill models.
"""
from pathlib import Path
from typing import Optional

from .skills import (
    BaseSkill,
    NavigateSkill,
    BattleSkill,
    HealSkill,
    PCManageSkill,
)


class SkillRouter:
    """
    Manages RL skill models and routes execution to the appropriate skill.

    Loads trained models on demand and handles skill switching.
    """

    SKILL_CLASSES = {
        "navigate": NavigateSkill,
        "battle": BattleSkill,
        "heal": HealSkill,
        "pc_manage": PCManageSkill,
    }

    def __init__(self, models_dir: str | Path):
        """
        Initialize skill router.

        Args:
            models_dir: Directory containing trained skill models.
                        Expected structure: models_dir/{skill_name}/best_model.zip
        """
        self.models_dir = Path(models_dir)
        self.skills: dict[str, BaseSkill] = {}
        self.active_skill: Optional[BaseSkill] = None
        self.active_skill_name: Optional[str] = None

    def load_skill(self, skill_name: str) -> BaseSkill:
        """
        Load a skill model if not already loaded.

        Args:
            skill_name: Name of skill to load ("navigate", "battle", etc.)

        Returns:
            Loaded skill instance
        """
        if skill_name in self.skills:
            return self.skills[skill_name]

        if skill_name not in self.SKILL_CLASSES:
            raise ValueError(f"Unknown skill: {skill_name}")

        # Find model path
        model_path = self.models_dir / skill_name / "best_model.zip"
        if not model_path.exists():
            # Try alternative path
            model_path = self.models_dir / f"{skill_name}.zip"

        skill_class = self.SKILL_CLASSES[skill_name]

        if model_path.exists():
            skill = skill_class(model_path=str(model_path))
        else:
            print(f"Warning: No model found for {skill_name} at {model_path}")
            skill = skill_class(model_path=None)

        self.skills[skill_name] = skill
        return skill

    def activate_skill(self, skill_name: str, **config) -> BaseSkill:
        """
        Activate a skill with given configuration.

        Args:
            skill_name: Name of skill to activate
            **config: Configuration parameters for the skill

        Returns:
            Activated skill instance
        """
        skill = self.load_skill(skill_name)
        skill.configure(**config)
        self.active_skill = skill
        self.active_skill_name = skill_name
        return skill

    def get_action(self, observation, deterministic: bool = True) -> int:
        """
        Get action from active skill.

        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take

        Raises:
            RuntimeError: If no skill is active
        """
        if self.active_skill is None:
            raise RuntimeError("No active skill. Call activate_skill first.")

        return self.active_skill.predict(observation, deterministic=deterministic)

    def check_termination(self, game_info: dict) -> tuple[bool, bool, Optional[str]]:
        """
        Check if active skill should terminate.

        Args:
            game_info: Current game state info

        Returns:
            (done, success, interrupt_reason)
        """
        if self.active_skill is None:
            return True, False, "no_skill"

        return self.active_skill.check_termination(game_info)

    def deactivate_skill(self):
        """Deactivate the current skill."""
        if self.active_skill:
            self.active_skill.reset_states()
        self.active_skill = None
        self.active_skill_name = None

    def get_applicable_skills(self, game_info: dict) -> list[str]:
        """
        Get list of skills applicable to current game state.

        Args:
            game_info: Current game state info

        Returns:
            List of applicable skill names
        """
        applicable = []
        for skill_name in self.SKILL_CLASSES:
            skill = self.load_skill(skill_name)
            if skill.is_applicable(game_info):
                applicable.append(skill_name)
        return applicable
