"""
LLM Orchestrator - high-level decision making for hybrid agent.
"""
import json
from pathlib import Path
from typing import Optional

from .skill_router import SkillRouter
from ..llm.providers import create_provider, LLMConfig


class Orchestrator:
    """
    LLM-based orchestrator that decides which RL skill to invoke.

    Observes game state periodically and delegates to specialized
    RL skills for execution.
    """

    SYSTEM_PROMPT = """You are an AI playing Pokemon Red. You control a hybrid agent that delegates
to specialized RL skills for different tasks.

Available skills:
- navigate: Move to a target map/location. Config: target_map_ids (list), target_x (int), target_y (int)
- battle: Fight Pokemon battles. Config: strategy ("aggressive", "defensive", "balanced")
- heal: Heal Pokemon at Pokemon Center. Config: none
- pc_manage: Deposit/withdraw Pokemon from PC. Config: action ("deposit", "withdraw")

You will receive game state information and must decide:
1. Which skill to invoke
2. Configuration for that skill

Respond in JSON format:
{
    "skill": "skill_name",
    "config": {"param": "value"},
    "reasoning": "brief explanation"
}

Common map IDs:
- Pallet Town: 0
- Viridian City: 1
- Pewter City: 2
- Cerulean City: 3
- Route 1: 12
- Route 2: 13
- Viridian Forest: 51
"""

    def __init__(
        self,
        models_dir: str | Path,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-3-haiku-20240307",
        decision_interval: int = 100,
    ):
        """
        Initialize orchestrator.

        Args:
            models_dir: Directory containing trained skill models
            llm_provider: LLM provider ("anthropic", "openai", etc.)
            llm_model: Model name to use
            decision_interval: Steps between LLM decisions when skill running
        """
        self.skill_router = SkillRouter(models_dir)
        self.decision_interval = decision_interval
        self.steps_since_decision = 0

        # Initialize LLM
        config = LLMConfig(provider=llm_provider, model=llm_model)
        self.llm = create_provider(config)

        # State tracking
        self.current_goal: Optional[str] = None
        self.last_decision: Optional[dict] = None

    def _format_game_state(self, game_info: dict) -> str:
        """Format game state for LLM prompt."""
        state_lines = [
            f"Map ID: {game_info.get('map_id', 'unknown')}",
            f"Position: {game_info.get('position', (0, 0))}",
            f"In Battle: {game_info.get('in_battle', False)}",
            f"Party Count: {game_info.get('party_count', 0)}",
        ]

        # Add HP info
        party_hp = game_info.get("party_hp", [])
        party_max_hp = game_info.get("party_max_hp", [])
        if party_hp and party_max_hp:
            hp_str = ", ".join(f"{h}/{m}" for h, m in zip(party_hp, party_max_hp))
            state_lines.append(f"Party HP: {hp_str}")

        # Add badges
        badges = game_info.get("badges", 0)
        state_lines.append(f"Badges: {badges}")

        # Add applicable skills
        applicable = self.skill_router.get_applicable_skills(game_info)
        state_lines.append(f"Applicable Skills: {', '.join(applicable)}")

        return "\n".join(state_lines)

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response JSON."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            return json.loads(response)
        except json.JSONDecodeError:
            # Default fallback
            return {
                "skill": "navigate",
                "config": {"target_map_ids": [1]},
                "reasoning": "JSON parse failed, defaulting to navigation"
            }

    def decide(self, game_info: dict) -> tuple[str, dict]:
        """
        Ask LLM to decide which skill to use.

        Args:
            game_info: Current game state

        Returns:
            (skill_name, config_dict)
        """
        # Format prompt
        user_prompt = f"""Current game state:
{self._format_game_state(game_info)}

What skill should I use and with what configuration?"""

        # Get LLM response
        response = self.llm.complete(
            system=self.SYSTEM_PROMPT,
            user=user_prompt
        )

        # Parse response
        decision = self._parse_llm_response(response)
        self.last_decision = decision

        skill_name = decision.get("skill", "navigate")
        config = decision.get("config", {})

        return skill_name, config

    def step(self, observation, game_info: dict) -> tuple[int, bool]:
        """
        Execute one step of the hybrid agent.

        Args:
            observation: Current environment observation
            game_info: Current game state info dict

        Returns:
            (action, skill_changed) - action to take and whether skill changed
        """
        skill_changed = False
        self.steps_since_decision += 1

        # Check if current skill should terminate
        if self.skill_router.active_skill:
            done, success, interrupt = self.skill_router.check_termination(game_info)
            if done:
                self.skill_router.deactivate_skill()
                skill_changed = True

        # If no active skill, ask LLM for decision
        if self.skill_router.active_skill is None:
            skill_name, config = self.decide(game_info)
            self.skill_router.activate_skill(skill_name, **config)
            self.steps_since_decision = 0
            skill_changed = True

        # Get action from active skill
        action = self.skill_router.get_action(observation)

        return action, skill_changed

    def reset(self):
        """Reset orchestrator state for new episode."""
        self.skill_router.deactivate_skill()
        self.steps_since_decision = 0
        self.current_goal = None
        self.last_decision = None
