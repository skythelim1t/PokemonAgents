"""LLM-based strategic agent for Pokemon Red."""

import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.agents.base import BaseAgent
from src.agents.llm.actions import StrategicAction, MoveToAction, action_from_string
from src.agents.llm.knowledge import KnowledgeBase
from src.agents.llm.providers import LLMConfig, create_provider
from src.agents.llm.prompts import SYSTEM_PROMPT, format_game_state_prompt
from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt
from src.executor.action_executor import ActionExecutor, ExecutorState

logger = logging.getLogger(__name__)


@dataclass
class LLMAgentConfig:
    """Configuration for the LLM agent."""

    # LLM settings
    provider: Literal["anthropic", "openai", "bedrock"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 32
    region: str = "us-east-1"  # For Bedrock

    # Execution settings
    max_steps_per_action: int = 30  # Max steps executor runs per strategic action

    # Vision settings
    use_vision: bool = False  # Include screen image in prompt

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # History
    history_length: int = 5  # Remember last N actions

    # Logging
    log_conversation: bool = False  # Print prompts and responses


class LLMAgent(BaseAgent):
    """
    LLM-based agent that makes strategic decisions.

    The agent calls an LLM to decide on high-level actions (EXPLORE_UP, ATTACK_1, etc.)
    and uses an Executor to translate these into button presses.
    """

    def __init__(self, config: LLMAgentConfig | None = None) -> None:
        """
        Initialize the LLM agent.

        Args:
            config: Agent configuration. If None, uses defaults.
        """
        self.config = config or LLMAgentConfig()

        # Initialize LLM provider
        llm_config = LLMConfig(
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            region=self.config.region,
        )
        self.provider = create_provider(llm_config)

        # Initialize executor
        self.executor = ActionExecutor(max_steps_per_action=self.config.max_steps_per_action)

        # State
        self._step_count = 0
        self._last_strategic_action: StrategicAction | MoveToAction | None = None
        self._action_history: list[str] = []
        self._total_tokens_used = 0

        # Battle state tracking
        self._was_in_battle: bool = False

        # Menu stuck detection
        self._menu_interact_count: int = 0
        self._menu_stuck_threshold: int = 5  # After 5 INTERACTs on menu, try CANCEL

        # Position tracking - remember recent positions to avoid loops
        self._recent_positions: list[tuple[int, int, int]] = []  # (map_id, x, y)
        self._max_position_history = 20

        # Blocked direction tracking - detect when movement fails
        self._last_position: tuple[int, int] | None = None
        self._last_movement_action: str | None = None
        self._blocked_direction: str | None = None  # Direction that just failed

        # Knowledge base for persistent notes and location tracking
        self.knowledge = KnowledgeBase()

        logger.info(
            f"LLMAgent initialized with {self.config.provider}/{self.config.model}"
        )

    def act(self, observation: NDArray[np.uint8], info: dict[str, Any]) -> int:
        """
        Choose an action based on the current game state.

        Args:
            observation: Current screen image (144, 160, 3)
            info: Game state information

        Returns:
            Button index (0-7)
        """
        self._step_count += 1

        # Track current position
        map_id = info.get("map_id", 0)
        position = info.get("position", (0, 0))
        current_pos = (map_id, position[0], position[1])

        # Check if last movement action was blocked (position didn't change)
        if self._last_position is not None and self._last_movement_action is not None:
            if position == self._last_position:
                # Position didn't change - direction is blocked
                self._blocked_direction = self._last_movement_action
                logger.debug(f"Direction blocked: {self._blocked_direction}")
            else:
                # Movement succeeded - clear blocked direction
                self._blocked_direction = None

        # Add to position history if different from last
        if not self._recent_positions or self._recent_positions[-1] != current_pos:
            self._recent_positions.append(current_pos)
            if len(self._recent_positions) > self._max_position_history:
                self._recent_positions.pop(0)

        # Record visit in knowledge base
        self.knowledge.record_visit(map_id, position[0], position[1])

        # Add position info to info dict for prompt
        info["recent_positions"] = self._recent_positions
        info["position_stuck"] = self._is_stuck_in_area()
        info["blocked_direction"] = self._blocked_direction
        info["knowledge"] = self.knowledge
        info["map_name"] = self.knowledge.get_map_name(map_id)

        # If executor is busy, let it continue
        if self.executor.is_busy:
            result = self.executor.step(info)

            # If action was interrupted (e.g., battle started), get new decision
            if result.state == ExecutorState.INTERRUPTED:
                logger.debug(f"Action interrupted: {result.message}")
                # Fall through to get new LLM decision
            else:
                return result.button

        # Executor is idle or completed - get new strategic decision from LLM
        try:
            # Skip LLM call in certain situations - just press A to advance
            menu_active = info.get("menu_active", False)
            has_choice = info.get("has_menu_choice", False)
            in_battle = info.get("in_battle", False)
            battle_menu_ready = info.get("battle_menu_ready", False)

            # Auto-INTERACT situations (no LLM needed):
            # 1. Dialogue showing with no choices
            # 2. Battle starting (intro animations/text)
            if menu_active and not has_choice and not in_battle:
                # Dialogue showing, no choices - just advance
                strategic_action = StrategicAction.INTERACT
                logger.debug("Dialogue active - auto INTERACT")
                self._menu_interact_count = 0  # Reset counter when in pure dialogue
            elif menu_active and has_choice and not in_battle:
                # Menu with choices - check if we're stuck
                # Count recent INTERACT actions on menus
                recent_menu_actions = self._action_history[-self._menu_stuck_threshold:]
                interact_count = sum(1 for a in recent_menu_actions if a == "INTERACT")

                if interact_count >= self._menu_stuck_threshold:
                    # We've been pressing INTERACT too many times without progress
                    # Try CANCEL to escape the menu
                    strategic_action = StrategicAction.CANCEL
                    logger.info("Menu stuck detected - forcing CANCEL to escape")
                    self._menu_interact_count = 0  # Reset after forcing CANCEL
                else:
                    # Let LLM decide
                    strategic_action = self._get_llm_decision(observation, info)
            elif in_battle and not battle_menu_ready:
                # Battle intro/animation - just advance
                strategic_action = StrategicAction.INTERACT
                logger.debug("Battle intro - auto INTERACT")
            else:
                # Not in menu - reset menu counter
                self._menu_interact_count = 0
                strategic_action = self._get_llm_decision(observation, info)

            # Track history
            self._last_strategic_action = strategic_action
            # Get action name for history (handle both StrategicAction and MoveToAction)
            if isinstance(strategic_action, MoveToAction):
                action_name = str(strategic_action)  # "MOVE_TO (x, y)"
            else:
                action_name = strategic_action.name
            self._action_history.append(action_name)
            if len(self._action_history) > self.config.history_length:
                self._action_history.pop(0)

            # Track position for blocked direction detection
            self._last_position = position
            if action_name.startswith("EXPLORE_"):
                self._last_movement_action = action_name
            else:
                self._last_movement_action = None

            # Start executing the action
            self.executor.start_action(strategic_action, info)
            result = self.executor.step(info)

            logger.debug(f"LLM chose: {action_name} -> button {result.button}")
            return result.button

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback: press A
            return 0

    def _get_llm_decision(
        self,
        observation: NDArray[np.uint8],
        info: dict[str, Any],
    ) -> StrategicAction | MoveToAction:
        """Get a decision from the LLM."""
        # Prepare image if using vision
        image_bytes = None
        if self.config.use_vision:
            # Apply walkability overlay to help Claude see which tiles are walkable
            emulator = info.get("emulator")
            if emulator is not None:
                try:
                    overlay_obs = create_walkability_overlay(observation, emulator)
                    image_bytes = self._observation_to_png(overlay_obs)
                except Exception as e:
                    logger.warning(f"Failed to create walkability overlay: {e}")
                    image_bytes = self._observation_to_png(observation)
            else:
                image_bytes = self._observation_to_png(observation)

        # Format prompt
        user_prompt = format_game_state_prompt(
            info,
            include_history=True,
            recent_actions=self._action_history,
            use_vision=self.config.use_vision,
        )

        # Log conversation if enabled
        if self.config.log_conversation:
            print("\n" + "=" * 50)
            print(f"LLM CALL (tokens used: {self._total_tokens_used})")
            if self.config.use_vision:
                print("[Screenshot attached]")
            print("=" * 50)
            # Show system prompt only on first call
            if self._total_tokens_used == 0:
                print("SYSTEM PROMPT:")
                print(SYSTEM_PROMPT)
                print("-" * 50)
            print(user_prompt)
            print("-" * 50)

        # Call LLM with retries
        for attempt in range(self.config.max_retries):
            try:
                response = self.provider.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    image=image_bytes,
                )

                # Track usage
                self._total_tokens_used += response.usage.get("input", 0) + response.usage.get(
                    "output", 0
                )

                # Log response if enabled
                if self.config.log_conversation:
                    print(f"Response: {response.content}")
                    print("=" * 50 + "\n")

                # Parse response
                action = self._parse_llm_response(response.content, info)
                return action

            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if self.config.log_conversation:
                    print(f"ERROR: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        # All retries failed, return default action
        logger.error("All LLM retries failed, using default action")
        return self._get_default_action(info)

    def _parse_llm_response(self, response: str, info: dict[str, Any]) -> StrategicAction | MoveToAction:
        """Parse the LLM response into a strategic action or MoveToAction."""
        action = action_from_string(response)

        if action:
            return action

        # Fallback: return default based on context
        logger.warning(f"Could not parse LLM response: {response[:50]}")
        return self._get_default_action(info)

    def _get_default_action(self, info: dict[str, Any]) -> StrategicAction:
        """Get a sensible default action based on game state."""
        if info.get("in_battle", False):
            return StrategicAction.ATTACK_1
        else:
            return StrategicAction.INTERACT

    def _observation_to_png(self, observation: NDArray[np.uint8]) -> bytes:
        """Convert observation array to PNG bytes."""
        image = Image.fromarray(observation)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        self._step_count = 0
        self._last_strategic_action = None
        self._action_history = []
        self._was_in_battle = False
        self._recent_positions = []
        self._menu_interact_count = 0
        self._last_position = None
        self._last_movement_action = None
        self._blocked_direction = None
        self.executor.reset()
        # Keep knowledge base - don't reset it between episodes
        # This allows Claude to remember things across runs
        logger.info(f"LLMAgent reset. Total tokens used: {self._total_tokens_used}")

    def update(
        self,
        observation: NDArray[np.uint8],
        action: int,
        reward: float,
        next_observation: NDArray[np.uint8],
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update after taking an action."""
        # Track battle state changes
        curr_in_battle = info.get("in_battle", False)
        self._was_in_battle = curr_in_battle

    @property
    def tokens_used(self) -> int:
        """Get total tokens used by this agent."""
        return self._total_tokens_used

    @property
    def last_action_name(self) -> str:
        """Get the name of the last strategic action."""
        if self._last_strategic_action:
            if isinstance(self._last_strategic_action, MoveToAction):
                return str(self._last_strategic_action)
            return self._last_strategic_action.name
        return "NONE"

    def _is_stuck_in_area(self) -> bool:
        """Check if we're stuck in a small area (visiting same positions repeatedly)."""
        if len(self._recent_positions) < 10:
            return False

        # Get last 10 positions
        recent = self._recent_positions[-10:]

        # Count unique positions
        unique_positions = set(recent)

        # If we've only visited 3 or fewer unique spots in last 10 moves, we're stuck
        return len(unique_positions) <= 3

    def _get_movement_suggestion(self) -> str | None:
        """Suggest a direction to try based on position history."""
        if len(self._recent_positions) < 5:
            return None

        # Get recent positions on current map
        current_map = self._recent_positions[-1][0]
        recent_on_map = [(x, y) for m, x, y in self._recent_positions[-10:] if m == current_map]

        if len(recent_on_map) < 3:
            return None

        # Find which directions we've been going
        xs = [p[0] for p in recent_on_map]
        ys = [p[1] for p in recent_on_map]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Suggest unexplored direction
        suggestions = []
        if max_x - min_x < 3:  # Not much horizontal movement
            suggestions.extend(["EXPLORE_LEFT", "EXPLORE_RIGHT"])
        if max_y - min_y < 3:  # Not much vertical movement
            suggestions.extend(["EXPLORE_UP", "EXPLORE_DOWN"])

        if suggestions:
            return suggestions[0]
        return None
