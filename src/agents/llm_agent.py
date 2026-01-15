"""LLM-based strategic agent for Pokemon Red."""

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.agents.base import BaseAgent
from src.agents.llm.actions import StrategicAction, SIMPLE_ACTION_TO_BUTTON, action_from_string
from src.agents.llm.providers import LLMConfig, BaseLLMProvider, create_provider
from src.agents.llm.prompts import SYSTEM_PROMPT, format_game_state_prompt

logger = logging.getLogger(__name__)


@dataclass
class LLMAgentConfig:
    """Configuration for the LLM agent."""

    # LLM settings
    provider: Literal["anthropic", "openai", "bedrock"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 64
    region: str = "us-east-1"  # For Bedrock

    # Efficiency settings
    action_skip: int = 10  # Only call LLM every N frames
    use_vision: bool = False  # Include screen image in prompt

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # History
    history_length: int = 10  # Remember last N actions


class LLMAgent(BaseAgent):
    """
    LLM-based agent that makes strategic decisions.

    The agent calls an LLM (Claude, GPT, or Bedrock models) to decide on
    high-level actions, which are then translated to Game Boy button presses.
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

        # State
        self._step_count = 0
        self._last_action: int = 0  # Button index
        self._last_strategic_action: StrategicAction | None = None
        self._action_history: list[str] = []
        self._total_tokens_used = 0

        # Cache for repeated actions
        self._cached_action: int | None = None
        self._cache_valid_until: int = 0

        # Battle state tracking
        self._was_in_battle: bool = False

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

        # Check if we should skip LLM call and repeat cached action
        if self._step_count < self._cache_valid_until and self._cached_action is not None:
            return self._cached_action

        # Call LLM for decision
        try:
            strategic_action = self._get_llm_decision(observation, info)
            button = self._strategic_to_button(strategic_action)

            # Cache the action
            self._cached_action = button
            self._cache_valid_until = self._step_count + self.config.action_skip

            # Track history
            self._last_action = button
            self._last_strategic_action = strategic_action
            self._action_history.append(strategic_action.name)
            if len(self._action_history) > self.config.history_length:
                self._action_history.pop(0)

            logger.debug(f"Step {self._step_count}: {strategic_action.name} -> button {button}")
            return button

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback: repeat last action or press A
            return self._last_action if self._last_action else 0

    def _get_llm_decision(
        self,
        observation: NDArray[np.uint8],
        info: dict[str, Any],
    ) -> StrategicAction:
        """Get a decision from the LLM."""
        # Format prompt
        user_prompt = format_game_state_prompt(
            info,
            include_history=True,
            recent_actions=self._action_history,
        )

        # Prepare image if using vision
        image_bytes = None
        if self.config.use_vision:
            image_bytes = self._observation_to_png(observation)

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

                # Parse response
                action = self._parse_llm_response(response.content, info)
                logger.debug(f"LLM chose: {action.name} (tokens: {response.usage})")
                return action

            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))  # Exponential backoff

        # All retries failed, return default action
        logger.error("All LLM retries failed, using default action")
        return self._get_default_action(info)

    def _parse_llm_response(self, response: str, info: dict[str, Any]) -> StrategicAction:
        """Parse the LLM response into a strategic action."""
        # Try to match action name
        action = action_from_string(response)

        if action and self._is_valid_action(action, info):
            return action

        # Fallback: return default based on context
        logger.warning(f"Could not parse LLM response: {response[:100]}")
        return self._get_default_action(info)

    def _is_valid_action(self, action: StrategicAction, info: dict[str, Any]) -> bool:
        """Check if an action is valid for the current game state."""
        in_battle = info.get("in_battle", False)

        # Battle-only actions
        battle_actions = {
            StrategicAction.USE_MOVE_1,
            StrategicAction.USE_MOVE_2,
            StrategicAction.USE_MOVE_3,
            StrategicAction.USE_MOVE_4,
            StrategicAction.FLEE,
        }

        if action in battle_actions and not in_battle:
            return False

        return True

    def _get_default_action(self, info: dict[str, Any]) -> StrategicAction:
        """Get a sensible default action based on game state."""
        if info.get("in_battle", False):
            return StrategicAction.USE_MOVE_1
        else:
            return StrategicAction.PRESS_A

    def _strategic_to_button(self, action: StrategicAction) -> int:
        """Convert a strategic action to a button index."""
        return SIMPLE_ACTION_TO_BUTTON.get(action, 0)

    def _observation_to_png(self, observation: NDArray[np.uint8]) -> bytes:
        """Convert observation array to PNG bytes."""
        image = Image.fromarray(observation)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        self._step_count = 0
        self._last_action = 0
        self._last_strategic_action = None
        self._action_history = []
        self._cached_action = None
        self._cache_valid_until = 0
        self._was_in_battle = False
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
        """
        Update after taking an action.

        Invalidates cache if game state changed significantly.
        """
        # Invalidate cache if entering/exiting battle
        curr_in_battle = info.get("in_battle", False)

        if self._was_in_battle != curr_in_battle:
            self._cached_action = None
            self._cache_valid_until = 0
            logger.debug("Battle state changed, invalidating cache")

        self._was_in_battle = curr_in_battle

    @property
    def tokens_used(self) -> int:
        """Get total tokens used by this agent."""
        return self._total_tokens_used

    @property
    def last_action_name(self) -> str:
        """Get the name of the last strategic action."""
        if self._last_strategic_action:
            return self._last_strategic_action.name
        return "NONE"
