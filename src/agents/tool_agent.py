"""Tool-based LLM agent for Pokemon Red.

This agent uses Claude's tool calling API instead of text parsing.
Claude calls tools (navigator, attack, update_knowledge) and receives
structured results back.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.agents.base import BaseAgent
from src.agents.llm.knowledge import KnowledgeBase
from src.agents.llm.providers import LLMConfig, create_provider, ToolCall
from src.agents.llm.tools import TOOL_DEFINITIONS, ToolExecutor, ToolResult
from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt
from src.agents.llm.decision_context import get_decision_context

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an AI playing Pokemon Red. ACT FAST - don't overthink!

=== CORE TOOLS ===
- navigator(x, y) - Move to screen position (0-9 x, 0-8 y). Your position is shown as @ in the grid.
- attack(move_slot) - Use attack 1-4 in battle
- press_button(button) - Press a/b/start/up/down/left/right

=== IMPORTANT: DON'T OVERTHINK ===
- Do NOT use think() except when truly stuck after many failed attempts
- Do NOT call query_knowledge() repeatedly - just act!
- Just pick a walkable tile and go. If blocked, try another.

=== WALKABILITY GRID ===
W=walkable, X=blocked, @=you (position shown in grid, usually center but can vary near edges)
- Pick tiles that have a CLEAR PATH of W tiles from @ to target
- If there's a wall of X between you and target, you can't reach it
- When stuck, try tiles in DIFFERENT DIRECTIONS (up/down/left/right from @)

=== NAVIGATION TIPS ===
- If navigator says "Stuck" or "BLOCKED", try a tile in the OPPOSITE direction
- Look for connected regions of W tiles - you can only reach tiles in YOUR region
- To leave an area, find edge tiles (row 0, row 8, column 0, column 9) that are W

=== INSIDE BUILDINGS ===
- Use find_exit() to locate the door/exit and get screen coordinates
- The exit coordinates will tell you exactly where to navigator() to
- Buildings have NO wild Pokemon - exit to explore routes

=== IN TOWNS (NO WILD POKEMON) ===
- TOWNS have NO wild Pokemon - you MUST leave to find routes
- Use find_route() to get directions to the nearest route
- Route 1 is NORTH of Pallet Town (row 0 edge)
- Walk to the EDGE of the map (row 0, row 8, col 0, col 9) to transition to routes

=== GAME KNOWLEDGE ===
- ROUTES have tall grass where wild Pokemon appear randomly when you walk
- Walk around repeatedly on routes to trigger battles

=== IN BATTLE ===
- Use attack(1-4) to fight
- run_away() to flee wild battles
"""


@dataclass
class ToolAgentConfig:
    """Configuration for the tool-based agent."""

    provider: Literal["anthropic", "openai", "bedrock"] = "bedrock"
    model: str = "anthropic.claude-3-haiku-20240307-v1:0"
    temperature: float = 0.7
    max_tokens: int = 1024  # More tokens for tool calls
    region: str = "us-east-1"

    max_tool_calls_per_turn: int = 1  # One tool call per turn to minimize token usage
    log_conversation: bool = False


class ToolAgent(BaseAgent):
    """
    Tool-based LLM agent that uses function calling.

    Instead of parsing text responses, this agent:
    1. Sends game state + screenshot to Claude with tool definitions
    2. Claude calls tools like navigator(5, 2) or attack(1)
    3. We execute the tool and return results
    4. Loop until Claude stops calling tools or max reached
    """

    def __init__(
        self,
        config: ToolAgentConfig | None = None,
        env=None,
        knowledge: KnowledgeBase | None = None,
    ) -> None:
        """
        Initialize the tool agent.

        Args:
            config: Agent configuration
            env: Pokemon environment (needed for tool execution)
            knowledge: Optional pre-loaded knowledge base (for resuming runs)
        """
        self.config = config or ToolAgentConfig()
        self.env = env

        # Initialize LLM provider
        llm_config = LLMConfig(
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            region=self.config.region,
        )
        self.provider = create_provider(llm_config)

        # Knowledge base - use provided or create fresh
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()

        # Tool executor (initialized when env is set)
        self.tool_executor: ToolExecutor | None = None

        # Conversation history for multi-turn
        self._messages: list[dict[str, Any]] = []
        self._total_tokens = 0
        self._turn_count = 0

        # Last action for display
        self._last_tool_call: str = "None"

        # Position history to detect oscillation
        self._position_history: list[tuple[int, int]] = []
        self._max_position_history = 10

        # Track last map and position for transition detection
        self._last_map_id: int | None = None
        self._last_position: tuple[int, int] | None = None

        # Tools to exclude from next turn (prevents loops like query_knowledge spam)
        self._excluded_tools: set[str] = set()

        logger.info(f"ToolAgent initialized with {self.config.provider}/{self.config.model}")

    def set_env(self, env) -> None:
        """Set the environment and initialize tool executor."""
        self.env = env
        self.tool_executor = ToolExecutor(env, self.knowledge)

    def set_render_callback(self, callback: callable) -> None:
        """Set a callback to render the screen during battle waits."""
        if self.tool_executor:
            self.tool_executor.set_render_callback(callback)

    def act(self, observation: NDArray[np.uint8], info: dict[str, Any]) -> int:
        """
        Choose an action by calling Claude with tools.

        This method:
        1. Prepares the game state as a message
        2. Calls Claude with tools
        3. Executes any tool calls
        4. Returns button press (or 0 if no direct action)

        Args:
            observation: Current screen image
            info: Game state information

        Returns:
            Button index to press (0-7)
        """
        if self.tool_executor is None:
            if self.env is not None:
                self.tool_executor = ToolExecutor(self.env, self.knowledge)
            else:
                logger.warning("No environment set, returning default action")
                return 0

        self._turn_count += 1

        # Track position history for oscillation detection
        current_pos = info.get("position", (0, 0))
        self._position_history.append(current_pos)
        if len(self._position_history) > self._max_position_history:
            self._position_history.pop(0)

        # Auto-update location in knowledge base
        map_id = info.get("map_id", 0)
        self.knowledge.update_location(map_id)

        # Auto-sync story flags from game state (starter, badges, etc.)
        self.knowledge.sync_from_game_state(info)

        # Detect and record map transitions
        if self._last_map_id is not None and map_id != self._last_map_id and self._last_position is not None:
            # Map changed! Record the transition
            direction = self._detect_transition_direction(self._last_position, current_pos)
            self.knowledge.record_map_transition(
                from_map_id=self._last_map_id,
                to_map_id=map_id,
                exit_coords=self._last_position,
                direction=direction,
            )

        self._last_map_id = map_id
        self._last_position = current_pos

        # Checkpoint on turn 1 and every 30 turns
        is_checkpoint = self._turn_count == 1 or self._turn_count % 30 == 0

        # Build the initial message with game state
        game_state_text = self._format_game_state(info)

        # Prepend checkpoint prompt if needed
        if is_checkpoint:
            checkpoint_prompt = """=== CHECKPOINT ===
Use think() to briefly assess: Where am I? What's my goal? What should I do next?
Then take action.
"""
            game_state_text = checkpoint_prompt + "\n" + game_state_text

        # Create screenshot with overlay
        emulator = self.env.emulator if self.env else None
        if emulator:
            try:
                overlay = create_walkability_overlay(observation, emulator)
                image_bytes = self._image_to_bytes(overlay)
            except Exception as e:
                logger.warning(f"Overlay failed: {e}")
                image_bytes = self._image_to_bytes(observation)
        else:
            image_bytes = self._image_to_bytes(observation)

        # Start new conversation for this turn
        self._messages = [{"role": "user", "content": game_state_text}]

        # Update tool executor with current turn for rate limiting
        if self.tool_executor:
            self.tool_executor._current_turn = self._turn_count

        # Get situation from decision context for tool filtering
        context = get_decision_context(info, self.knowledge)
        situation = context.get("situation", "overworld")

        # Filter tools by situation to reduce tokens sent to LLM
        if situation == "battle":
            # Only battle-relevant tools
            battle_tools = {"attack", "run_away", "switch_pokemon", "think"}
            available_tools = [t for t in TOOL_DEFINITIONS
                               if t["name"] in battle_tools and t["name"] not in self._excluded_tools]
        elif situation == "dialogue":
            # Only dialogue tools
            dialogue_tools = {"press_button", "think"}
            available_tools = [t for t in TOOL_DEFINITIONS
                               if t["name"] in dialogue_tools and t["name"] not in self._excluded_tools]
        else:  # overworld, menu_choice
            # Exclude battle-only tools
            battle_only = {"attack", "run_away", "switch_pokemon"}
            available_tools = [t for t in TOOL_DEFINITIONS
                               if t["name"] not in battle_only and t["name"] not in self._excluded_tools]

        # Clear exclusions for next turn (will be set again if needed)
        tools_used_this_turn: set[str] = set()

        # Tool calling loop
        last_button = 0
        for i in range(self.config.max_tool_calls_per_turn):
            try:
                # Call LLM with tools
                response = self.provider.complete_with_tools(
                    system_prompt=SYSTEM_PROMPT,
                    messages=self._messages,
                    tools=available_tools,
                    image=image_bytes if i == 0 else None,  # Only send image on first call
                )

                self._total_tokens += response.usage.get("input", 0) + response.usage.get("output", 0)

                if self.config.log_conversation:
                    self._log_response(response)

                # Check if Claude called any tools
                if not response.tool_calls:
                    # No tool calls - Claude is done
                    if response.content:
                        logger.debug(f"Claude says: {response.content}")
                    break

                # Execute each tool call
                tool_results = []
                for tool_call in response.tool_calls:
                    self._last_tool_call = f"{tool_call.name}({tool_call.input})"
                    logger.debug(f"Tool call: {self._last_tool_call}")

                    # Track tools that shouldn't be called consecutively
                    if tool_call.name in ("query_knowledge", "think"):
                        tools_used_this_turn.add(tool_call.name)

                    result = self.tool_executor.execute(tool_call.name, tool_call.input)

                    if self.config.log_conversation:
                        print(f"  → {tool_call.name}: {result.message}")

                    # Format tool result for Claude
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": self._format_tool_result(result),
                    })

                    # If tool returned a screenshot, use it for next iteration
                    if result.screenshot is not None:
                        image_bytes = self._image_to_bytes(result.screenshot)

                # Add assistant's tool calls to history
                self._messages.append({
                    "role": "assistant",
                    "content": response.raw_response.get("content", [])
                })

                # Add tool results to history
                self._messages.append({
                    "role": "user",
                    "content": tool_results
                })

                # Truncate conversation history if too long (keep first + last N messages)
                # This prevents token explosion from many tool calls in one turn
                max_messages = 6  # Initial + up to 2 tool call rounds (4 messages each)
                if len(self._messages) > max_messages:
                    # Keep first message (game state) and last N-1 messages
                    self._messages = [self._messages[0]] + self._messages[-(max_messages - 1):]

                # Check stop reason
                if response.stop_reason == "end_turn":
                    break

            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                break

        # Exclude no-action tools from next turn to prevent loops
        self._excluded_tools = tools_used_this_turn

        return last_button

    def _format_game_state(self, info: dict[str, Any]) -> str:
        """Format game state for the prompt using semantic decision context."""
        lines = []

        # Get semantic decision context
        context = get_decision_context(info, self.knowledge)

        # Header with situation and status
        map_id = info.get("map_id", 0)
        map_name = self.knowledge.get_map_name(map_id)
        lines.append(f"=== {context['situation'].upper()} ===")
        lines.append(f"Location: {map_name} ({context['location_type']})")
        lines.append(f"Party: {context['party_summary']}")
        lines.append(f"Health: {context['health_status']} | Badges: {info.get('badges', 0)}/8")

        # Warnings (critical info)
        if context['warnings']:
            lines.append("")
            for warning in context['warnings']:
                lines.append(f"WARNING: {warning}")

        # Suggested action
        lines.append("")
        lines.append(f"Suggested: {context['suggested_action']}")

        # Situation-specific details
        if context['situation'] == "battle":
            battle_ctx = context.get('battle_context', {})
            lines.append("")
            lines.append("--- Battle Details ---")
            if battle_ctx.get('your_pokemon'):
                lines.append(f"You: {battle_ctx['your_pokemon']}")
            if battle_ctx.get('enemy_pokemon'):
                lines.append(f"Enemy: {battle_ctx['enemy_pokemon']}")
            if battle_ctx.get('advantage'):
                lines.append(f"Assessment: {battle_ctx['advantage']}")
            if battle_ctx.get('available_moves'):
                lines.append(f"Your moves: {', '.join(battle_ctx['available_moves'])}")
            if battle_ctx.get('menu_ready'):
                lines.append("Menu ready - select attack(1-4) or run_away()")

        elif context['situation'] == "dialogue":
            lines.append("")
            lines.append("Press A to advance dialogue.")

        elif context['situation'] == "menu_choice":
            lines.append("")
            lines.append("Menu active. Press A to confirm, B to cancel.")

        else:  # overworld
            lines.append("")
            lines.append("--- Navigation ---")
            lines.append("Use navigator(x, y) to move. Your position is marked @ in the grid.")

            # Check for oscillation (pacing back and forth)
            if len(self._position_history) >= 4:
                recent = self._position_history[-6:]
                unique_positions = set(recent)
                if len(unique_positions) <= 2:
                    lines.append("")
                    lines.append("!!! STUCK: Pacing between same positions !!!")
                    # Provide location-specific guidance
                    from src.agents.llm.decision_context import _get_location_type
                    loc_type = _get_location_type(info)
                    if loc_type == "town":
                        lines.append("You're in a TOWN with no wild Pokemon.")
                        lines.append("Use find_route() to find reachable edge tiles and leave!")
                    else:
                        lines.append("Use find_route() to find reachable edges, or try different directions.")

            # Add walkability grid
            emulator = self.env.emulator if self.env else None
            if emulator:
                try:
                    walkability = format_walkability_for_prompt(emulator)
                    lines.append("")
                    lines.append(walkability)
                except Exception:
                    pass

        # Knowledge base (minimal - goal and vital notes only)
        lines.append("")
        lines.append("--- Knowledge ---")
        lines.append(self.knowledge.format_for_prompt())

        return "\n".join(lines)

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format a tool result as text for Claude. Kept compact to save tokens."""
        lines = [result.message]

        # Only show game state for significant changes (battle status)
        if result.game_state:
            gs = result.game_state
            # Compact one-line status
            status_parts = []
            if gs.get("in_battle"):
                status_parts.append("IN BATTLE")
                if gs.get("battle", {}).get("menu_ready"):
                    status_parts.append("menu ready")
            if gs.get("location"):
                status_parts.append(gs["location"])
            if gs.get("hp_percent") is not None:
                status_parts.append(f"HP:{gs['hp_percent']}%")
            if status_parts:
                lines.append(f"[{' | '.join(status_parts)}]")

        # Walkability grid (already compact in the result)
        if result.walkability_text:
            lines.append("")
            lines.append(result.walkability_text)

        return "\n".join(lines)

    def _image_to_bytes(self, image: NDArray[np.uint8]) -> bytes:
        """Convert numpy image to JPEG bytes (smaller than PNG for game graphics)."""
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def _log_response(self, response) -> None:
        """Log the LLM response."""
        print(f"\n{'='*50}")
        print(f"Turn {self._turn_count} (tokens: {self._total_tokens})")
        print(f"{'='*50}")
        if response.content:
            print(f"Claude: {response.content}")
        if response.tool_calls:
            print(f"Tool calls: {[f'{tc.name}({tc.input})' for tc in response.tool_calls]}")

    def _detect_transition_direction(
        self, last_pos: tuple[int, int], new_pos: tuple[int, int]
    ) -> str:
        """
        Detect which direction the player moved to trigger a map transition.

        Uses the last position's proximity to map edges. When you're at the
        north edge (low y) of a map and transition, you went north.
        Different maps have different coordinate systems, so we can't rely
        on coordinate changes.
        """
        last_x, last_y = last_pos

        # Determine direction based on which edge of the OLD map the player was at
        # Pokemon Red maps typically have their origin at northwest corner
        # Low y = north edge, High y = south edge
        # Low x = west edge, High x = east edge

        # Check y-axis edges first (most common for route connections)
        if last_y <= 3:
            return "north"  # Was at north edge, went north
        elif last_y >= 12:
            return "south"  # Was at south edge, went south

        # Check x-axis edges
        if last_x <= 3:
            return "west"  # Was at west edge, went west
        elif last_x >= 12:
            return "east"  # Was at east edge, went east

        # If position wasn't clearly at an edge, use secondary heuristics
        # Check if position suggests they were at an edge of a small map
        if last_y <= 5:
            return "north"
        elif last_y >= 8:
            return "south"
        elif last_x <= 5:
            return "west"
        elif last_x >= 8:
            return "east"

        return "unknown"

    def reset(self) -> None:
        """Reset agent state."""
        self._messages = []
        self._turn_count = 0
        self._last_tool_call = "None"
        self._position_history = []
        self._last_map_id = None
        self._last_position = None
        # Keep knowledge base and token count

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
        pass

    @property
    def last_action_name(self) -> str:
        """Get the name of the last tool call."""
        return self._last_tool_call

    @property
    def tokens_used(self) -> int:
        """Get total tokens used."""
        return self._total_tokens
