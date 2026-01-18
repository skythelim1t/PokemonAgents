"""Executor layer for translating strategic actions to button sequences."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.agents.llm.actions import StrategicAction, MoveToAction
from src.agents.llm.screen_navigator import (
    screen_to_world_coords,
    SCREEN_WIDTH_TILES,
    SCREEN_HEIGHT_TILES,
)


class ExecutorState(Enum):
    """State of the executor."""
    IDLE = auto()           # No action in progress
    EXECUTING = auto()      # Action in progress
    COMPLETED = auto()      # Action finished successfully
    INTERRUPTED = auto()    # Action was interrupted (battle started, etc.)


@dataclass
class ExecutorResult:
    """Result of executor step."""
    button: int             # Button to press (0-7)
    state: ExecutorState    # Current executor state
    steps_taken: int        # Steps taken for current action
    message: str = ""       # Optional status message


# Button indices: ["a", "b", "start", "select", "up", "down", "left", "right"]
BUTTON_A = 0
BUTTON_B = 1
BUTTON_START = 2
BUTTON_SELECT = 3
BUTTON_UP = 4
BUTTON_DOWN = 5
BUTTON_LEFT = 6
BUTTON_RIGHT = 7


class ActionExecutor:
    """
    Translates high-level strategic actions into button press sequences.

    The executor maintains state and generates appropriate button presses
    until the action is complete or interrupted.
    """

    def __init__(self, max_steps_per_action: int = 50) -> None:
        """
        Initialize the executor.

        Args:
            max_steps_per_action: Maximum steps before giving up on an action.
        """
        self.max_steps = max_steps_per_action
        self._current_action: StrategicAction | MoveToAction | None = None
        self._steps_taken: int = 0
        self._state: ExecutorState = ExecutorState.IDLE

        # Track state for detecting completion
        self._start_position: tuple[int, int] | None = None
        self._start_map: int | None = None
        self._start_in_battle: bool = False
        self._start_hp: int = 0

        # MOVE_TO target coordinates
        self._target_x: int | None = None
        self._target_y: int | None = None

        # Pathfinding state for obstacle avoidance
        self._last_position: tuple[int, int] | None = None
        self._blocked_directions: set[int] = set()  # Directions blocked at current position
        self._stuck_count: int = 0  # How many steps we haven't moved
        self._visited_positions: set[tuple[int, int]] = set()  # Avoid revisiting

        # Smart EXPLORE obstacle avoidance state
        self._explore_target_direction: int | None = None  # Original direction LLM wanted
        self._explore_going_around: bool = False  # Currently navigating around obstacle
        self._explore_last_pos: tuple[int, int] | None = None  # Track movement for EXPLORE
        self._explore_stuck_count: int = 0  # Steps without movement
        self._explore_sideways_steps: int = 0  # How many steps we've gone sideways
        self._explore_sideways_dir: int | None = None  # Which perpendicular direction we're using
        self._explore_retry_interval: int = 4  # Try original direction every N sideways moves

    def start_action(self, action: StrategicAction | MoveToAction, info: dict[str, Any]) -> None:
        """
        Start executing a new action.

        Args:
            action: The strategic action to execute (or MoveToAction with coordinates).
            info: Current game state info.
        """
        self._current_action = action
        self._steps_taken = 0
        self._state = ExecutorState.EXECUTING

        # Store starting state to detect changes
        self._start_position = info.get("position", (0, 0))
        self._start_map = info.get("map_id", 0)
        self._start_in_battle = info.get("in_battle", False)
        self._start_hp = info.get("party_hp", 0)

        # Handle MOVE_TO coordinates (screen coordinates -> world coordinates)
        if isinstance(action, MoveToAction):
            # The LLM provides SCREEN coordinates (0-9 x, 0-8 y)
            # We need to convert to WORLD coordinates based on player position
            screen_x = action.target_x
            screen_y = action.target_y

            # Validate screen coordinates are in bounds
            screen_x = max(0, min(screen_x, SCREEN_WIDTH_TILES - 1))
            screen_y = max(0, min(screen_y, SCREEN_HEIGHT_TILES - 1))

            # Convert screen coords to world coords
            player_world_x, player_world_y = self._start_position
            world_x, world_y = screen_to_world_coords(
                screen_x, screen_y, player_world_x, player_world_y
            )

            self._target_x = world_x
            self._target_y = world_y

            # Reset pathfinding state
            self._last_position = None
            self._blocked_directions = set()
            self._stuck_count = 0
            self._visited_positions = set()
        else:
            self._target_x = None
            self._target_y = None

        # Reset EXPLORE state
        self._explore_target_direction = None
        self._explore_going_around = False
        self._explore_last_pos = None
        self._explore_stuck_count = 0
        self._explore_sideways_steps = 0
        self._explore_sideways_dir = None

    def step(self, info: dict[str, Any]) -> ExecutorResult:
        """
        Execute one step of the current action.

        Args:
            info: Current game state info.

        Returns:
            ExecutorResult with button to press and executor state.
        """
        if self._current_action is None or self._state != ExecutorState.EXECUTING:
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.IDLE,
                steps_taken=0,
                message="No action in progress"
            )

        self._steps_taken += 1

        # Check for interrupts (battle started during exploration, etc.)
        if self._check_interrupted(info):
            self._state = ExecutorState.INTERRUPTED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.INTERRUPTED,
                steps_taken=self._steps_taken,
                message="Action interrupted"
            )

        # Check if max steps reached
        if self._steps_taken >= self.max_steps:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Max steps reached"
            )

        # Execute based on action type
        action = self._current_action

        # === COORDINATE MOVEMENT ===
        if isinstance(action, MoveToAction):
            return self._execute_move_to(info)

        # === EXPLORATION ACTIONS ===
        if action == StrategicAction.EXPLORE_UP:
            return self._execute_explore(BUTTON_UP, info)
        elif action == StrategicAction.EXPLORE_DOWN:
            return self._execute_explore(BUTTON_DOWN, info)
        elif action == StrategicAction.EXPLORE_LEFT:
            return self._execute_explore(BUTTON_LEFT, info)
        elif action == StrategicAction.EXPLORE_RIGHT:
            return self._execute_explore(BUTTON_RIGHT, info)

        # === INTERACTION ACTIONS ===
        elif action == StrategicAction.INTERACT:
            return self._execute_interact(info)
        elif action == StrategicAction.CANCEL:
            return self._execute_simple(BUTTON_B)
        elif action == StrategicAction.OPEN_MENU:
            return self._execute_simple(BUTTON_START)

        # === BATTLE ACTIONS ===
        elif action in (StrategicAction.ATTACK_1, StrategicAction.ATTACK_2,
                       StrategicAction.ATTACK_3, StrategicAction.ATTACK_4):
            return self._execute_attack(action, info)
        elif action == StrategicAction.RUN_AWAY:
            return self._execute_run(info)

        # === COMPLEX ACTIONS ===
        elif action == StrategicAction.HEAL_AT_POKECENTER:
            return self._execute_heal(info)

        # === DEFAULT ===
        else:
            return self._execute_simple(BUTTON_A)

    def _check_interrupted(self, info: dict[str, Any]) -> bool:
        """Check if the current action should be interrupted."""
        # Battle started during exploration
        if not self._start_in_battle and info.get("in_battle", False):
            if self._current_action in (
                StrategicAction.EXPLORE_UP, StrategicAction.EXPLORE_DOWN,
                StrategicAction.EXPLORE_LEFT, StrategicAction.EXPLORE_RIGHT,
                StrategicAction.HEAL_AT_POKECENTER
            ):
                return True
        return False

    def _execute_simple(self, button: int) -> ExecutorResult:
        """Execute a simple single-button action."""
        self._state = ExecutorState.COMPLETED
        return ExecutorResult(
            button=button,
            state=ExecutorState.COMPLETED,
            steps_taken=self._steps_taken,
            message="Simple action completed"
        )

    def _execute_interact(self, info: dict[str, Any]) -> ExecutorResult:
        """Execute INTERACT - press A and wait for text/dialogue to render."""
        # Press A on first step
        if self._steps_taken == 1:
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.EXECUTING,
                steps_taken=self._steps_taken,
                message="Pressing A"
            )

        # Wait a few steps for text to render before completing
        # This gives time for dialogue boxes to appear and text to print
        min_wait_steps = 8

        if self._steps_taken >= min_wait_steps:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,  # Can press A again to help advance
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Interaction complete"
            )

        # Keep waiting (press nothing or A to help advance text)
        return ExecutorResult(
            button=BUTTON_A,
            state=ExecutorState.EXECUTING,
            steps_taken=self._steps_taken,
            message=f"Waiting for text ({self._steps_taken}/{min_wait_steps})"
        )

    def _execute_explore(self, direction: int, info: dict[str, Any]) -> ExecutorResult:
        """Execute exploration - walk in direction, complete quickly if blocked.

        When blocked, completes immediately so the LLM can use vision to decide
        the best way around the obstacle (instead of blindly trying directions).
        """
        current_pos = info.get("position", (0, 0))
        current_map = info.get("map_id", 0)

        # Check if we moved to new map
        if current_map != self._start_map:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=direction,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Reached new map"
            )

        # Track if we moved since last step
        if self._explore_last_pos is not None:
            if current_pos != self._explore_last_pos:
                self._explore_stuck_count = 0
            else:
                self._explore_stuck_count += 1
        self._explore_last_pos = current_pos

        # If blocked for 3+ steps, complete so LLM can choose better direction with vision
        if self._explore_stuck_count >= 3:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=direction,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Blocked - use vision to find way around"
            )

        # If we've been walking for a while and moved, that's progress
        if self._steps_taken >= 8 and current_pos != self._start_position:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=direction,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Moved - checking surroundings"
            )

        # Give up after max steps
        if self._steps_taken >= self.max_steps:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=direction,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Max steps reached"
            )

        # Keep walking in target direction
        return ExecutorResult(
            button=direction,
            state=ExecutorState.EXECUTING,
            steps_taken=self._steps_taken,
            message=f"Walking {self._steps_taken}/{self.max_steps}"
        )

    def _execute_move_to(self, info: dict[str, Any]) -> ExecutorResult:
        """Execute MOVE_TO - navigate to target with obstacle avoidance."""
        current_pos = info.get("position", (0, 0))
        current_x, current_y = current_pos
        current_map = info.get("map_id", 0)

        target_x = self._target_x
        target_y = self._target_y

        # Check if we reached the target
        if current_x == target_x and current_y == target_y:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message=f"Reached target ({target_x}, {target_y})"
            )

        # Check if we changed maps (might have exited/entered building)
        if current_map != self._start_map:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Map changed during navigation"
            )

        # Track if we moved since last step
        if self._last_position is not None:
            if current_pos == self._last_position:
                # Didn't move - mark the direction we tried as blocked
                self._stuck_count += 1
            else:
                # We moved! Clear blocked directions for new position
                self._blocked_directions = set()
                self._stuck_count = 0
                self._visited_positions.add(current_pos)

        self._last_position = current_pos

        # Calculate direction to target
        dx = target_x - current_x
        dy = target_y - current_y

        # Build list of directions to try, prioritized by distance to target
        # In Pokemon, Y increases going DOWN, decreases going UP
        directions_to_try = []

        if abs(dy) > abs(dx):
            # Prioritize vertical
            if dy < 0:
                directions_to_try.append((BUTTON_UP, "up"))
            if dy > 0:
                directions_to_try.append((BUTTON_DOWN, "down"))
            if dx < 0:
                directions_to_try.append((BUTTON_LEFT, "left"))
            if dx > 0:
                directions_to_try.append((BUTTON_RIGHT, "right"))
        else:
            # Prioritize horizontal
            if dx < 0:
                directions_to_try.append((BUTTON_LEFT, "left"))
            if dx > 0:
                directions_to_try.append((BUTTON_RIGHT, "right"))
            if dy < 0:
                directions_to_try.append((BUTTON_UP, "up"))
            if dy > 0:
                directions_to_try.append((BUTTON_DOWN, "down"))

        # Add perpendicular directions for obstacle avoidance
        all_dirs = [
            (BUTTON_UP, "up"), (BUTTON_DOWN, "down"),
            (BUTTON_LEFT, "left"), (BUTTON_RIGHT, "right")
        ]
        for d in all_dirs:
            if d not in directions_to_try:
                directions_to_try.append(d)

        # Find first non-blocked direction
        chosen_direction = None
        chosen_name = ""

        for direction, name in directions_to_try:
            if direction not in self._blocked_directions:
                chosen_direction = direction
                chosen_name = name
                break

        # If all directions blocked at this position, we're stuck
        if chosen_direction is None:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message=f"All directions blocked at ({current_x}, {current_y})"
            )

        # Mark this direction as tried (will be confirmed blocked if we don't move)
        if self._stuck_count >= 2:
            # We've been stuck trying this direction, mark it blocked
            self._blocked_directions.add(chosen_direction)

        # Give up after too many total steps or too many blocked directions
        if self._steps_taken > self.max_steps or len(self._blocked_directions) >= 4:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message=f"Cannot reach ({target_x}, {target_y}) - path blocked"
            )

        distance = abs(dx) + abs(dy)
        blocked_info = f", blocked:{len(self._blocked_directions)}" if self._blocked_directions else ""
        return ExecutorResult(
            button=chosen_direction,
            state=ExecutorState.EXECUTING,
            steps_taken=self._steps_taken,
            message=f"Moving {chosen_name} to ({target_x}, {target_y}), dist={distance}{blocked_info}"
        )

    def _execute_attack(self, action: StrategicAction, info: dict[str, Any]) -> ExecutorResult:
        """Execute a battle attack - navigate to move, select it, wait for turn to complete."""
        if not info.get("in_battle", False):
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Not in battle"
            )

        # Battle attack sequence:
        # Steps 1-2: Press A to enter FIGHT menu
        # Steps 3-N: Navigate to move
        # Steps N+1-N+2: Press A to confirm move
        # Steps N+3+: Press A to advance through battle text (our attack, enemy attack, etc.)
        # Complete when: we've pressed A enough times to get through a full turn

        move_num = {
            StrategicAction.ATTACK_1: 0,
            StrategicAction.ATTACK_2: 1,
            StrategicAction.ATTACK_3: 2,
            StrategicAction.ATTACK_4: 3,
        }.get(action, 0)

        # Phase 1: Open FIGHT menu (steps 1-2)
        if self._steps_taken <= 2:
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.EXECUTING,
                steps_taken=self._steps_taken,
                message="Opening FIGHT menu"
            )

        # Phase 2: Navigate to the right move (steps 3 to 2+move_num)
        elif self._steps_taken <= 2 + move_num:
            return ExecutorResult(
                button=BUTTON_DOWN,
                state=ExecutorState.EXECUTING,
                steps_taken=self._steps_taken,
                message=f"Selecting move {move_num + 1}"
            )

        # Phase 3: Confirm move selection (steps after navigation, 2 presses)
        elif self._steps_taken <= 4 + move_num:
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.EXECUTING,
                steps_taken=self._steps_taken,
                message="Confirming move"
            )

        # Phase 4: Wait for battle animations and advance text
        # Keep pressing A until the battle menu is ready again
        # This means the full turn (our attack + enemy attack) is complete
        # Be conservative - better to wait too long than miss the results
        else:
            min_steps = 30 + move_num  # Wait at least 30 steps before checking menu
            max_steps = 80 + move_num  # Maximum steps to avoid infinite loop

            # Check if battle menu is ready (turn complete)
            battle_menu_ready = info.get("battle_menu_ready", False)

            # Also check if battle ended
            if not info.get("in_battle", False):
                self._state = ExecutorState.COMPLETED
                return ExecutorResult(
                    button=BUTTON_A,
                    state=ExecutorState.COMPLETED,
                    steps_taken=self._steps_taken,
                    message="Battle ended"
                )

            # After minimum steps, check if menu is ready
            if self._steps_taken >= min_steps and battle_menu_ready:
                self._state = ExecutorState.COMPLETED
                return ExecutorResult(
                    button=BUTTON_A,
                    state=ExecutorState.COMPLETED,
                    steps_taken=self._steps_taken,
                    message="Turn complete - menu ready"
                )

            # Safety: don't wait forever
            if self._steps_taken >= max_steps:
                self._state = ExecutorState.COMPLETED
                return ExecutorResult(
                    button=BUTTON_A,
                    state=ExecutorState.COMPLETED,
                    steps_taken=self._steps_taken,
                    message="Turn complete - max steps"
                )

            # Keep pressing A to advance through battle text
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.EXECUTING,
                steps_taken=self._steps_taken,
                message=f"Waiting for turn ({self._steps_taken})"
            )

    def _execute_run(self, info: dict[str, Any]) -> ExecutorResult:
        """Execute RUN from battle and wait for result."""
        # Check if we escaped (not in battle anymore)
        if not info.get("in_battle", False):
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Escaped!"
            )

        # RUN sequence in Pokemon Red battle menu:
        # The menu is: FIGHT | BAG
        #              POKEMON | RUN
        # We need: Right (to BAG column) -> Down (to RUN) -> A (select)
        if self._steps_taken == 1:
            return ExecutorResult(button=BUTTON_RIGHT, state=ExecutorState.EXECUTING,
                                  steps_taken=self._steps_taken, message="Moving to RUN")
        elif self._steps_taken == 2:
            return ExecutorResult(button=BUTTON_DOWN, state=ExecutorState.EXECUTING,
                                  steps_taken=self._steps_taken, message="Moving to RUN")
        elif self._steps_taken <= 4:
            return ExecutorResult(button=BUTTON_A, state=ExecutorState.EXECUTING,
                                  steps_taken=self._steps_taken, message="Selecting RUN")
        # Wait for run result - keep pressing A until menu is back or we escaped
        # Be conservative with wait times
        else:
            battle_menu_ready = info.get("battle_menu_ready", False)

            # If menu is ready again, run failed - enemy will attack
            if self._steps_taken >= 25 and battle_menu_ready:
                self._state = ExecutorState.COMPLETED
                return ExecutorResult(button=BUTTON_A, state=ExecutorState.COMPLETED,
                                      steps_taken=self._steps_taken, message="Run failed - enemy turn")

            # Safety max
            if self._steps_taken >= 60:
                self._state = ExecutorState.COMPLETED
                return ExecutorResult(button=BUTTON_A, state=ExecutorState.COMPLETED,
                                      steps_taken=self._steps_taken, message="Run attempted")

            return ExecutorResult(button=BUTTON_A, state=ExecutorState.EXECUTING,
                                  steps_taken=self._steps_taken, message="Waiting for run result")

    def _execute_heal(self, info: dict[str, Any]) -> ExecutorResult:
        """
        Execute HEAL_AT_POKECENTER - complex multi-step action.

        This is simplified: just keeps pressing A to advance through
        Pokemon Center dialogue. Assumes player is already near/in the center.
        """
        # For now, just spam A to heal if we're already in center
        # A full implementation would need pathfinding
        if self._steps_taken >= 20:
            self._state = ExecutorState.COMPLETED
            return ExecutorResult(
                button=BUTTON_A,
                state=ExecutorState.COMPLETED,
                steps_taken=self._steps_taken,
                message="Heal sequence done"
            )

        return ExecutorResult(
            button=BUTTON_A,
            state=ExecutorState.EXECUTING,
            steps_taken=self._steps_taken,
            message="Healing..."
        )

    @property
    def is_busy(self) -> bool:
        """Check if executor is currently executing an action."""
        return self._state == ExecutorState.EXECUTING

    @property
    def current_action(self) -> StrategicAction | None:
        """Get the current action being executed."""
        return self._current_action

    @property
    def state(self) -> ExecutorState:
        """Get the executor state."""
        return self._state

    def reset(self) -> None:
        """Reset the executor state."""
        self._current_action = None
        self._steps_taken = 0
        self._state = ExecutorState.IDLE
        self._start_position = None
        self._start_map = None
        self._start_in_battle = False
        self._start_hp = 0
        self._target_x = None
        self._target_y = None
        # Reset EXPLORE state
        self._explore_target_direction = None
        self._explore_going_around = False
        self._explore_last_pos = None
        self._explore_stuck_count = 0
        self._explore_sideways_steps = 0
        self._explore_sideways_dir = None
