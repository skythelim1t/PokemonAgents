"""Strategic action definitions for LLM agents."""

from enum import Enum, auto


class StrategicAction(Enum):
    """High-level strategic actions the LLM can choose."""

    # Navigation
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()

    # Interaction
    PRESS_A = auto()  # Confirm, interact, advance dialogue
    PRESS_B = auto()  # Cancel, go back

    # Battle actions
    USE_MOVE_1 = auto()  # Select first move
    USE_MOVE_2 = auto()  # Select second move
    USE_MOVE_3 = auto()  # Select third move
    USE_MOVE_4 = auto()  # Select fourth move
    FLEE = auto()  # Attempt to flee battle

    # Menu
    OPEN_MENU = auto()  # Press START
    WAIT = auto()  # Do nothing this turn


# Button indices: ["a", "b", "start", "select", "up", "down", "left", "right"]
#                   0     1      2        3       4      5       6       7

# Simple 1:1 mapping for initial version
SIMPLE_ACTION_TO_BUTTON: dict[StrategicAction, int] = {
    StrategicAction.MOVE_UP: 4,  # up
    StrategicAction.MOVE_DOWN: 5,  # down
    StrategicAction.MOVE_LEFT: 6,  # left
    StrategicAction.MOVE_RIGHT: 7,  # right
    StrategicAction.PRESS_A: 0,  # a
    StrategicAction.PRESS_B: 1,  # b
    StrategicAction.USE_MOVE_1: 0,  # a (selects first move in battle)
    StrategicAction.USE_MOVE_2: 5,  # down (navigate to second move)
    StrategicAction.USE_MOVE_3: 5,  # down (navigate further)
    StrategicAction.USE_MOVE_4: 5,  # down (navigate further)
    StrategicAction.FLEE: 1,  # b (attempt to run)
    StrategicAction.OPEN_MENU: 2,  # start
    StrategicAction.WAIT: 0,  # default to a
}

# Full multi-button sequences for complex actions (future use)
ACTION_TO_BUTTONS: dict[StrategicAction, list[int]] = {
    StrategicAction.MOVE_UP: [4],
    StrategicAction.MOVE_DOWN: [5],
    StrategicAction.MOVE_LEFT: [6],
    StrategicAction.MOVE_RIGHT: [7],
    StrategicAction.PRESS_A: [0],
    StrategicAction.PRESS_B: [1],
    StrategicAction.USE_MOVE_1: [0, 0],  # FIGHT -> first move
    StrategicAction.USE_MOVE_2: [0, 5, 0],  # FIGHT -> down -> select
    StrategicAction.USE_MOVE_3: [0, 5, 5, 0],  # FIGHT -> down -> down -> select
    StrategicAction.USE_MOVE_4: [0, 5, 5, 5, 0],  # FIGHT -> down x3 -> select
    StrategicAction.FLEE: [5, 5, 5, 0],  # down to RUN -> select
    StrategicAction.OPEN_MENU: [2],
    StrategicAction.WAIT: [],
}


def action_from_string(action_str: str) -> StrategicAction | None:
    """Parse a string into a StrategicAction."""
    action_str = action_str.strip().upper().replace(" ", "_")

    for action in StrategicAction:
        if action.name == action_str or action.name in action_str:
            return action

    return None
