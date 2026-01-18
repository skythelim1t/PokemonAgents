"""Strategic action definitions for LLM agents."""

import re
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class MoveToAction:
    """Represents a MOVE_TO action with target coordinates."""
    target_x: int
    target_y: int

    def __str__(self) -> str:
        return f"MOVE_TO ({self.target_x}, {self.target_y})"


class StrategicAction(Enum):
    """
    High-level strategic actions the LLM can choose.

    These are translated by the Executor into button sequences.
    """

    # === EXPLORATION (Executor walks until obstacle/event) ===
    EXPLORE_UP = auto()      # Walk north until blocked or encounter
    EXPLORE_DOWN = auto()    # Walk south until blocked or encounter
    EXPLORE_LEFT = auto()    # Walk west until blocked or encounter
    EXPLORE_RIGHT = auto()   # Walk east until blocked or encounter

    # === COORDINATE MOVEMENT (Executor navigates to target) ===
    MOVE_TO = auto()         # Navigate to specific (x, y) coordinates

    # === INTERACTION ===
    INTERACT = auto()        # Press A - talk to NPC, read sign, pick up item
    CANCEL = auto()          # Press B - close menu, cancel action
    OPEN_MENU = auto()       # Press START - open main menu

    # === LOCATIONS (Executor handles full sequence) ===
    HEAL_AT_POKECENTER = auto()  # Enter center, heal, exit

    # === BATTLE ACTIONS (LLM chooses, Executor navigates menu) ===
    ATTACK_1 = auto()        # Use move in slot 1
    ATTACK_2 = auto()        # Use move in slot 2
    ATTACK_3 = auto()        # Use move in slot 3
    ATTACK_4 = auto()        # Use move in slot 4
    RUN_AWAY = auto()        # Attempt to flee battle

    # === WAITING ===
    WAIT = auto()            # Do nothing this turn

    # === KNOWLEDGE/NOTES (handled by agent, not executor) ===
    # These are parsed specially - format: "NOTE: <text>" or "GOAL: <text>"
    ADD_NOTE = auto()        # Add a note (parsed from response)
    SET_GOAL = auto()        # Set current goal (parsed from response)


# Map actions to simple descriptions for prompts
ACTION_DESCRIPTIONS = {
    StrategicAction.EXPLORE_UP: "Walk north until you hit something or encounter a Pokemon",
    StrategicAction.EXPLORE_DOWN: "Walk south until you hit something or encounter a Pokemon",
    StrategicAction.EXPLORE_LEFT: "Walk west until you hit something or encounter a Pokemon",
    StrategicAction.EXPLORE_RIGHT: "Walk east until you hit something or encounter a Pokemon",
    StrategicAction.INTERACT: "Talk to NPC, read sign, advance dialogue, confirm selection",
    StrategicAction.CANCEL: "Close menu, go back, cancel current action",
    StrategicAction.OPEN_MENU: "Open the START menu",
    StrategicAction.HEAL_AT_POKECENTER: "Go to Pokemon Center and heal your team",
    StrategicAction.ATTACK_1: "Use your Pokemon's first move",
    StrategicAction.ATTACK_2: "Use your Pokemon's second move",
    StrategicAction.ATTACK_3: "Use your Pokemon's third move",
    StrategicAction.ATTACK_4: "Use your Pokemon's fourth move",
    StrategicAction.RUN_AWAY: "Try to run away from battle",
    StrategicAction.WAIT: "Wait and do nothing",
}


def parse_move_to(action_str: str) -> MoveToAction | None:
    """Parse MOVE_TO (x, y) from string. Returns MoveToAction or None."""
    # Match patterns like: MOVE_TO (5, 10), MOVE_TO 5 10, MOVE_TO(5,10), GO_TO (5, 10)
    patterns = [
        r"(?:MOVE_TO|GO_TO|NAVIGATE_TO|WALK_TO)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
        r"(?:MOVE_TO|GO_TO|NAVIGATE_TO|WALK_TO)\s+(\d+)\s+(\d+)",
        r"(?:MOVE_TO|GO_TO|NAVIGATE_TO|WALK_TO)\s+(\d+)\s*,\s*(\d+)",
    ]

    action_upper = action_str.strip().upper()
    for pattern in patterns:
        match = re.search(pattern, action_upper)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return MoveToAction(target_x=x, target_y=y)

    return None


def action_from_string(action_str: str) -> StrategicAction | MoveToAction | None:
    """Parse a string into a StrategicAction or MoveToAction."""
    # First check for MOVE_TO with coordinates
    move_to = parse_move_to(action_str)
    if move_to:
        return move_to

    action_str = action_str.strip().upper().replace(" ", "_").replace("-", "_")

    # Direct enum name match
    for action in StrategicAction:
        if action.name == action_str:
            return action

    # Fuzzy matching for common variations
    mappings = {
        # Exploration
        "UP": StrategicAction.EXPLORE_UP,
        "DOWN": StrategicAction.EXPLORE_DOWN,
        "LEFT": StrategicAction.EXPLORE_LEFT,
        "RIGHT": StrategicAction.EXPLORE_RIGHT,
        "NORTH": StrategicAction.EXPLORE_UP,
        "SOUTH": StrategicAction.EXPLORE_DOWN,
        "WEST": StrategicAction.EXPLORE_LEFT,
        "EAST": StrategicAction.EXPLORE_RIGHT,
        "GO_UP": StrategicAction.EXPLORE_UP,
        "GO_DOWN": StrategicAction.EXPLORE_DOWN,
        "GO_LEFT": StrategicAction.EXPLORE_LEFT,
        "GO_RIGHT": StrategicAction.EXPLORE_RIGHT,
        "MOVE_UP": StrategicAction.EXPLORE_UP,
        "MOVE_DOWN": StrategicAction.EXPLORE_DOWN,
        "MOVE_LEFT": StrategicAction.EXPLORE_LEFT,
        "MOVE_RIGHT": StrategicAction.EXPLORE_RIGHT,

        # Interaction
        "A": StrategicAction.INTERACT,
        "PRESS_A": StrategicAction.INTERACT,
        "TALK": StrategicAction.INTERACT,
        "CONFIRM": StrategicAction.INTERACT,
        "SELECT": StrategicAction.INTERACT,
        "YES": StrategicAction.INTERACT,
        "B": StrategicAction.CANCEL,
        "PRESS_B": StrategicAction.CANCEL,
        "BACK": StrategicAction.CANCEL,
        "NO": StrategicAction.CANCEL,
        "EXIT": StrategicAction.CANCEL,
        "START": StrategicAction.OPEN_MENU,
        "MENU": StrategicAction.OPEN_MENU,

        # Healing
        "HEAL": StrategicAction.HEAL_AT_POKECENTER,
        "POKECENTER": StrategicAction.HEAL_AT_POKECENTER,
        "REST": StrategicAction.HEAL_AT_POKECENTER,

        # Battle
        "ATTACK": StrategicAction.ATTACK_1,
        "FIGHT": StrategicAction.ATTACK_1,
        "MOVE_1": StrategicAction.ATTACK_1,
        "MOVE_2": StrategicAction.ATTACK_2,
        "MOVE_3": StrategicAction.ATTACK_3,
        "MOVE_4": StrategicAction.ATTACK_4,
        "USE_MOVE_1": StrategicAction.ATTACK_1,
        "USE_MOVE_2": StrategicAction.ATTACK_2,
        "USE_MOVE_3": StrategicAction.ATTACK_3,
        "USE_MOVE_4": StrategicAction.ATTACK_4,
        "RUN": StrategicAction.RUN_AWAY,
        "FLEE": StrategicAction.RUN_AWAY,
        "ESCAPE": StrategicAction.RUN_AWAY,

        # Wait
        "NOTHING": StrategicAction.WAIT,
        "SKIP": StrategicAction.WAIT,
    }

    if action_str in mappings:
        return mappings[action_str]

    # Partial match - check if action name is contained
    for action in StrategicAction:
        if action.name in action_str or action_str in action.name:
            return action

    return None
