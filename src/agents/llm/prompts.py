"""Prompt templates for LLM agents."""

from typing import Any


SYSTEM_PROMPT = """You are an AI playing Pokemon Red on Game Boy. Your goal is to progress through the game: catch Pokemon, defeat gym leaders, and become the Pokemon Champion.

You will receive game state information and must choose ONE action from the available actions. Respond with ONLY the action name, nothing else.

AVAILABLE ACTIONS:
- MOVE_UP: Move north/up on the map
- MOVE_DOWN: Move south/down on the map
- MOVE_LEFT: Move west/left on the map
- MOVE_RIGHT: Move east/right on the map
- PRESS_A: Confirm selection, interact with objects/NPCs, advance dialogue
- PRESS_B: Cancel, go back, exit menus
- USE_MOVE_1: In battle, use the first move
- USE_MOVE_2: In battle, use the second move
- USE_MOVE_3: In battle, use the third move
- USE_MOVE_4: In battle, use the fourth move
- FLEE: In battle, attempt to run away
- OPEN_MENU: Open the start menu
- WAIT: Do nothing this turn

STRATEGY TIPS:
- In battle, use your moves to defeat enemies
- Explore new areas to find items and Pokemon
- Heal at Pokemon Centers when HP is low
- Talk to NPCs for hints and items
- If stuck in a menu, use PRESS_B to back out
- If stuck against a wall, try a different direction
"""


def format_game_state_prompt(
    info: dict[str, Any],
    include_history: bool = True,
    recent_actions: list[str] | None = None,
) -> str:
    """
    Format the game state into a prompt for the LLM.

    Args:
        info: Game state information dict from environment
        include_history: Whether to include recent action history
        recent_actions: List of recent action names

    Returns:
        Formatted prompt string
    """
    lines = ["CURRENT GAME STATE:"]

    # Location
    position = info.get("position", (0, 0))
    lines.append(f"Location: Map {info.get('map_id', 'unknown')}, Position ({position[0]}, {position[1]})")

    # Progress
    lines.append(f"Badges: {info.get('badges', 0)}/8")
    lines.append(f"Pokemon Caught: {info.get('pokemon_caught', 0)}")
    lines.append(f"Total Party Level: {info.get('total_level', 0)}")

    # Party HP
    party_hp = info.get("party_hp", 0)
    party_max = info.get("party_max_hp", 1)
    hp_percent = (party_hp / party_max * 100) if party_max > 0 else 0
    lines.append(f"Party HP: {party_hp}/{party_max} ({hp_percent:.0f}%)")
    lines.append(f"Party Alive: {info.get('party_alive', 0)} Pokemon")

    # Exploration
    lines.append(f"Maps Explored: {info.get('maps_visited', 0)}")

    # Battle state
    if info.get("in_battle", False):
        lines.append("")
        lines.append("** IN BATTLE **")
        enemy_hp = info.get("enemy_hp", 0)
        enemy_max = info.get("enemy_max_hp", 1)
        enemy_percent = (enemy_hp / enemy_max * 100) if enemy_max > 0 else 0
        lines.append(f"Enemy HP: {enemy_hp}/{enemy_max} ({enemy_percent:.0f}%)")
        lines.append(f"Enemy Level: {info.get('enemy_level', 0)}")
        lines.append("")
        lines.append("Battle actions: USE_MOVE_1, USE_MOVE_2, USE_MOVE_3, USE_MOVE_4, FLEE")
    else:
        lines.append("")
        lines.append("Not in battle - explore, interact, or navigate!")

    # Recent actions (to help avoid loops)
    if include_history and recent_actions:
        recent = recent_actions[-5:]
        lines.append(f"")
        lines.append(f"Recent actions: {', '.join(recent)}")

    lines.append("")
    lines.append("What action should I take? (respond with ONLY the action name)")

    return "\n".join(lines)


def format_battle_prompt(info: dict[str, Any]) -> str:
    """
    Specialized prompt for battle situations.

    Args:
        info: Game state information dict

    Returns:
        Battle-focused prompt string
    """
    lines = ["BATTLE SITUATION:"]

    party_hp = info.get("party_hp", 0)
    party_max = info.get("party_max_hp", 1)
    enemy_hp = info.get("enemy_hp", 0)
    enemy_max = info.get("enemy_max_hp", 1)

    lines.append(f"Your Party HP: {party_hp}/{party_max}")
    lines.append(f"Enemy HP: {enemy_hp}/{enemy_max}, Level {info.get('enemy_level', 0)}")

    # Tactical advice
    if party_hp < party_max * 0.2:
        lines.append("")
        lines.append("WARNING: HP is critically low! Consider fleeing.")

    if enemy_hp < enemy_max * 0.2:
        lines.append("")
        lines.append("Enemy is nearly defeated! Finish them off!")

    lines.append("")
    lines.append("Choose: USE_MOVE_1, USE_MOVE_2, USE_MOVE_3, USE_MOVE_4, or FLEE")

    return "\n".join(lines)
