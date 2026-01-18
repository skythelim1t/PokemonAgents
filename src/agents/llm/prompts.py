"""Prompt templates for LLM agents."""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.emulator.pyboy_wrapper import EmulatorWrapper


# Map-specific navigation targets (map_id -> list of (name, x, y))
# These are approximate useful locations on each map
# Listed in priority order - first target is usually the best destination
# Pallet Town: x=3-12, y=0-11 (exit to Route 1 is at y=0, around x=10-11)
MAP_NAVIGATION_TARGETS: dict[int, list[tuple[str, int, int]]] = {
    # Pallet Town (map 0) - Goal: Go to Route 1 for wild Pokemon
    # The town exit is at the NORTH edge - walk to y=0 to transition to Route 1
    0: [
        ("Route 1 EXIT (north edge)", 10, 1),  # Walk here to exit to Route 1
        ("Your house", 5, 6),
        # Oak's Lab intentionally omitted - don't go back inside
    ],
    # Viridian City (map 1)
    1: [
        ("Pokemon Center (HEAL)", 19, 19),
        ("Route 2 exit (north)", 17, 0),
        ("Pokemart", 22, 19),
        ("Route 1 exit (south)", 17, 35),
    ],
    # Pewter City (map 2)
    2: [
        ("Pokemon Center (HEAL)", 13, 17),
        ("Pewter Gym", 16, 11),
        ("Route 3 exit (east)", 35, 19),
        ("Route 2 exit (south)", 17, 35),
    ],
    # Route 1 (map 12) - Has tall grass for wild Pokemon
    # Note: Route 1 has ledges that block direct north path
    # Tall grass is accessible in southern section (y=30-34)
    12: [
        ("Tall grass (WILD POKEMON)", 11, 33),  # Accessible from Pallet Town
        ("Tall grass area 2", 9, 31),
        ("Pallet Town (south)", 10, 35),
        # Viridian requires going around ledges or entering from north
    ],
    # Route 2 (map 13)
    13: [
        ("Viridian Forest entrance", 4, 25),
        ("Pewter City (north)", 4, 0),
        ("Viridian City (south)", 4, 47),
    ],
    # Oak's Lab (map 40) - Exit is triggered by walking DOWN past y=11
    40: [
        ("EXIT (walk south)", 5, 12),  # Walking here triggers exit to Pallet Town
        ("Oak's position", 4, 2),
        ("Pokeball table", 6, 2),
    ],
    # Viridian Pokemon Center (map 41)
    41: [
        ("Exit door", 3, 7),
        ("Healing counter", 3, 2),
    ],
}


def _add_navigation_suggestions(lines: list[str], map_id: int, current_x: int, current_y: int) -> None:
    """Add navigation target suggestions based on current map."""
    targets = MAP_NAVIGATION_TARGETS.get(map_id, [])
    if not targets:
        # No known targets for this map
        lines.append("")
        lines.append("Suggested: Try moving to edges of the map to find exits")
        return

    lines.append("")
    lines.append("Navigation targets:")
    for name, x, y in targets:
        dx = x - current_x
        dy = y - current_y
        distance = abs(dx) + abs(dy)
        if distance > 0:  # Don't show if already there
            lines.append(f"  - {name}: MOVE_TO ({x}, {y})  [dist: {distance}]")


def _detect_coordinate_oscillation(recent_actions: list[str]) -> tuple[bool, str | None]:
    """
    Detect if the agent is oscillating between the same MOVE_TO coordinates.

    Returns:
        (is_oscillating, suggestion_message)
    """
    if not recent_actions or len(recent_actions) < 4:
        return False, None

    # Extract MOVE_TO coordinates from recent actions
    move_to_coords = []
    for action in recent_actions[-6:]:
        if action.startswith("MOVE_TO"):
            # Parse "MOVE_TO (x, y)" -> (x, y)
            try:
                coords_part = action.replace("MOVE_TO", "").strip()
                coords_part = coords_part.strip("()")
                parts = coords_part.split(",")
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                move_to_coords.append((x, y))
            except (ValueError, IndexError):
                continue

    if len(move_to_coords) < 4:
        return False, None

    # Check for A-B-A-B pattern (oscillation between 2 coordinates)
    unique_coords = set(move_to_coords[-4:])
    if len(unique_coords) <= 2:
        # We're oscillating between at most 2 coordinates
        coords_list = list(unique_coords)
        if len(coords_list) == 2:
            return True, f"You're stuck oscillating between {coords_list[0]} and {coords_list[1]}!"
        else:
            return True, f"You keep going to {coords_list[0]} but nothing is happening!"

    return False, None


SYSTEM_PROMPT = """You are an AI playing Pokemon Red. You navigate using SCREEN COORDINATES.

=== WALKABILITY OVERLAY ===
The screenshot has a color overlay showing what you can walk on:
- GREEN tiles = WALKABLE (paths, grass, doors)
- RED tiles = BLOCKED (trees, water, walls, obstacles)
- BLUE tile = YOU (your position at center)

IMPORTANT: Only MOVE_TO green tiles! Red tiles will block you.

=== SCREEN COORDINATE SYSTEM ===
The screen is a 10x9 tile grid:
- YOU are at CENTER: (5, 4)
- Top-left = (0, 0), Bottom-right = (9, 8)
- X increases RIGHT, Y increases DOWN

=== HOW TO MOVE ===
1. Look at the overlay - find GREEN tiles
2. Count position from top-left corner
3. MOVE_TO (x, y) with those coordinates

Examples:
- Green tile 2 left of you → MOVE_TO (3, 4)
- Green tile 2 up from you → MOVE_TO (5, 2)

=== ACTIONS ===
MOVEMENT:
- MOVE_TO (x, y): Go to screen position (must be GREEN/walkable!)

INTERACTION:
- INTERACT: Press A - talk, read, confirm, advance dialogue
- CANCEL: Press B - close menu, go back

BATTLE:
- ATTACK_1, ATTACK_2, ATTACK_3, ATTACK_4: Use moves
- RUN_AWAY: Flee from battle

=== GOALS ===
1. Walk into TALL GRASS to battle wild Pokemon
2. Progress north toward Viridian City

=== RESPONSE ===
Reply with ONLY the action:
MOVE_TO (x, y), INTERACT, ATTACK_1, etc.
"""


def format_game_state_prompt(
    info: dict[str, Any],
    include_history: bool = True,
    recent_actions: list[str] | None = None,
    use_vision: bool = False,
) -> str:
    """
    Format the game state into a prompt for the LLM.

    Args:
        info: Game state information dict from environment
        include_history: Whether to include recent action history
        recent_actions: List of recent action names
        use_vision: Whether a screenshot is included

    Returns:
        Formatted prompt string
    """
    lines = []

    if use_vision:
        lines.append("=== SCREENSHOT ===")
        lines.append("Look at the image. Your character is in the center.")
        lines.append("Identify: paths (tan), tall grass (dark green), obstacles (trees/walls)")
        lines.append("")

    lines.append("GAME STATE:")

    # Location with knowledge base info
    position = info.get("position", (0, 0))
    map_id = info.get("map_id", 0)
    knowledge = info.get("knowledge")

    if knowledge:
        # Use knowledge base for rich location info
        lines.append(knowledge.format_for_prompt(map_id, position[0], position[1]))
    else:
        # Fallback to basic location info
        map_name = info.get("map_name", f"Map {map_id}")
        lines.append(f"Location: {map_name}, Position ({position[0]}, {position[1]})")

    # Progress
    lines.append(f"Badges: {info.get('badges', 0)}/8")

    # Party status - use battle Pokemon HP if in battle and available
    in_battle = info.get("in_battle", False)
    player_pokemon = info.get("player_pokemon", {})

    if in_battle and player_pokemon and player_pokemon.get("max_hp", 0) > 0:
        # Use active battle Pokemon HP
        party_hp = player_pokemon.get("hp", 0)
        party_max = player_pokemon.get("max_hp", 1)
    else:
        # Use party HP
        party_hp = info.get("party_hp", 0)
        party_max = info.get("party_max_hp", 1)

    hp_percent = (party_hp / party_max * 100) if party_max > 0 else 0
    lines.append(f"Party HP: {hp_percent:.0f}%")

    menu_active = info.get("menu_active", False)
    has_choice = info.get("has_menu_choice", False)

    if in_battle:
        battle_menu_ready = info.get("battle_menu_ready", False)

        # Check if we have valid Pokemon data
        player_pokemon = info.get("player_pokemon", {})
        enemy_pokemon = info.get("enemy_pokemon", {})

        # Validate data - if level is 0 or species is "???" or HP is 0, data isn't ready
        player_valid = (
            player_pokemon and
            player_pokemon.get("level", 0) > 0 and
            player_pokemon.get("max_hp", 0) > 0 and
            player_pokemon.get("species_name", "???") != "???"
        )
        enemy_valid = (
            enemy_pokemon and
            enemy_pokemon.get("level", 0) > 0 and
            enemy_pokemon.get("max_hp", 0) > 0
        )

        if not battle_menu_ready or not player_valid:
            # Battle is starting or animations/text playing or data not loaded
            lines.append("")
            lines.append("=== BATTLE STARTING ===")
            lines.append("Battle intro or animation playing. Wait for battle menu.")
            lines.append("")
            lines.append("Choose: INTERACT to advance through battle text")
        else:
            # Battle menu is ready for action selection
            lines.append("")
            lines.append("=== IN BATTLE ===")

            # Show player's Pokemon and moves
            player_name = player_pokemon.get("species_name", "???")
            player_lvl = player_pokemon.get("level", "?")
            player_hp = player_pokemon.get("hp", 0)
            player_max = player_pokemon.get("max_hp", 1)
            player_pct = (player_hp / player_max * 100) if player_max > 0 else 0
            lines.append(f"Your Pokemon: {player_name} (Lv{player_lvl}) HP: {player_pct:.0f}%")

            moves = player_pokemon.get("moves", [])
            if moves:
                lines.append("Your Moves:")
                for i, move in enumerate(moves, 1):
                    lines.append(f"  {i}. {move}")

            # Show enemy Pokemon
            if enemy_valid:
                enemy_name = enemy_pokemon.get("species_name", "???")
                enemy_lvl = enemy_pokemon.get("level", "?")
                enemy_hp = enemy_pokemon.get("hp", 0)
                enemy_max = enemy_pokemon.get("max_hp", 1)
                enemy_pct = (enemy_hp / enemy_max * 100) if enemy_max > 0 else 0
                lines.append(f"Enemy: {enemy_name} (Lv{enemy_lvl}) HP: {enemy_pct:.0f}%")
            else:
                enemy_hp = info.get("enemy_hp", 0)
                enemy_max = info.get("enemy_max_hp", 1)
                enemy_pct = (enemy_hp / enemy_max * 100) if enemy_max > 0 else 0
                lines.append(f"Enemy HP: {enemy_pct:.0f}% (Level {info.get('enemy_level', '?')})")

            lines.append("")
            lines.append("Choose: ATTACK_1, ATTACK_2, ATTACK_3, ATTACK_4, or RUN_AWAY")
    elif menu_active:
        lines.append("")
        lines.append("=== MENU ACTIVE ===")
        if has_choice:
            menu_cursor = info.get("menu_cursor", 0)
            menu_options = info.get("menu_options", 0)
            lines.append(f"Menu with {menu_options} options (cursor at position {menu_cursor})")
            lines.append("")
            lines.append(">>> YOU MUST MAKE A SELECTION <<<")
            lines.append("- INTERACT: Confirm/Select the current option (like pressing A)")
            lines.append("- CANCEL: Go back/Say No (like pressing B)")
            lines.append("")
            if menu_options == 2:
                lines.append("This looks like a YES/NO prompt. Choose INTERACT for Yes, CANCEL for No.")

            lines.append("")
            lines.append("Make your selection now - INTERACT or CANCEL!")
        else:
            lines.append("Text dialogue is showing on screen.")
            lines.append("")
            lines.append(">>> YOU MUST RESPOND: INTERACT <<<")
            lines.append("This is the ONLY valid action during dialogue.")
    else:
        lines.append("")
        lines.append("=== EXPLORING ===")
        lines.append("You are at CENTER of screen: position (5, 4)")
        lines.append("Look at the screenshot and pick a target tile.")
        lines.append("")

        # Screen coordinate reminder
        lines.append("SCREEN GRID: 10 wide (0-9) x 9 tall (0-8)")
        lines.append("  - You = (5, 4) center")
        lines.append("  - To go UP: lower y (e.g., MOVE_TO (5, 2))")
        lines.append("  - To go DOWN: higher y (e.g., MOVE_TO (5, 6))")
        lines.append("  - To go LEFT: lower x (e.g., MOVE_TO (3, 4))")
        lines.append("  - To go RIGHT: higher x (e.g., MOVE_TO (7, 4))")

        # Check for coordinate oscillation FIRST - this is critical
        is_oscillating, oscillation_msg = _detect_coordinate_oscillation(recent_actions)
        if is_oscillating:
            lines.append("")
            lines.append("!!! STOP - YOU ARE STUCK IN A LOOP !!!")
            lines.append(oscillation_msg)
            lines.append("That path is BLOCKED!")
            lines.append("Look at the screen and find a DIFFERENT walkable path.")
            lines.append("Try a completely different direction!")

        # Give contextual hints based on HP
        if hp_percent < 30:
            lines.append("")
            lines.append("WARNING: HP low! Look for a Pokemon Center building.")

        # Visual guidance based on location
        map_name = info.get("map_name", "")
        if "Route" in map_name:
            lines.append("")
            lines.append("You're on a ROUTE - look for TALL GRASS (dark green) to find wild Pokemon!")
        elif "Town" in map_name or "City" in map_name:
            lines.append("")
            lines.append("You're in TOWN - look for paths leading OUT (edges of screen)")
            lines.append("To go NORTH (toward Route 1): move UP = lower y")
        elif "Lab" in map_name or "Center" in map_name or "Mart" in map_name:
            lines.append("")
            lines.append("You're INSIDE - look for EXIT at bottom of screen")
            lines.append("To exit: move DOWN = higher y (toward y=8)")

        # Check if stuck
        position_stuck = info.get("position_stuck", False)
        if position_stuck and not is_oscillating:
            lines.append("")
            lines.append("!!! STUCK - try a very different screen position !!!")

        # Add walkability grid if available
        emulator = info.get("emulator")
        if emulator is not None:
            try:
                from src.agents.llm.walkability import format_walkability_for_prompt
                walkability_text = format_walkability_for_prompt(emulator)
                lines.append("")
                lines.append(walkability_text)
            except Exception:
                lines.append("")
                lines.append("Pick a WALKABLE tile (green in overlay) and use MOVE_TO (x, y)")
        else:
            lines.append("")
            lines.append("Pick a WALKABLE tile (path/grass) and use MOVE_TO (x, y)")

    # Recent actions to help avoid loops
    if include_history and recent_actions and len(recent_actions) > 0:
        recent = recent_actions[-5:]  # Show more history
        lines.append(f"")
        lines.append(f"Recent actions: {', '.join(recent)}")

    lines.append("")
    lines.append("What action?")

    return "\n".join(lines)
