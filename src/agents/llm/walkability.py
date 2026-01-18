"""Walkability detection and overlay for Pokemon Red.

Uses PyBoy's built-in game_area_collision() for accurate collision detection
across all tilesets, then creates a visual overlay on the screenshot.

Includes ledge detection - ledges are one-way (can jump DOWN but not UP).
"""

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.emulator.pyboy_wrapper import EmulatorWrapper

from src.emulator.memory_map import GameState

# Memory addresses for Pokemon Red
TILE_MAP_START = 0xC3A0  # wTileMap - 20x18 screen tile buffer

# Screen dimensions
SCREEN_TILES_X = 10  # 10 game tiles wide (each is 16x16 = 2x2 screen tiles)
SCREEN_TILES_Y = 9   # 9 game tiles tall
TILE_SIZE = 16       # Pixels per game tile
SCREEN_TILE_SIZE = 8  # Pixels per screen tile (8x8)

# Ledge tile IDs for Pokemon Red overworld tileset
# These tiles can only be crossed going DOWN (south), not up
# Based on Pokemon Red disassembly - ledge tiles in outdoor tilesets
LEDGE_TILES_DOWN = {
    0x2C,  # Standard ledge (jump down)
    0x2D,  # Ledge variant
    0x2E,  # Ledge variant
    0x2F,  # Ledge variant
    0x37,  # Another ledge type
    0x38,  # Ledge
    0x39,  # Ledge
    0x3A,  # Ledge
}

# Right-facing ledges (can only cross going RIGHT/east)
LEDGE_TILES_RIGHT = {
    0x30,  # Ledge right
    0x31,
}

# Left-facing ledges (can only cross going LEFT/west)
LEDGE_TILES_LEFT = {
    0x32,  # Ledge left
    0x33,
}


def get_player_screen_tile(emulator: "EmulatorWrapper") -> tuple[int, int]:
    """
    Get the player's position on the screen grid.

    Returns (x, y) in game tile coordinates (0-9 for x, 0-8 for y).
    The player is usually at (5, 4) when centered, but can be offset
    near map edges.
    """
    game_state = GameState(emulator)
    return game_state.get_player_screen_position()


def get_ledge_direction(tile_id: int) -> str | None:
    """
    Check if a tile is a ledge and return the allowed direction.

    Args:
        tile_id: The tile ID to check

    Returns:
        "down", "left", "right" if it's a ledge (the direction you CAN go),
        or None if not a ledge
    """
    if tile_id in LEDGE_TILES_DOWN:
        return "down"
    if tile_id in LEDGE_TILES_RIGHT:
        return "right"
    if tile_id in LEDGE_TILES_LEFT:
        return "left"
    return None


def create_walkability_grid(emulator: "EmulatorWrapper") -> list[list[bool]]:
    """
    Create a 10x9 grid showing which tiles are walkable.

    Uses PyBoy's built-in game_area_collision() which properly handles
    all tilesets and ROM bank switching.

    Returns:
        2D list where True = walkable, False = blocked
    """
    # Get collision grid from PyBoy (18x20 screen tiles, 0=blocked, 1=walkable)
    collision = emulator.pyboy.game_area_collision()

    # Convert to 10x9 game tile grid (each game tile = 2x2 screen tiles)
    # Sample top-left screen tile of each 2x2 block
    grid = []
    for y in range(SCREEN_TILES_Y):  # 0-8
        row = []
        for x in range(SCREEN_TILES_X):  # 0-9
            # Map game tile (x, y) to screen tile (x*2, y*2)
            screen_x = x * 2
            screen_y = y * 2
            # Check if walkable (collision value > 0 means walkable)
            is_walkable = collision[screen_y, screen_x] > 0
            row.append(is_walkable)
        grid.append(row)

    return grid


def is_walkability_valid(grid: list[list[bool]]) -> bool:
    """
    Check if a walkability grid appears valid.

    An invalid grid (e.g., during battle transition) will have ALL tiles blocked,
    which is impossible for any real overworld map - there's always somewhere to stand.

    Returns:
        True if grid has at least one walkable tile, False if all blocked
    """
    for row in grid:
        for is_walkable in row:
            if is_walkable:
                return True
    return False


def wait_for_stable_overworld(emulator: "EmulatorWrapper", max_frames: int = 60) -> list[list[bool]] | None:
    """
    Wait for the game to reach a stable overworld state with valid walkability.

    After battles or screen transitions, the collision data can be invalid.
    This function waits until we get a valid walkability grid.

    Args:
        emulator: Emulator instance
        max_frames: Maximum frames to wait (default 60 = 1 second at 60fps)

    Returns:
        Valid walkability grid, or None if timeout reached
    """
    from src.emulator.memory_map import GameState

    game_state = GameState(emulator)

    for _ in range(max_frames):
        # Check if we're in battle - can't read walkability during battle
        if game_state.is_in_battle():
            emulator.tick(1)
            continue

        # Try to read walkability
        grid = create_walkability_grid(emulator)

        # Check if it's valid (has at least one walkable tile)
        if is_walkability_valid(grid):
            return grid

        # Not valid yet - wait a frame and retry
        emulator.tick(1)

    return None  # Timeout


def get_screen_tiles(emulator: "EmulatorWrapper") -> list[list[int]]:
    """
    Read the current screen tile IDs from the tile map buffer.

    Returns a 10x9 grid of tile IDs (one per game tile, sampling top-left of each 2x2).
    Used for ledge detection (which still needs tile IDs).
    """
    tiles = []
    for y in range(SCREEN_TILES_Y):
        row = []
        for x in range(SCREEN_TILES_X):
            # Each game tile is 2x2 screen tiles, sample top-left
            addr = TILE_MAP_START + (y * 2) * 20 + (x * 2)
            tile_id = emulator.read_memory(addr)
            row.append(tile_id)
        tiles.append(row)
    return tiles


def create_ledge_grid(emulator: "EmulatorWrapper") -> list[list[str | None]]:
    """
    Create a 10x9 grid showing ledge directions.

    NOTE: Ledge detection is currently disabled because the hardcoded tile IDs
    don't match actual ledges in all tilesets. Returns all None (no ledges).

    Returns:
        2D list where each cell is None (no ledge detection active)
    """
    # Disabled - the tile IDs don't match actual ledges and cause path blocking
    # TODO: Find correct ledge tile IDs or use a different detection method
    return [[None] * SCREEN_TILES_X for _ in range(SCREEN_TILES_Y)]


def can_move_to(
    from_x: int, from_y: int,
    to_x: int, to_y: int,
    walkability: list[list[bool]],
    ledges: list[list[str | None]]
) -> bool:
    """
    Check if movement from one tile to another is allowed.

    Considers both walkability and ledge direction constraints.

    Args:
        from_x, from_y: Source position
        to_x, to_y: Target position
        walkability: Walkability grid
        ledges: Ledge direction grid

    Returns:
        True if movement is allowed
    """
    # Check bounds
    if not (0 <= to_x < 10 and 0 <= to_y < 9):
        return False

    # Check if target is walkable
    if not walkability[to_y][to_x]:
        return False

    # Calculate movement direction
    dx = to_x - from_x
    dy = to_y - from_y

    # Check if target tile is a ledge
    ledge_dir = ledges[to_y][to_x]
    if ledge_dir is not None:
        # Ledge tiles can only be entered from specific directions
        if ledge_dir == "down":
            # Can only step onto this from above (moving down/south)
            if dy <= 0:  # Not moving down
                return False
        elif ledge_dir == "right":
            # Can only step onto this from the left (moving right/east)
            if dx <= 0:  # Not moving right
                return False
        elif ledge_dir == "left":
            # Can only step onto this from the right (moving left/west)
            if dx >= 0:  # Not moving left
                return False

    # Also check if SOURCE tile is a ledge - you can jump OFF ledges in the allowed direction
    # but the target check above handles the main case

    return True


def create_walkability_overlay(
    screen: NDArray[np.uint8],
    emulator: "EmulatorWrapper",
    show_coords: bool = True,
) -> NDArray[np.uint8]:
    """
    Create a screenshot with walkability overlay.

    - Green tint on walkable tiles
    - Red tint on blocked tiles
    - Grid lines and coordinate labels
    - Player position highlighted (detected from sprite layer)

    Args:
        screen: Original screen image (144, 160, 3)
        emulator: Emulator instance for reading memory
        show_coords: Whether to show coordinate labels

    Returns:
        Screen with overlay (144, 160, 3)
    """
    walkability = create_walkability_grid(emulator)

    # If walkability is invalid (transitioning), return original screen
    if not is_walkability_valid(walkability):
        return screen

    # Get actual player screen position (not hardcoded)
    player_x, player_y = get_player_screen_tile(emulator)

    # Convert to PIL for drawing
    img = Image.fromarray(screen).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw walkability overlay on each tile
    for y in range(SCREEN_TILES_Y):
        for x in range(SCREEN_TILES_X):
            px = x * TILE_SIZE
            py = y * TILE_SIZE

            is_walkable = walkability[y][x]

            # Check if this is the player position (detected from sprite)
            is_player = (x == player_x and y == player_y)

            if is_player:
                # Player position - blue highlight
                color = (0, 100, 255, 100)
            elif is_walkable:
                # Walkable - green tint
                color = (0, 255, 0, 60)
            else:
                # Blocked - red tint
                color = (255, 0, 0, 80)

            draw.rectangle(
                [px, py, px + TILE_SIZE - 1, py + TILE_SIZE - 1],
                fill=color,
            )

    # Draw grid lines
    grid_color = (255, 255, 255, 120)
    for x in range(0, 160, TILE_SIZE):
        draw.line([(x, 0), (x, 144)], fill=grid_color, width=1)
    for y in range(0, 144, TILE_SIZE):
        draw.line([(0, y), (160, y)], fill=grid_color, width=1)

    # Add coordinate labels if requested
    if show_coords:
        label_color = (255, 255, 255, 255)
        # Corner labels
        try:
            draw.text((2, 2), "0,0", fill=label_color)
            draw.text((145, 2), "9,0", fill=label_color)
            draw.text((2, 134), "0,8", fill=label_color)
            draw.text((145, 134), "9,8", fill=label_color)
            # Player label at detected position
            draw.text((player_x * TILE_SIZE + 2, player_y * TILE_SIZE + 2), "YOU", fill=(0, 150, 255, 255))
        except Exception:
            pass  # Font issues, skip labels

    # Composite overlay onto original
    result = Image.alpha_composite(img, overlay)
    return np.array(result.convert("RGB"))


def find_reachable_tiles(
    walkability: list[list[bool]],
    start_x: int | None = None,
    start_y: int | None = None,
    ledges: list[list[str | None]] | None = None,
    emulator: "EmulatorWrapper | None" = None
) -> set[tuple[int, int]]:
    """
    Find all tiles reachable from the player's position using flood fill.

    Accounts for ledge direction constraints if ledge grid is provided.

    Args:
        walkability: 10x9 grid of walkable tiles
        start_x: Player's x position (if None, detected from emulator or defaults to 5)
        start_y: Player's y position (if None, detected from emulator or defaults to 4)
        ledges: Optional ledge direction grid
        emulator: Optional emulator to detect player position from

    Returns:
        Set of (x, y) tuples that are reachable from start position
    """
    # Detect player position if not provided
    if start_x is None or start_y is None:
        if emulator is not None:
            start_x, start_y = get_player_screen_tile(emulator)
        else:
            # Fallback to center
            start_x = start_x if start_x is not None else 5
            start_y = start_y if start_y is not None else 4
    reachable = set()
    # Store (x, y) positions to visit along with how we got there
    to_visit = [(start_x, start_y, start_x, start_y)]  # (x, y, from_x, from_y)

    while to_visit:
        x, y, from_x, from_y = to_visit.pop()
        if (x, y) in reachable:
            continue
        if x < 0 or x >= 10 or y < 0 or y >= 9:
            continue

        # Check if move is valid
        if ledges is not None and (x, y) != (start_x, start_y):
            if not can_move_to(from_x, from_y, x, y, walkability, ledges):
                continue
        elif not walkability[y][x]:
            continue

        reachable.add((x, y))

        # Add neighbors (with current position as "from")
        to_visit.append((x + 1, y, x, y))
        to_visit.append((x - 1, y, x, y))
        to_visit.append((x, y + 1, x, y))
        to_visit.append((x, y - 1, x, y))

    return reachable


def find_path(
    walkability: list[list[bool]],
    start_x: int,
    start_y: int,
    target_x: int,
    target_y: int,
    ledges: list[list[str | None]] | None = None
) -> list[tuple[int, int]] | None:
    """
    Find a path from start to target using BFS.

    Accounts for ledge direction constraints if ledge grid is provided.

    Args:
        walkability: 10x9 grid of walkable tiles
        start_x, start_y: Starting position
        target_x, target_y: Target position
        ledges: Optional ledge direction grid (from create_ledge_grid)

    Returns:
        List of (x, y) positions from start to target (excluding start),
        or None if no path exists.
    """
    from collections import deque

    if not (0 <= target_x < 10 and 0 <= target_y < 9):
        return None
    if not walkability[target_y][target_x]:
        return None

    # BFS
    queue = deque([(start_x, start_y, [])])
    visited = {(start_x, start_y)}

    while queue:
        x, y, path = queue.popleft()

        if x == target_x and y == target_y:
            return path

        # Try all 4 directions
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy

            if (nx, ny) in visited:
                continue

            # Check if move is valid (walkability + ledge constraints)
            if ledges is not None:
                if not can_move_to(x, y, nx, ny, walkability, ledges):
                    continue
            else:
                # No ledge info, just check walkability
                if not (0 <= nx < 10 and 0 <= ny < 9):
                    continue
                if not walkability[ny][nx]:
                    continue

            visited.add((nx, ny))
            queue.append((nx, ny, path + [(nx, ny)]))

    return None  # No path found


def find_edge_exits(
    walkability: list[list[bool]],
    ledges: list[list[str | None]] | None = None,
    player_x: int = 5,
    player_y: int = 4
) -> list[tuple[int, int, str]]:
    """
    Find REACHABLE walkable tiles on the edges of the screen (potential exits).

    Only returns exits that the player can actually walk to from their position.
    Accounts for ledge constraints if provided.

    Args:
        walkability: 10x9 grid of walkable tiles
        ledges: Optional ledge direction grid
        player_x: Player's x position on screen (default 5)
        player_y: Player's y position on screen (default 4)

    Returns:
        List of (x, y, direction) tuples for reachable edge tiles
    """
    # First find all tiles reachable from player position
    reachable = find_reachable_tiles(walkability, start_x=player_x, start_y=player_y, ledges=ledges)

    exits = []

    # Top edge (y=0) - exit north
    for x in range(10):
        if (x, 0) in reachable:
            exits.append((x, 0, "north"))

    # Bottom edge (y=8) - exit south
    for x in range(10):
        if (x, 8) in reachable:
            exits.append((x, 8, "south"))

    # Left edge (x=0) - exit west
    for y in range(9):
        if (0, y) in reachable:
            exits.append((0, y, "west"))

    # Right edge (x=9) - exit east
    for y in range(9):
        if (9, y) in reachable:
            exits.append((9, y, "east"))

    return exits


def format_walkability_for_prompt(emulator: "EmulatorWrapper", compact: bool = False) -> str:
    """
    Format walkability information as text for the prompt.

    Returns a text representation showing which screen positions are walkable.
    The player position is detected from the sprite layer, not hardcoded.

    Args:
        emulator: Emulator instance
        compact: If True, use minimal formatting to save tokens
    """
    walkability = create_walkability_grid(emulator)

    # Check if walkability is valid (not in transition)
    if not is_walkability_valid(walkability):
        return ">>> WALKABILITY GRID <<<\n[Screen transitioning - walkability not available]"

    # Get actual player screen position (not hardcoded)
    player_x, player_y = get_player_screen_tile(emulator)

    lines = []
    lines.append(">>> WALKABILITY GRID <<<")
    lines.append("  0 1 2 3 4 5 6 7 8 9")

    for y, row in enumerate(walkability):
        row_str = f"{y} "
        for x, is_walkable in enumerate(row):
            if x == player_x and y == player_y:
                row_str += "@ "  # Player position (detected from sprite)
            elif is_walkable:
                row_str += "W "
            else:
                row_str += "X "
        lines.append(row_str)

    if not compact:
        lines.append(f"W=walkable, X=blocked, @=you (at {player_x},{player_y})")

    return "\n".join(lines)
