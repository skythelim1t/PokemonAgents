"""Tool definitions for LLM agent.

Defines the tools Claude can call to interact with the game.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

from src.agents.llm.menu_navigator import MenuNavigator
from src.agents.llm.battle_navigator import BattleMenuNavigator

# Tool definitions in Anthropic's format
TOOL_DEFINITIONS = [
    {
        "name": "navigator",
        "description": """Navigate to a screen position. The screen is a 10x9 grid where:
- Your position is marked @ in the walkability grid (usually center, but varies near map edges)
- Top-left is (0, 0), bottom-right is (9, 8)
- X increases right, Y increases down

The walkability grid in the prompt shows which tiles are walkable (W) vs blocked (X).
Only navigate to walkable (W) tiles!

Returns: New screenshot with walkability overlay, updated game state, and walkability grid.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Target X coordinate (0-9, left to right)",
                    "minimum": 0,
                    "maximum": 9
                },
                "y": {
                    "type": "integer",
                    "description": "Target Y coordinate (0-8, top to bottom)",
                    "minimum": 0,
                    "maximum": 8
                }
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "press_button",
        "description": """Press a game button. Use this for:
- 'a': Interact, confirm, advance dialogue
- 'b': Cancel, go back
- 'start': Open menu

In battle, use specific attack buttons instead of navigating menus manually.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "enum": ["a", "b", "start", "select", "up", "down", "left", "right"],
                    "description": "The button to press"
                }
            },
            "required": ["button"]
        }
    },
    {
        "name": "attack",
        "description": """Use an attack move in battle. Only works during battle when it's your turn.

Returns: Result of the attack, updated battle state.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "move_slot": {
                    "type": "integer",
                    "description": "Which move to use (1-4)",
                    "minimum": 1,
                    "maximum": 4
                }
            },
            "required": ["move_slot"]
        }
    },
    {
        "name": "run_away",
        "description": """Attempt to flee from a wild Pokemon battle. Only works in wild battles, not trainer battles.

Returns: Whether escape was successful.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "update_knowledge",
        "description": """Update your persistent knowledge base. Organize information for future reference.

Categories and actions:
- "add_note": General observation {"action": "add_note", "content": "Route 1 has Rattata and Pidgey", "vital": false, "note_category": "navigation"}
- "set_goal": Current objective {"action": "set_goal", "content": "Reach Pewter City and challenge Brock"}
- "add_question": Something to investigate {"action": "add_question", "content": "Where is the Moon Stone?"}
- "resolve_question": Answer found {"action": "resolve_question", "content": "Where is the Moon Stone?"}
- "add_strategy": Battle/area strategy {"action": "add_strategy", "category": "battle", "key": "brock", "content": "Use Water Gun, avoid Tackle"}
- "learn_matchup": Type advantage {"action": "learn_matchup", "enemy_type": "rock", "our_pokemon": "Squirtle"}
- "record_failure": Learn from mistake {"action": "record_failure", "context": "Fought Brock at level 8", "outcome": "wiped", "lesson": "Need level 12+"}
- "set_flag": Story progress {"action": "set_flag", "flag": "beat_brock", "value": true}

Note categories: general, battle, navigation, item, npc
Mark notes as vital=true for important permanent info (e.g., "Brock uses rock types").""",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_note", "remove_note", "set_goal", "add_question", "resolve_question",
                             "add_strategy", "learn_matchup", "record_failure", "set_flag"],
                    "description": "The type of knowledge update"
                },
                "content": {
                    "type": "string",
                    "description": "The main content (note text, goal, question, strategy)"
                },
                "vital": {
                    "type": "boolean",
                    "description": "For add_note: mark as vital (never auto-pruned). Use for critical permanent info."
                },
                "note_category": {
                    "type": "string",
                    "enum": ["general", "battle", "navigation", "item", "npc"],
                    "description": "For add_note: category of the note"
                },
                "category": {
                    "type": "string",
                    "enum": ["battle", "area", "grinding"],
                    "description": "For add_strategy: type of strategy"
                },
                "key": {
                    "type": "string",
                    "description": "For add_strategy: opponent name or area name"
                },
                "enemy_type": {
                    "type": "string",
                    "description": "For learn_matchup: the enemy type (rock, water, etc.)"
                },
                "our_pokemon": {
                    "type": "string",
                    "description": "For learn_matchup: our Pokemon that works well"
                },
                "context": {
                    "type": "string",
                    "description": "For record_failure: what was happening"
                },
                "outcome": {
                    "type": "string",
                    "description": "For record_failure: what went wrong"
                },
                "lesson": {
                    "type": "string",
                    "description": "For record_failure: what to do differently"
                },
                "flag": {
                    "type": "string",
                    "description": "For set_flag: the story flag name"
                },
                "value": {
                    "type": "boolean",
                    "description": "For set_flag: true/false"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "query_knowledge",
        "description": """Search knowledge base AND auto-take an action. Returns knowledge then moves/attacks for you.

Use to recall past notes, strategies, failures. After showing results, this tool automatically:
- In battle: uses attack(1)
- In town: uses find_route() to leave
- On route: uses navigator() to explore
- In building: uses find_exit() to leave""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (e.g., 'brock', 'route 1', 'water type')"
                }
            },
            "required": []
        }
    },
    {
        "name": "think",
        "description": """Reason through a complex problem. ONLY use when truly stuck or facing a major decision.

DO NOT use think() for routine navigation or simple actions.
DO NOT use think() every turn - it wastes time.

Only use when:
- You've tried multiple approaches and failed
- Facing a gym leader or major battle
- Genuinely unsure what to do next

For normal gameplay, just use navigator() and attack() directly.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "situation": {
                    "type": "string",
                    "description": "Brief description of current situation"
                },
                "analysis": {
                    "type": "string",
                    "description": "Your analysis of what's happening and why"
                },
                "options": {
                    "type": "string",
                    "description": "Possible actions you could take"
                },
                "plan": {
                    "type": "string",
                    "description": "What you've decided to do and why"
                }
            },
            "required": ["plan"]
        }
    },
    {
        "name": "wait",
        "description": """Wait until a condition is met. Use this for:
- Waiting for dialogue/text to finish before pressing A
- Waiting for battle animations to complete
- Waiting until you can select a move (battle_menu_ready)
- Fixed frame delays when needed

IMPORTANT: Do NOT use wait(battle_starts) to find wild Pokemon! Wild Pokemon only appear when you WALK through tall grass using navigator(). Use wait(battle_menu_ready) when already in battle.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "until": {
                    "type": "string",
                    "enum": ["dialogue_done", "animation_done", "battle_menu_ready", "battle_ends", "frames"],
                    "description": "Condition to wait for"
                },
                "frames": {
                    "type": "integer",
                    "description": "Number of frames to wait (only used when until='frames')",
                    "minimum": 1,
                    "maximum": 300
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max frames to wait before giving up (default: 120)",
                    "minimum": 1,
                    "maximum": 300
                }
            },
            "required": ["until"]
        }
    },
    {
        "name": "find_exit",
        "description": """Find and GO TO the exit/door in the current building. AUTO-NAVIGATES for you!

Use this when inside buildings, Pokemon Centers, or any indoor location. This tool will:
1. Find the nearest exit/door
2. Automatically walk there to exit the building""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "find_route",
        "description": """Leave town and go to a route with wild Pokemon. AUTO-NAVIGATES for you!

Use this when in a TOWN. This tool will:
1. Find the nearest reachable edge of the screen
2. Automatically walk there to exit the town

Towns have no wild Pokemon - use this to leave and find routes with tall grass!""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "heal_at_pokecenter",
        "description": """Heal your Pokemon at a Pokemon Center. You must be inside a Pokemon Center to use this.

This tool:
1. Walks to the counter
2. Talks to the nurse
3. Confirms healing
4. Waits for healing animation
5. Returns to normal state

Returns: Success/failure and updated party HP.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "use_item",
        "description": """Use an item from your bag on a Pokemon.

Opens the start menu, navigates to Items, finds the item, and uses it.
For healing items (Potion, Super Potion, etc.), specify which Pokemon to heal.

Common items: Potion, Super Potion, Antidote, Paralyze Heal, Awakening, Pokeball""",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "Name of the item to use (e.g., 'Potion', 'Antidote')"
                },
                "pokemon_slot": {
                    "type": "integer",
                    "description": "Which Pokemon to use it on (1-6). Required for healing/status items.",
                    "minimum": 1,
                    "maximum": 6
                }
            },
            "required": ["item_name"]
        }
    },
    {
        "name": "check_pokemon",
        "description": """Open the Pokemon menu and get detailed party status.

Returns information about all Pokemon in your party including:
- Species, level, HP
- Status conditions
- Move list

Use this to check your team's status before making decisions.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "save_game",
        "description": """Save the game. Opens start menu, selects SAVE, confirms, and closes menu.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "switch_pokemon",
        "description": """Switch to a different Pokemon.

In battle: Switches your active Pokemon (uses your turn).
Outside battle: Opens Pokemon menu and moves selected Pokemon to front.

Use this for type advantage or to save a low HP Pokemon.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "pokemon_slot": {
                    "type": "integer",
                    "description": "Which Pokemon to switch to (1-6, where 1 is current lead)",
                    "minimum": 1,
                    "maximum": 6
                }
            },
            "required": ["pokemon_slot"]
        }
    },
    {
        "name": "set_goal",
        "description": """Set a new main goal to pursue. Use templates for common objectives.

Available templates:
- get_starter: Get your first Pokemon from Oak's Lab
- deliver_parcel: Deliver Oak's Parcel and get the Pokedex
- beat_brock: Beat Brock and earn the Boulder Badge
- beat_misty: Beat Misty and earn the Cascade Badge
- beat_surge: Beat Lt. Surge and earn the Thunder Badge
- train_pokemon: Train Pokemon by battling wild Pokemon

Templates include predefined subgoals that help you track progress step by step.
Goals auto-complete when you achieve the objective (e.g., getting a badge).""",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal description (used if no template)"
                },
                "template": {
                    "type": "string",
                    "description": "Optional template name: get_starter, deliver_parcel, beat_brock, beat_misty, beat_surge, train_pokemon"
                }
            },
            "required": []
        }
    },
    {
        "name": "add_subgoal",
        "description": """Add a subgoal to the current active goal.

Use this to break down your goal into smaller steps when the template subgoals
aren't sufficient or when you have a custom goal.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "subgoal": {
                    "type": "string",
                    "description": "The subgoal description (e.g., 'Find the hidden switch in the gym')"
                }
            },
            "required": ["subgoal"]
        }
    },
    {
        "name": "complete_subgoal",
        "description": """Mark the current subgoal as completed and advance to the next one.

Use this when you've achieved a subgoal. The system will automatically:
- Mark the subgoal as done
- Advance to the next pending subgoal
- Update the goal's progress percentage
- Complete the main goal if all subgoals are done""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "check_progress",
        "description": """Check progress on the current goal and its subgoals.

Returns:
- Current main goal and status
- Current subgoal you're working on
- List of completed subgoals
- List of pending subgoals
- Overall progress percentage""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    message: str
    screenshot: NDArray[np.uint8] | None = None  # New screenshot if available
    game_state: dict[str, Any] | None = None  # Updated game state
    walkability_text: str | None = None  # Walkability grid text


class ToolExecutor:
    """Executes tool calls and returns results."""

    def __init__(self, env, knowledge_base):
        """
        Initialize the tool executor.

        Args:
            env: The Pokemon environment
            knowledge_base: The agent's knowledge base
        """
        self.env = env
        self.knowledge = knowledge_base
        self.menu = MenuNavigator(env)  # Menu navigator for verified menu actions
        self.battle = BattleMenuNavigator(env)  # Battle menu navigator
        self._steps_per_navigation = 30  # Max steps to reach target
        self._last_dir: int | None = None  # Track last direction for stuck detection
        self._blocked_attempts = 0  # Track consecutive blocked tile attempts
        self._max_blocked_attempts = 2  # Force analysis after this many blocked attempts
        self._last_query_turn = -10  # Rate limit query_knowledge (allow every 10 turns)
        self._current_turn = 0  # Current turn counter (set by ToolAgent)
        self._recent_targets: list[tuple[int, int]] = []  # Last 2 nav targets to prevent back-and-forth

    def set_render_callback(self, callback: callable) -> None:
        """Set a callback to render the screen during battle waits."""
        self.battle.set_render_callback(callback)

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            ToolResult with outcome
        """
        if tool_name == "navigator":
            return self._execute_navigator(tool_input)
        elif tool_name == "press_button":
            return self._execute_button(tool_input)
        elif tool_name == "attack":
            return self._execute_attack(tool_input)
        elif tool_name == "run_away":
            return self._execute_run()
        elif tool_name == "update_knowledge":
            return self._execute_knowledge(tool_input)
        elif tool_name == "query_knowledge":
            return self._execute_query_knowledge(tool_input)
        elif tool_name == "think":
            return self._execute_think(tool_input)
        elif tool_name == "wait":
            return self._execute_wait(tool_input)
        elif tool_name == "find_exit":
            return self._execute_find_exit()
        elif tool_name == "find_route":
            return self._execute_find_route()
        elif tool_name == "heal_at_pokecenter":
            return self._execute_heal_at_pokecenter()
        elif tool_name == "use_item":
            return self._execute_use_item(tool_input)
        elif tool_name == "check_pokemon":
            return self._execute_check_pokemon()
        elif tool_name == "save_game":
            return self._execute_save_game()
        elif tool_name == "switch_pokemon":
            return self._execute_switch_pokemon(tool_input)
        elif tool_name == "set_goal":
            return self._execute_set_goal(tool_input)
        elif tool_name == "add_subgoal":
            return self._execute_add_subgoal(tool_input)
        elif tool_name == "complete_subgoal":
            return self._execute_complete_subgoal()
        elif tool_name == "check_progress":
            return self._execute_check_progress()
        else:
            return ToolResult(
                success=False,
                message=f"Unknown tool: {tool_name}"
            )

    def _execute_navigator(self, params: dict) -> ToolResult:
        """
        Navigate to screen coordinates using BFS pathfinding.

        Architecture:
        1. Convert screen target to world coords ONCE
        2. Loop: move toward world target until reached
        3. Each step: derive current screen coords from world positions
        """
        import logging
        logger = logging.getLogger(__name__)

        from src.agents.llm.walkability import (
            create_walkability_overlay,
            format_walkability_for_prompt,
            create_walkability_grid,
            find_path,
            get_player_screen_tile,
            is_walkability_valid,
            wait_for_stable_overworld,
        )

        # Check if menu is open - can't navigate with menu open
        info = self.env._get_info()
        if info.get("menu_active") and not info.get("in_battle"):
            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=False,
                message="Menu is open! Press B to close the menu before navigating.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        screen_x = params.get("x", 5)
        screen_y = params.get("y", 4)

        # Validate coordinates
        screen_x = max(0, min(screen_x, 9))
        screen_y = max(0, min(screen_y, 8))

        # Prevent back-and-forth: reject if target is in last 2 positions
        target = (screen_x, screen_y)
        if target in self._recent_targets:
            from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt
            obs = self.env.emulator.get_screen()
            overlay = create_walkability_overlay(obs, self.env.emulator)
            return ToolResult(
                success=False,
                message=f"Already visited ({screen_x},{screen_y}) recently! Pick a NEW target to make progress.",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=format_walkability_for_prompt(self.env.emulator)
            )

        emulator = self.env.emulator

        # === STEP 1: Get player screen position and convert to world coordinates ===
        info = self.env._get_info()
        player_world_x, player_world_y = info["position"]
        start_map_id = info.get("map_id", 0)

        # Get player's actual screen position (not hardcoded to center)
        player_screen_x, player_screen_y = get_player_screen_tile(emulator)

        # target_world = player_world + (target_screen - player_screen)
        target_world_x = player_world_x + (screen_x - player_screen_x)
        target_world_y = player_world_y + (screen_y - player_screen_y)

        # Wait for stable overworld state (handles post-battle transitions)
        walkability = wait_for_stable_overworld(emulator, max_frames=60)
        if walkability is None:
            # Timeout - still in transition, ask agent to wait
            obs = emulator.get_screen()
            return ToolResult(
                success=False,
                message="Game is transitioning (post-battle or screen change). Press A or wait a moment, then try again.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # DEBUG: Print walkability grid
        logger.info(f"NAV: target=({screen_x},{screen_y}) player_screen=({player_screen_x},{player_screen_y}) player_world=({player_world_x},{player_world_y})")
        grid_str = "WALKABILITY:\n  0 1 2 3 4 5 6 7 8 9\n"
        for y, row in enumerate(walkability):
            grid_str += f"{y} "
            for x, w in enumerate(row):
                if x == player_screen_x and y == player_screen_y:
                    grid_str += "@ "
                elif x == screen_x and y == screen_y:
                    # Show if target is walkable (T) or blocked (!)
                    grid_str += "T " if w else "! "
                elif w:
                    grid_str += "W "
                else:
                    grid_str += "X "
            grid_str += "\n"
        logger.info(grid_str)
        if not walkability[screen_y][screen_x]:
            self._blocked_attempts += 1
            obs = emulator.get_screen()
            overlay = create_walkability_overlay(obs, emulator)
            msg = f"({screen_x},{screen_y}) is BLOCKED (wall). Pick a W tile."
            if self._blocked_attempts >= self._max_blocked_attempts:
                self._blocked_attempts = 0
                msg = f"STOP! {msg}"
            return ToolResult(
                success=False,
                message=msg,
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=format_walkability_for_prompt(emulator)
            )

        # Check if target is REACHABLE (path exists from player to target)
        initial_path = find_path(walkability, player_screen_x, player_screen_y, screen_x, screen_y)
        if initial_path is None:
            self._blocked_attempts += 1
            obs = emulator.get_screen()
            overlay = create_walkability_overlay(obs, emulator)
            msg = f"({screen_x},{screen_y}) is UNREACHABLE (no path). Pick a tile you can walk to."
            if self._blocked_attempts >= self._max_blocked_attempts:
                self._blocked_attempts = 0
                msg = f"STOP! {msg}"
            return ToolResult(
                success=False,
                message=msg,
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=format_walkability_for_prompt(emulator)
            )

        # Reset blocked counter since target is reachable
        self._blocked_attempts = 0

        # === STEP 2: Movement loop ===
        last_pos = None
        last_distance = abs(target_world_x - player_world_x) + abs(target_world_y - player_world_y)
        stuck_count = 0
        button_names = {4: "up", 5: "down", 6: "left", 7: "right"}
        # Track blocked directions PER POSITION to avoid oscillation
        blocked_at_pos: dict[tuple[int, int], set[int]] = {}
        # Track visited positions to avoid going back
        visited_positions: set[tuple[int, int]] = {(player_world_x, player_world_y)}
        last_button_tried: int | None = None  # Track direction from previous step

        for step in range(self._steps_per_navigation):
            # Read current position
            curr_info = self.env._get_info()
            player_x, player_y = curr_info["position"]
            current_distance = abs(target_world_x - player_x) + abs(target_world_y - player_y)

            # Check if reached target
            if player_x == target_world_x and player_y == target_world_y:
                # Track this target to prevent back-and-forth
                self._recent_targets.append(target)
                if len(self._recent_targets) > 2:
                    self._recent_targets.pop(0)  # Keep only last 2

                obs = emulator.get_screen()
                overlay = create_walkability_overlay(obs, emulator)
                return ToolResult(
                    success=True,
                    message=f"Reached ({screen_x},{screen_y}). Keep exploring W tiles.",
                    screenshot=overlay,
                    game_state=self._get_game_state(),
                    walkability_text=format_walkability_for_prompt(emulator)
                )

            # Check for battle
            if curr_info.get("in_battle", False):
                # Wait for battle intro to finish and menu to be ready
                # This prevents LLM from trying to attack before menu is available
                self.battle.wait_for_battle_menu(timeout=300)

                # Get updated info with Pokemon details
                info = self.env._get_info()
                player = info.get("player_pokemon", {})
                enemy = info.get("enemy_pokemon", {})

                # Build battle intro message
                lines = ["Wild Pokemon appeared!"]
                lines.append("")
                lines.append("=== BATTLE START ===")
                if player:
                    p_hp = player.get("hp", 0)
                    p_max = player.get("max_hp", 1)
                    p_pct = int(p_hp / max(p_max, 1) * 100)
                    moves = player.get("moves", [])
                    lines.append(f"Your {player.get('species_name', '???')} Lv{player.get('level', '?')}: {p_hp}/{p_max} HP ({p_pct}%)")
                    if moves:
                        lines.append(f"Moves: {', '.join(moves)}")
                if enemy:
                    e_hp = enemy.get("hp", 0)
                    e_max = enemy.get("max_hp", 1)
                    e_pct = int(e_hp / max(e_max, 1) * 100)
                    lines.append(f"Enemy {enemy.get('species_name', '???')} Lv{enemy.get('level', '?')}: {e_hp}/{e_max} HP ({e_pct}%)")
                lines.append("")
                lines.append("Battle menu ready - use attack(move_slot) to fight!")

                obs = emulator.get_screen()
                return ToolResult(
                    success=True,
                    message="\n".join(lines),
                    screenshot=obs,
                    game_state=self._get_game_state(),
                    walkability_text=None
                )

            # Check for map transition (coordinates become invalid on new map)
            curr_map_id = curr_info.get("map_id", 0)
            if curr_map_id != start_map_id:
                # Clear recent targets - new area means fresh start
                self._recent_targets.clear()

                obs = emulator.get_screen()
                overlay = create_walkability_overlay(obs, emulator)
                new_location = self.knowledge.get_map_name(curr_map_id)
                return ToolResult(
                    success=True,
                    message=f"Entered new area: {new_location}. Pick a new target from this grid.",
                    screenshot=overlay,
                    game_state=self._get_game_state(),
                    walkability_text=format_walkability_for_prompt(emulator)
                )

            # Detect if stuck (player didn't move)
            pos_key = (player_x, player_y)
            if pos_key == last_pos:
                stuck_count += 1
                # The direction we tried last step didn't work - add it to blocked for THIS position
                if last_button_tried is not None:
                    if pos_key not in blocked_at_pos:
                        blocked_at_pos[pos_key] = set()
                    blocked_at_pos[pos_key].add(last_button_tried)
                # If we've tried all 4 directions at THIS position, give up
                if pos_key in blocked_at_pos and len(blocked_at_pos[pos_key]) >= 4:
                    obs = emulator.get_screen()
                    # Check if menu/dialogue is the cause
                    stuck_info = self.env._get_info()
                    if stuck_info.get("menu_active"):
                        return ToolResult(
                            success=False,
                            message="Can't move - menu or dialogue is open! Press B to close it, then try again.",
                            screenshot=obs,
                            game_state=self._get_game_state()
                        )
                    overlay = create_walkability_overlay(obs, emulator)
                    return ToolResult(
                        success=False,
                        message=f"Stuck trying to reach ({screen_x},{screen_y}). All directions blocked at this position. Try different tile.",
                        screenshot=overlay,
                        game_state=self._get_game_state(),
                        walkability_text=format_walkability_for_prompt(emulator)
                    )
            else:
                # Player moved to new position - reset stuck count but keep blocked_at_pos memory
                stuck_count = 0
                last_distance = current_distance
                visited_positions.add(pos_key)  # Track visited positions
            last_pos = pos_key

            # === Calculate target's current screen position ===
            # Get current player screen position (can change near edges)
            curr_player_screen_x, curr_player_screen_y = get_player_screen_tile(emulator)

            # target_screen = (target_world - player_world) + player_screen
            target_screen_x = (target_world_x - player_x) + curr_player_screen_x
            target_screen_y = (target_world_y - player_y) + curr_player_screen_y

            # Get current walkability (with validation)
            current_walkability = create_walkability_grid(emulator)
            if not is_walkability_valid(current_walkability):
                # Invalid walkability during navigation - likely screen transition
                obs = emulator.get_screen()
                return ToolResult(
                    success=False,
                    message="Screen changed during navigation. Try again.",
                    screenshot=obs,
                    game_state=self._get_game_state()
                )

            # Determine movement direction
            path = None
            if 0 <= target_screen_x < 10 and 0 <= target_screen_y < 9:
                # Target is on screen - use BFS pathfinding from current player screen position
                path = find_path(current_walkability, curr_player_screen_x, curr_player_screen_y, target_screen_x, target_screen_y)

                if path and len(path) > 0:
                    # Take first step of path
                    next_x, next_y = path[0]
                    dx, dy = next_x - curr_player_screen_x, next_y - curr_player_screen_y
                else:
                    # No path found - move directly toward target
                    dx = target_screen_x - curr_player_screen_x
                    dy = target_screen_y - curr_player_screen_y
            else:
                # Target is off screen - move in its direction
                dx = target_world_x - player_x
                dy = target_world_y - player_y

            # Convert direction to button press
            # Prioritize larger axis
            if abs(dy) >= abs(dx):
                if dy < 0:
                    button = 4  # up
                elif dy > 0:
                    button = 5  # down
                elif dx < 0:
                    button = 6  # left
                else:
                    button = 7  # right
            else:
                if dx < 0:
                    button = 6  # left
                elif dx > 0:
                    button = 7  # right
                elif dy < 0:
                    button = 4  # up
                else:
                    button = 5  # down

            # Get blocked directions for current position
            blocked_here = blocked_at_pos.get(pos_key, set())

            # Also block directions that would take us to visited positions (avoid oscillation)
            direction_to_delta = {4: (0, -1), 5: (0, 1), 6: (-1, 0), 7: (1, 0)}
            for dir_btn, (ddx, ddy) in direction_to_delta.items():
                neighbor = (player_x + ddx, player_y + ddy)
                if neighbor in visited_positions and neighbor != (target_world_x, target_world_y):
                    blocked_here = blocked_here | {dir_btn}  # Don't modify original set

            # If this direction already failed at this position, try an alternate that makes progress
            if button in blocked_here:
                # Calculate which directions make progress toward target
                dx_to_target = target_world_x - player_x
                dy_to_target = target_world_y - player_y

                # Rank directions by how much progress they make
                # (direction_button, progress_score) - higher is better
                direction_scores = []
                if 4 not in blocked_here:  # up (dy--)
                    direction_scores.append((4, -dy_to_target))  # positive if target is above
                if 5 not in blocked_here:  # down (dy++)
                    direction_scores.append((5, dy_to_target))   # positive if target is below
                if 6 not in blocked_here:  # left (dx--)
                    direction_scores.append((6, -dx_to_target))  # positive if target is left
                if 7 not in blocked_here:  # right (dx++)
                    direction_scores.append((7, dx_to_target))   # positive if target is right

                if direction_scores:
                    # Sort by progress score descending, pick best
                    direction_scores.sort(key=lambda x: x[1], reverse=True)
                    button = direction_scores[0][0]
                # else: all directions failed, will be caught by stuck detection

            # DEBUG: Log each step with screen position
            path_valid = "n/a"
            if path and len(path) > 0:
                first_x, first_y = path[0]
                if 0 <= first_y < 9 and 0 <= first_x < 10:
                    path_valid = "W" if current_walkability[first_y][first_x] else "X!"
            logger.info(f"  step {step}: world=({player_x},{player_y}) screen=({curr_player_screen_x},{curr_player_screen_y}) target_screen=({target_screen_x},{target_screen_y}) path={path}[{path_valid}] -> {button_names.get(button, '?')} (blocked@pos: {[button_names.get(d) for d in blocked_here]})")

            # Track that we're trying this direction BEFORE the step
            last_button_tried = button
            self.env.step(button)

        # Loop finished - return current state
        obs = emulator.get_screen()
        overlay = create_walkability_overlay(obs, emulator)
        new_info = self.env._get_info()
        new_x, new_y = new_info["position"]

        dist = abs(target_world_x - new_x) + abs(target_world_y - new_y)
        if dist == 0:
            msg = f"Reached ({screen_x},{screen_y})"
        else:
            msg = f"Moving toward ({screen_x},{screen_y}), {dist} tiles away"

        return ToolResult(
            success=True,
            message=msg,
            screenshot=overlay,
            game_state=self._get_game_state(),
            walkability_text=format_walkability_for_prompt(emulator)
        )

    def _execute_button(self, params: dict) -> ToolResult:
        """Press a button."""
        from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt

        button = params.get("button", "a")
        button_map = {
            "a": 0, "b": 1, "start": 2, "select": 3,
            "up": 4, "down": 5, "left": 6, "right": 7
        }

        if button not in button_map:
            return ToolResult(success=False, message=f"Invalid button: {button}")

        # Press button
        self.env.step(button_map[button])

        # Get new state
        obs = self.env.emulator.get_screen()
        info = self.env._get_info()
        emulator = self.env.emulator

        if info.get("in_battle"):
            return ToolResult(
                success=True,
                message=f"Pressed {button}",
                screenshot=obs,
                game_state=self._get_game_state(),
                walkability_text=None
            )
        else:
            overlay = create_walkability_overlay(obs, emulator)
            return ToolResult(
                success=True,
                message=f"Pressed {button}",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=format_walkability_for_prompt(emulator)
            )

    def _execute_attack(self, params: dict) -> ToolResult:
        """Execute an attack in battle using verified navigation with proper timing."""
        move_slot = params.get("move_slot", 1)

        if not self.battle.is_in_battle():
            return ToolResult(
                success=False,
                message="Not in battle! Use navigator to explore."
            )

        # Wait for battle menu to be ready BEFORE selecting move
        # This prevents attacking during battle intro/animations
        if not self.battle.wait_for_battle_menu(timeout=120):
            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=False,
                message="Battle menu not ready yet. Wait for animations to finish.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Use BattleMenuNavigator for verified attack execution
        if not self.battle.execute_attack(move_slot):
            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=False,
                message=f"Failed to use move {move_slot}. Try again.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Advance through attack animation and text
        self.battle.advance_battle_text(max_presses=30)

        # Wait for enemy turn to complete and menu to return
        # This prevents spamming attack before enemy's turn finishes
        menu_returned = self.battle.wait_for_battle_menu(timeout=180)

        obs = self.env.emulator.get_screen()

        # Build battle status report
        info = self.env._get_info()
        report = self._format_battle_report(move_slot, info)

        # Check if battle ended (enemy fainted, we fainted, etc.)
        if not self.battle.is_in_battle():
            from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt
            emulator = self.env.emulator
            overlay = create_walkability_overlay(obs, emulator)
            return ToolResult(
                success=True,
                message=report + "\n\nBATTLE ENDED!",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=format_walkability_for_prompt(emulator)
            )

        if not menu_returned:
            # Still waiting for enemy turn - tell LLM to wait
            return ToolResult(
                success=True,
                message=report + "\n\nWaiting for enemy turn...",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        return ToolResult(
            success=True,
            message=report,
            screenshot=obs,
            game_state=self._get_game_state()
        )

    def _execute_run(self) -> ToolResult:
        """Try to run from battle using verified navigation with proper timing."""
        if not self.battle.is_in_battle():
            return ToolResult(
                success=False,
                message="Not in battle!"
            )

        # Wait for battle menu to be ready BEFORE selecting RUN
        if not self.battle.wait_for_battle_menu(timeout=120):
            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=False,
                message="Battle menu not ready yet. Wait for animations to finish.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Use BattleMenuNavigator for verified run execution
        if not self.battle.execute_run():
            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=False,
                message="Failed to select RUN. Try again.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Advance through "Got away safely!" or "Can't escape!" text
        self.battle.advance_battle_text(max_presses=10)

        obs = self.env.emulator.get_screen()

        if self.battle.is_in_battle():
            return ToolResult(
                success=False,
                message="Couldn't escape!",
                screenshot=obs,
                game_state=self._get_game_state()
            )
        else:
            from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt
            emulator = self.env.emulator
            overlay = create_walkability_overlay(obs, emulator)
            return ToolResult(
                success=True,
                message="Got away safely!",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=format_walkability_for_prompt(emulator)
            )

    def _execute_knowledge(self, params: dict) -> ToolResult:
        """Update the knowledge base."""
        action = params.get("action", "add_note")
        content = params.get("content", "")

        if action == "add_note" or action == "add":
            vital = params.get("vital", False)
            note_category = params.get("note_category", "general")
            self.knowledge.add_note(content, vital=vital, category=note_category)
            vital_marker = " [VITAL]" if vital else ""
            return ToolResult(success=True, message=f"Added note{vital_marker}: {content}")

        elif action == "remove_note" or action == "remove":
            self.knowledge.remove_note(content)
            return ToolResult(success=True, message=f"Removed note: {content}")

        elif action == "set_goal":
            self.knowledge.set_goal(content)
            return ToolResult(success=True, message=f"Set goal: {content}")

        elif action == "add_question":
            self.knowledge.add_question(content)
            return ToolResult(success=True, message=f"Added question: {content}")

        elif action == "resolve_question":
            self.knowledge.resolve_question(content)
            return ToolResult(success=True, message=f"Resolved question: {content}")

        elif action == "add_strategy":
            category = params.get("category", "battle")
            key = params.get("key", "")
            if category == "battle":
                self.knowledge.add_battle_strategy(key, content)
                return ToolResult(success=True, message=f"Added battle strategy for {key}: {content}")
            elif category == "area":
                self.knowledge.add_area_strategy(key, content)
                return ToolResult(success=True, message=f"Added area strategy for {key}: {content}")
            elif category == "grinding":
                level_range = params.get("level_range", "")
                self.knowledge.add_grinding_spot(key, level_range, content)
                return ToolResult(success=True, message=f"Added grinding spot: {key} ({level_range})")
            else:
                return ToolResult(success=False, message=f"Unknown strategy category: {category}")

        elif action == "learn_matchup":
            enemy_type = params.get("enemy_type", "")
            our_pokemon = params.get("our_pokemon", "")
            self.knowledge.learn_type_matchup(enemy_type, our_pokemon)
            return ToolResult(success=True, message=f"Learned: {our_pokemon} is effective against {enemy_type}")

        elif action == "record_failure":
            context = params.get("context", "")
            outcome = params.get("outcome", "")
            lesson = params.get("lesson", "")
            # Get current step from env if available
            step = 0
            if self.env:
                try:
                    step = self.env._step_count
                except Exception:
                    pass
            self.knowledge.record_failure(step, context, action, outcome, lesson)
            return ToolResult(success=True, message=f"Recorded failure: {context} -> {outcome}")

        elif action == "set_flag":
            flag = params.get("flag", "")
            value = params.get("value", True)
            self.knowledge.set_story_flag(flag, value)
            return ToolResult(success=True, message=f"Set flag {flag} = {value}")

        else:
            return ToolResult(success=False, message=f"Unknown action: {action}")

    def _execute_query_knowledge(self, params: dict) -> ToolResult:
        """Query the knowledge base for relevant information."""
        # Rate limit: only allow 1 call per 10 turns
        turns_since_last = self._current_turn - self._last_query_turn
        if turns_since_last < 10:
            remaining = 10 - turns_since_last
            # Get action suggestion based on current state
            info = self.env._get_info()
            if info.get("in_battle"):
                action_hint = "Use attack(1-4) to fight or run_away() to flee"
            else:
                from src.agents.llm.decision_context import _get_location_type
                loc_type = _get_location_type(info)
                if loc_type == "town":
                    action_hint = "Use find_route() to leave town and find wild Pokemon"
                elif loc_type in ("building", "pokecenter", "pokemart"):
                    action_hint = "Use find_exit() to leave this building"
                else:
                    action_hint = "Use navigator(x, y) to move around"
            return ToolResult(
                success=False,
                message=f"Knowledge cooldown: wait {remaining} more turns. Take action instead!\n→ {action_hint}"
            )
        self._last_query_turn = self._current_turn

        query = params.get("query", "")
        section = params.get("section", "all")
        situation = params.get("situation", "")

        lines = []

        # If situation is specified, get context-relevant knowledge
        if situation:
            context = self.knowledge.get_context_for_situation(situation)
            if context and context != "No specific knowledge for this situation.":
                lines.append(f"=== Knowledge for {situation} ===")
                lines.append(context)

        # If query is specified, search for it
        if query:
            results = self.knowledge.query(query, section=section, top_k=10)
            if results:
                lines.append(f"=== Search results for '{query}' ===")
                for result in results:
                    lines.append(f"  - {result}")
            else:
                lines.append(f"No results found for '{query}'")

        # If neither, show summary
        if not query and not situation:
            lines.append("=== Knowledge Base Summary ===")
            lines.append(self.knowledge.get_summary_stats())

            # Show open questions
            if self.knowledge.open_questions:
                lines.append("\nOpen questions:")
                for q in self.knowledge.open_questions[-5:]:
                    lines.append(f"  - {q}")

            # Show recent failures
            if self.knowledge.failures:
                lines.append("\nRecent failures:")
                for f in self.knowledge.failures[-3:]:
                    lines.append(f"  - {f.get('context', '')}: {f.get('lesson', f.get('outcome', ''))}")

        # Auto-execute an action based on current game state
        knowledge_text = "\n".join(lines) if lines else "Knowledge base is empty."

        info = self.env._get_info()
        if info.get("in_battle"):
            # In battle - auto-attack with move 1
            lines.append("\n→ AUTO-ACTION: Attacking...")
            action_result = self._execute_attack({"move_slot": 1})
        else:
            from src.agents.llm.decision_context import _get_location_type
            loc_type = _get_location_type(info)
            if loc_type == "town":
                lines.append("\n→ AUTO-ACTION: Leaving town...")
                action_result = self._execute_find_route()
            elif loc_type in ("building", "pokecenter", "pokemart"):
                lines.append("\n→ AUTO-ACTION: Finding exit...")
                action_result = self._execute_find_exit()
            else:
                # On route - pick a random walkable direction to explore
                lines.append("\n→ AUTO-ACTION: Exploring...")
                # Try to move in a direction
                from src.agents.llm.walkability import get_player_screen_tile
                px, py = get_player_screen_tile(self.env.emulator)
                # Try moving down first (into grass), or up if blocked
                action_result = self._execute_navigator({"x": px, "y": min(py + 2, 8)})

        # Combine knowledge + action result
        full_message = knowledge_text + "\n\n" + action_result.message

        return ToolResult(
            success=action_result.success,
            message=full_message,
            screenshot=action_result.screenshot,
            game_state=action_result.game_state,
            walkability_text=action_result.walkability_text
        )

    def _execute_think(self, params: dict) -> ToolResult:
        """Process thinking/reasoning - no game action, just records the thought."""
        import logging
        logger = logging.getLogger(__name__)

        situation = params.get("situation", "")
        analysis = params.get("analysis", "")
        options = params.get("options", "")
        plan = params.get("plan", "")

        # Build a summary of the thinking
        thought_parts = []
        if situation:
            thought_parts.append(f"Situation: {situation}")
        if analysis:
            thought_parts.append(f"Analysis: {analysis}")
        if options:
            thought_parts.append(f"Options: {options}")
        if plan:
            thought_parts.append(f"Plan: {plan}")

        full_thought = " | ".join(thought_parts) if thought_parts else plan

        # Log for debugging/auditing
        logger.info(f"[THINK] {full_thought[:500]}")

        # Save a condensed version to knowledge base for reference
        # Truncate to avoid bloating the KB
        summary = plan[:200] if plan else full_thought[:200]
        self.knowledge.add_note(
            f"[PLAN] {summary}",
            vital=False,
            category="reasoning"
        )

        # Return acknowledgment
        return ToolResult(
            success=True,
            message=f"Reasoning recorded. Plan: {plan[:300]}" if plan else "Reasoning recorded."
        )

    def _check_wait_condition(self, condition: str, info: dict) -> bool | None:
        """
        Check if a wait condition is met.

        Returns:
            True if condition met, False if not met, None if invalid condition
        """
        if condition == "dialogue_done":
            # No text box active
            return not info.get("menu_active", True)

        elif condition == "animation_done":
            # Walking/text animations complete, input accepted
            return info.get("waiting_for_input", False)

        elif condition == "battle_menu_ready":
            # Can select FIGHT/BAG/POKEMON/RUN
            return info.get("battle_menu_ready", False)

        elif condition == "battle_ends":
            # Battle is over
            return not info.get("in_battle", True)

        elif condition == "battle_starts":
            # Entered a battle
            return info.get("in_battle", False)

        return None  # Unknown condition

    def _execute_wait(self, params: dict) -> ToolResult:
        """Wait until a condition is met or timeout."""
        from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt

        condition = params.get("until", "frames")
        timeout = params.get("timeout", 120)
        frames = params.get("frames", 60)

        # Validate condition first
        valid_conditions = ["dialogue_done", "animation_done", "battle_menu_ready", "battle_ends", "frames"]
        # Also accept battle_starts but it will be rejected with helpful message below
        if condition not in valid_conditions and condition != "battle_starts":
            return ToolResult(
                success=False,
                message=f"Invalid wait condition: '{condition}'. Valid options: {', '.join(valid_conditions)}"
            )

        # Special check: battle_starts should almost never be used
        # Wild battles happen when WALKING through grass, not by waiting
        if condition == "battle_starts":
            from src.agents.llm.decision_context import _get_location_type
            from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt
            info = self.env._get_info()
            loc_type = _get_location_type(info)

            if loc_type in ("town", "building", "pokecenter", "pokemart"):
                return ToolResult(
                    success=False,
                    message=f"Cannot find wild Pokemon here! You're in a {loc_type}. Wild Pokemon only appear in tall grass on ROUTES. Leave this area and find a route."
                )
            else:
                # On routes/dungeons/anywhere else, don't wait - walk through grass instead!
                obs = self.env.emulator.get_screen()
                try:
                    overlay = create_walkability_overlay(obs, self.env.emulator)
                    walkability_text = format_walkability_for_prompt(self.env.emulator)
                except Exception:
                    overlay = obs
                    walkability_text = None
                return ToolResult(
                    success=False,
                    message="Don't wait for battles - WALK through tall grass! Wild Pokemon appear randomly when you move through tall grass (darker green patches). Use navigator(x, y) to walk around in grass areas. Each step has a chance to trigger a battle.",
                    screenshot=overlay,
                    game_state=self._get_game_state(),
                    walkability_text=walkability_text
                )

        if condition == "frames":
            # Fixed frame wait - tick emulator without pressing any button
            self.env.emulator.tick(frames)
            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=True,
                message=f"Waited {frames} frames",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Condition-based wait
        for frame in range(timeout):
            info = self.env._get_info()
            result = self._check_wait_condition(condition, info)
            if result is True:
                obs = self.env.emulator.get_screen()
                # Add walkability overlay if not in battle
                if not info.get("in_battle"):
                    try:
                        obs = create_walkability_overlay(obs, self.env.emulator)
                        walkability_text = format_walkability_for_prompt(self.env.emulator)
                    except Exception:
                        walkability_text = None
                else:
                    walkability_text = None

                return ToolResult(
                    success=True,
                    message=f"Condition '{condition}' met after {frame} frames",
                    screenshot=obs,
                    game_state=self._get_game_state(),
                    walkability_text=walkability_text
                )
            self.env.emulator.tick(1)  # Advance one frame without pressing buttons

        # Timeout reached
        obs = self.env.emulator.get_screen()
        info = self.env._get_info()
        if not info.get("in_battle"):
            try:
                obs = create_walkability_overlay(obs, self.env.emulator)
                walkability_text = format_walkability_for_prompt(self.env.emulator)
            except Exception:
                walkability_text = None
        else:
            walkability_text = None

        return ToolResult(
            success=False,
            message=f"Timeout waiting for '{condition}' after {timeout} frames",
            screenshot=obs,
            game_state=self._get_game_state(),
            walkability_text=walkability_text
        )

    def _execute_find_exit(self) -> ToolResult:
        """Find the exit/door in the current indoor location."""
        from src.agents.llm.walkability import (
            create_walkability_grid,
            create_walkability_overlay,
            format_walkability_for_prompt,
            get_player_screen_tile,
            find_path,
        )

        info = self.env._get_info()
        emulator = self.env.emulator

        # Check if we're indoors
        env_type = info.get("environment_type", "outdoors")
        if env_type == "outdoors":
            obs = emulator.get_screen()
            return ToolResult(
                success=False,
                message="You're outdoors - no need to find an exit. Use navigator() to explore.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Get exit locations (world coordinates)
        exits = info.get("exits", [])
        if not exits:
            obs = emulator.get_screen()
            return ToolResult(
                success=False,
                message="No exits detected on this map. Look for doors or stairs visually.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Get player's current positions
        player_world = info.get("position", (0, 0))
        player_screen = get_player_screen_tile(emulator)
        walkability = create_walkability_grid(emulator)

        # Convert exit world coordinates to screen coordinates
        # screen = world - player_world + player_screen
        exit_screen_coords = []
        for exit_world in exits:
            exit_x, exit_y = exit_world
            screen_x = exit_x - player_world[0] + player_screen[0]
            screen_y = exit_y - player_world[1] + player_screen[1]

            # Check if exit is on screen
            if 0 <= screen_x < 10 and 0 <= screen_y < 9:
                # Check if we can reach it
                path = find_path(walkability, player_screen[0], player_screen[1], screen_x, screen_y)
                reachable = path is not None
                exit_screen_coords.append({
                    "screen": (screen_x, screen_y),
                    "world": (exit_x, exit_y),
                    "reachable": reachable,
                    "distance": abs(screen_x - player_screen[0]) + abs(screen_y - player_screen[1])
                })

        # Sort by reachability and distance
        exit_screen_coords.sort(key=lambda e: (not e["reachable"], e["distance"]))

        obs = emulator.get_screen()
        try:
            overlay = create_walkability_overlay(obs, emulator)
            walkability_text = format_walkability_for_prompt(emulator)
        except Exception:
            overlay = obs
            walkability_text = None

        if not exit_screen_coords:
            # Exits exist but are off-screen - need to move toward them
            # Find direction to nearest exit
            nearest_exit = exits[0]
            dx = nearest_exit[0] - player_world[0]
            dy = nearest_exit[1] - player_world[1]

            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
                edge_x = 9 if dx > 0 else 0
                message = f"Exit is off-screen to the {direction}. Navigate to edge ({edge_x}, 4) to scroll the map."
            else:
                direction = "down" if dy > 0 else "up"
                edge_y = 8 if dy > 0 else 0
                message = f"Exit is off-screen {direction}. Navigate to edge (4, {edge_y}) to scroll the map."

            return ToolResult(
                success=True,
                message=message,
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=walkability_text
            )

        # Found exits on screen
        best_exit = exit_screen_coords[0]
        sx, sy = best_exit["screen"]

        if best_exit["reachable"]:
            # Auto-navigate to the exit instead of just giving directions
            nav_result = self._execute_navigator({"x": sx, "y": sy})
            message = f"EXIT FOUND at ({sx}, {sy}). Auto-navigating...\n\n{nav_result.message}"
            return ToolResult(
                success=nav_result.success,
                message=message,
                screenshot=nav_result.screenshot,
                game_state=nav_result.game_state,
                walkability_text=nav_result.walkability_text
            )
        else:
            message = f"Exit visible at ({sx}, {sy}) but blocked. Try navigating around obstacles to reach it."

        return ToolResult(
            success=True,
            message=message,
            screenshot=overlay,
            game_state=self._get_game_state(),
            walkability_text=walkability_text
        )

    def _execute_find_route(self) -> ToolResult:
        """Find the way to exit the current area by analyzing walkable screen edges."""
        from src.agents.llm.walkability import (
            create_walkability_grid,
            create_walkability_overlay,
            format_walkability_for_prompt,
            get_player_screen_tile,
            find_path,
        )
        from src.agents.llm.decision_context import _get_location_type

        info = self.env._get_info()
        emulator = self.env.emulator
        loc_type = _get_location_type(info)
        map_id = info.get("map_id", 0)
        player_world_pos = info.get("position", (0, 0))

        # If already on a route, just encourage exploration
        if loc_type == "route":
            obs = emulator.get_screen()
            overlay = create_walkability_overlay(obs, emulator)
            return ToolResult(
                success=True,
                message="You're already on a route! Walk through tall grass (darker green patches) to find wild Pokemon. Each step in grass has a chance to trigger a battle.",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=format_walkability_for_prompt(emulator)
            )

        # For buildings, suggest find_exit instead
        if loc_type in ("building", "pokecenter", "pokemart", "gym"):
            obs = emulator.get_screen()
            return ToolResult(
                success=False,
                message=f"You're in a {loc_type}. Use find_exit() to find the door, then exit to reach routes.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        # Dynamic edge analysis - works for any map
        walkability = create_walkability_grid(emulator)
        player_screen_x, player_screen_y = get_player_screen_tile(emulator)

        # Check for learned transitions from this map
        # We'll match them with reachable edges after we compute those
        known_exits = self.knowledge.get_known_exits(map_id)

        # Find walkable tiles on each edge and check if reachable
        edges = {
            "NORTH (row 0)": [(x, 0) for x in range(10) if walkability[0][x]],
            "SOUTH (row 8)": [(x, 8) for x in range(10) if walkability[8][x]],
            "WEST (col 0)": [(0, y) for y in range(9) if walkability[y][0]],
            "EAST (col 9)": [(9, y) for y in range(9) if walkability[y][9]],
        }

        # Check which edges have reachable tiles
        reachable_edges = {}
        for edge_name, tiles in edges.items():
            reachable_tiles = []
            for tx, ty in tiles:
                path = find_path(walkability, player_screen_x, player_screen_y, tx, ty)
                if path is not None:
                    distance = len(path)
                    reachable_tiles.append((tx, ty, distance))
            if reachable_tiles:
                # Sort by distance, keep closest
                reachable_tiles.sort(key=lambda t: t[2])
                reachable_edges[edge_name] = reachable_tiles[0]  # (x, y, distance)

        obs = emulator.get_screen()
        overlay = create_walkability_overlay(obs, emulator)
        walkability_text = format_walkability_for_prompt(emulator)

        if not reachable_edges:
            # No reachable edges - need to move to see more of the map
            # Find the general direction with the most walkable tiles visible
            visible_edges = {k: len(v) for k, v in edges.items() if v}
            if visible_edges:
                best_dir = max(visible_edges, key=visible_edges.get)
                return ToolResult(
                    success=True,
                    message=f"No reachable edge tiles from current position. Try moving toward {best_dir} to find a path. Look for W tiles that connect to the edge.",
                    screenshot=overlay,
                    game_state=self._get_game_state(),
                    walkability_text=walkability_text
                )
            else:
                return ToolResult(
                    success=True,
                    message="No walkable edge tiles visible. Move around to explore more of the map and find exits.",
                    screenshot=overlay,
                    game_state=self._get_game_state(),
                    walkability_text=walkability_text
                )

        # Build message with all reachable edges
        lines = []

        # Map direction names to edge names
        direction_to_edge = {
            "north": "NORTH (row 0)",
            "south": "SOUTH (row 8)",
            "west": "WEST (col 0)",
            "east": "EAST (col 9)",
        }

        # Match learned exits with reachable edges to get proper coordinates
        matched_exits = []
        if known_exits:
            for t in known_exits:
                edge_name = direction_to_edge.get(t.direction)
                if edge_name and edge_name in reachable_edges:
                    tx, ty, dist = reachable_edges[edge_name]
                    matched_exits.append({
                        "destination": t.to_map_name,
                        "direction": t.direction,
                        "edge_coords": (tx, ty),
                        "distance": dist,
                    })

        # First, show learned exits matched with reachable edges
        if matched_exits:
            lines.append("=== KNOWN DESTINATIONS ===")
            for exit in matched_exits:
                tx, ty = exit["edge_coords"]
                lines.append(f"  - {exit['destination']} ({exit['direction'].upper()}): navigator({tx}, {ty}) - {exit['distance']} steps")
            lines.append("")

        # Then show all reachable edges
        lines.append("=== REACHABLE EDGES ===")
        for edge_name, (tx, ty, dist) in sorted(reachable_edges.items(), key=lambda x: x[1][2]):
            lines.append(f"  - {edge_name}: navigator({tx}, {ty}) - {dist} steps away")

        # Auto-navigate to the best edge instead of just giving directions
        # This prevents the LLM from getting stuck in a find_route loop
        if matched_exits:
            best = matched_exits[0]
            tx, ty = best["edge_coords"]
            destination = best['destination']
            lines.append(f"\nAuto-navigating to {destination}...")
        else:
            closest_edge = min(reachable_edges.items(), key=lambda x: x[1][2])
            edge_name, (tx, ty, dist) = closest_edge
            destination = edge_name.split()[0]
            lines.append(f"\nAuto-navigating {destination}...")

        if loc_type == "town":
            lines.append("(Towns have NO wild Pokemon - leaving to find routes)")

        # Actually navigate to the edge
        nav_result = self._execute_navigator({"x": tx, "y": ty})

        # Combine messages
        full_message = "\n".join(lines) + "\n\n" + nav_result.message

        return ToolResult(
            success=nav_result.success,
            message=full_message,
            screenshot=nav_result.screenshot,
            game_state=nav_result.game_state,
            walkability_text=nav_result.walkability_text
        )

    def _format_battle_report(self, move_slot: int, info: dict) -> str:
        """Format a battle status report after an attack."""
        lines = [f"Used move {move_slot}!"]
        lines.append("")
        lines.append("=== BATTLE STATUS ===")

        player = info.get("player_pokemon", {})
        enemy = info.get("enemy_pokemon", {})

        if player:
            p_hp = player.get("hp", 0)
            p_max = player.get("max_hp", 1)
            p_pct = int(p_hp / max(p_max, 1) * 100)
            lines.append(f"Your {player.get('species_name', '???')} Lv{player.get('level', '?')}: {p_hp}/{p_max} HP ({p_pct}%)")

        if enemy:
            e_hp = enemy.get("hp", 0)
            e_max = enemy.get("max_hp", 1)
            e_pct = int(e_hp / max(e_max, 1) * 100)
            lines.append(f"Enemy {enemy.get('species_name', '???')} Lv{enemy.get('level', '?')}: {e_hp}/{e_max} HP ({e_pct}%)")

        return "\n".join(lines)

    def _get_game_state(self) -> dict[str, Any]:
        """Get current game state for tool result."""
        info = self.env._get_info()

        state = {
            "location": self.knowledge.get_map_name(info.get("map_id", 0)),
            "position": info.get("position", (0, 0)),
            "badges": info.get("badges", 0),
            "in_battle": info.get("in_battle", False),
            "hp_percent": int(info.get("party_hp", 0) / max(info.get("party_max_hp", 1), 1) * 100),
        }

        # Add environment type and exits for navigation
        env_type = info.get("environment_type", "outdoors")
        state["environment"] = env_type
        if env_type != "outdoors":
            exits = info.get("exits", [])
            if exits:
                state["exits"] = exits

        if info.get("in_battle"):
            state["battle"] = {
                "your_pokemon": info.get("player_pokemon", {}),
                "enemy_pokemon": info.get("enemy_pokemon", {}),
                "menu_ready": info.get("battle_menu_ready", False),
            }

        if info.get("menu_active"):
            state["menu_active"] = True
            state["has_choice"] = info.get("has_menu_choice", False)

        return state

    def _execute_heal_at_pokecenter(self) -> ToolResult:
        """Heal Pokemon at a Pokemon Center."""
        from src.agents.llm.decision_context import _get_location_type
        from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt

        # Check if we're in a Pokemon Center
        info = self.env._get_info()
        loc_type = _get_location_type(info)

        if loc_type != "pokecenter":
            return ToolResult(
                success=False,
                message=f"Not in a Pokemon Center! You're in a {loc_type}. Find and enter a Pokemon Center first."
            )

        # Check if already at full HP
        party_hp = info.get("party_hp", 0)
        party_max_hp = info.get("party_max_hp", 1)
        if party_hp >= party_max_hp:
            return ToolResult(
                success=True,
                message="Party is already at full HP! No need to heal."
            )

        # Walk up to the counter (nurse is at the top)
        for _ in range(5):
            self.env.step(4)  # up
            self.env.emulator.tick(10)

        # Talk to nurse (press A)
        self.env.step(0)  # A to initiate
        self.env.emulator.tick(30)

        # Press A through dialogue (nurse greeting)
        for _ in range(5):
            self.env.step(0)  # A to advance dialogue
            self.env.emulator.tick(20)

        # Confirm healing (usually "Yes" option)
        self.env.step(0)  # A to confirm
        self.env.emulator.tick(120)  # Wait for healing animation (jingle plays)

        # Press A to dismiss "Pokemon healed" message
        for _ in range(3):
            self.env.step(0)  # A to advance
            self.env.emulator.tick(20)

        # Press B to close any remaining dialogue
        for _ in range(3):
            self.env.step(1)  # B
            self.env.emulator.tick(10)

        # Get updated state
        obs = self.env.emulator.get_screen()
        new_info = self.env._get_info()
        new_hp = new_info.get("party_hp", 0)
        new_max_hp = new_info.get("party_max_hp", 1)
        hp_pct = int(new_hp / max(new_max_hp, 1) * 100)

        try:
            overlay = create_walkability_overlay(obs, self.env.emulator)
            walkability_text = format_walkability_for_prompt(self.env.emulator)
        except Exception:
            overlay = obs
            walkability_text = None

        if new_hp >= new_max_hp:
            return ToolResult(
                success=True,
                message=f"Pokemon fully healed! Party HP: {hp_pct}%",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=walkability_text
            )
        else:
            return ToolResult(
                success=False,
                message=f"Healing may have failed. Party HP: {hp_pct}%. Try pressing A to talk to the nurse again.",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=walkability_text
            )

    def _execute_use_item(self, params: dict) -> ToolResult:
        """Use an item from the bag."""
        from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt

        item_name = params.get("item_name", "")
        pokemon_slot = params.get("pokemon_slot", 1)

        if not item_name:
            return ToolResult(
                success=False,
                message="No item specified! Provide item_name like 'Potion' or 'Antidote'."
            )

        # Check if in battle - different menu flow
        info = self.env._get_info()
        in_battle = info.get("in_battle", False)

        if in_battle:
            # In battle: navigate to BAG option (hardcoded for battle menu)
            self.env.step(5)  # down to BAG
            self.env.emulator.tick(10)
            self.env.step(0)  # A to select BAG
            self.env.emulator.tick(30)  # Wait for bag menu

            # In bag, items are listed - select first item (simplified)
            self.env.step(0)  # A to select first item
            self.env.emulator.tick(20)

            # Select Pokemon to use on
            for _ in range(pokemon_slot - 1):
                self.env.step(5)  # down to correct Pokemon
                self.env.emulator.tick(5)

            self.env.step(0)  # A to use
            self.env.emulator.tick(60)  # Wait for item use animation

            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=True,
                message=f"Attempted to use {item_name} on Pokemon #{pokemon_slot} in battle.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        else:
            # Use MenuNavigator for verified menu navigation
            if not self.menu.open_item_menu():
                # Close any partially opened menu
                self.menu.close_menu()
                return ToolResult(
                    success=False,
                    message="Failed to open Item menu. Try again."
                )

            # For now, select first item (simplified - would need item search)
            self.menu.select_current()  # Select first item
            self.menu.wait_for_menu(timeout=30)
            self.menu.select_current()  # Select "USE"
            self.menu.wait_for_menu(timeout=30)

            # Navigate to correct Pokemon slot
            self.menu.move_cursor_to(pokemon_slot - 1)
            self.menu.select_current()  # Use on this Pokemon

            # Wait for item use animation
            self.env.emulator.tick(60)

            # Advance through any dialogue ("used Potion on X")
            self.menu.advance_dialogue(max_presses=3)

            # Close any remaining menus
            self.menu.close_menu()

            obs = self.env.emulator.get_screen()
            try:
                overlay = create_walkability_overlay(obs, self.env.emulator)
                walkability_text = format_walkability_for_prompt(self.env.emulator)
            except Exception:
                overlay = obs
                walkability_text = None

            return ToolResult(
                success=True,
                message=f"Attempted to use {item_name} on Pokemon #{pokemon_slot}.",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=walkability_text
            )

    def _execute_check_pokemon(self) -> ToolResult:
        """Open Pokemon menu and get party status."""
        from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt

        info = self.env._get_info()

        # If in battle, can't check Pokemon menu normally
        if info.get("in_battle"):
            # Return battle Pokemon info directly
            player_pokemon = info.get("player_pokemon", {})
            party_info = f"Active Pokemon: {player_pokemon.get('species_name', 'Unknown')} "
            party_info += f"Lv{player_pokemon.get('level', '?')} "
            hp = player_pokemon.get('hp', 0)
            max_hp = player_pokemon.get('max_hp', 1)
            party_info += f"HP: {hp}/{max_hp} ({int(hp/max(max_hp,1)*100)}%)\n"
            party_info += f"Moves: {', '.join(player_pokemon.get('moves', ['unknown']))}"

            return ToolResult(
                success=True,
                message=f"In battle - showing active Pokemon:\n{party_info}",
                game_state=self._get_game_state()
            )

        # Use MenuNavigator for verified menu navigation
        if not self.menu.open_pokemon_menu():
            # Close any partially opened menu
            self.menu.close_menu()
            return ToolResult(
                success=False,
                message="Failed to open Pokemon menu. Try pressing START first."
            )

        # Read party info from game state
        new_info = self.env._get_info()
        party_count = new_info.get("party_count", 0)
        party_hp = new_info.get("party_hp", 0)
        party_max_hp = new_info.get("party_max_hp", 1)
        party_alive = new_info.get("party_alive", 0)

        party_summary = f"Party: {party_alive}/{party_count} Pokemon conscious\n"
        party_summary += f"Total HP: {party_hp}/{party_max_hp} ({int(party_hp/max(party_max_hp,1)*100)}%)"

        # Close the menu using verified navigation
        self.menu.close_menu()

        obs = self.env.emulator.get_screen()
        try:
            overlay = create_walkability_overlay(obs, self.env.emulator)
            walkability_text = format_walkability_for_prompt(self.env.emulator)
        except Exception:
            overlay = obs
            walkability_text = None

        return ToolResult(
            success=True,
            message=f"Pokemon party status:\n{party_summary}",
            screenshot=overlay,
            game_state=self._get_game_state(),
            walkability_text=walkability_text
        )

    def _execute_save_game(self) -> ToolResult:
        """Save the game."""
        from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt

        info = self.env._get_info()

        # Can't save in battle
        if info.get("in_battle"):
            return ToolResult(
                success=False,
                message="Cannot save during battle!"
            )

        # Use MenuNavigator for verified menu navigation
        if not self.menu.open_save_menu():
            # Close any partially opened menu
            self.menu.close_menu()
            return ToolResult(
                success=False,
                message="Failed to open SAVE menu. Try again."
            )

        # Wait for save prompt text, then confirm "Yes"
        self.menu.wait_for_text()
        self.menu.select_current()  # A to confirm save

        # Wait for save to complete (takes a while due to writing)
        self.env.emulator.tick(120)

        # Advance through "saved the game" dialogue
        self.menu.advance_dialogue(max_presses=3)

        # Close any remaining menus
        self.menu.close_menu()

        obs = self.env.emulator.get_screen()
        try:
            overlay = create_walkability_overlay(obs, self.env.emulator)
            walkability_text = format_walkability_for_prompt(self.env.emulator)
        except Exception:
            overlay = obs
            walkability_text = None

        return ToolResult(
            success=True,
            message="Game saved!",
            screenshot=overlay,
            game_state=self._get_game_state(),
            walkability_text=walkability_text
        )

    def _execute_switch_pokemon(self, params: dict) -> ToolResult:
        """Switch to a different Pokemon."""
        from src.agents.llm.walkability import create_walkability_overlay, format_walkability_for_prompt

        pokemon_slot = params.get("pokemon_slot", 1)

        if pokemon_slot < 1 or pokemon_slot > 6:
            return ToolResult(
                success=False,
                message=f"Invalid Pokemon slot: {pokemon_slot}. Must be 1-6."
            )

        info = self.env._get_info()
        in_battle = info.get("in_battle", False)

        if in_battle:
            # Use BattleMenuNavigator for verified switch in battle
            if not self.battle.execute_switch(pokemon_slot):
                obs = self.env.emulator.get_screen()
                return ToolResult(
                    success=False,
                    message=f"Failed to switch to Pokemon #{pokemon_slot}. Try again.",
                    screenshot=obs,
                    game_state=self._get_game_state()
                )

            # Advance through switch animation and text
            self.battle.advance_battle_text(max_presses=10)

            obs = self.env.emulator.get_screen()
            return ToolResult(
                success=True,
                message=f"Switched to Pokemon #{pokemon_slot} in battle.",
                screenshot=obs,
                game_state=self._get_game_state()
            )

        else:
            # Use MenuNavigator for verified menu navigation
            if not self.menu.open_pokemon_menu():
                # Close any partially opened menu
                self.menu.close_menu()
                return ToolResult(
                    success=False,
                    message="Failed to open Pokemon menu. Try again."
                )

            # Navigate to desired Pokemon slot
            if not self.menu.navigate_to_pokemon(pokemon_slot):
                self.menu.close_menu()
                return ToolResult(
                    success=False,
                    message=f"Failed to navigate to Pokemon #{pokemon_slot}."
                )

            # Select Pokemon to open submenu
            self.menu.select_current()
            self.menu.wait_for_menu(timeout=30)

            # Choose SWITCH option from submenu (usually index 1: STATS=0, SWITCH=1)
            self.menu.move_cursor_to(1)  # Navigate to SWITCH
            self.menu.select_current()  # Select SWITCH
            self.menu.wait_for_menu(timeout=30)

            # Select where to put it (first slot = index 0)
            self.menu.move_cursor_to(0)  # First slot
            self.menu.select_current()  # Confirm swap

            # Wait for swap animation
            self.env.emulator.tick(30)

            # Close any remaining menus
            self.menu.close_menu()

            obs = self.env.emulator.get_screen()
            try:
                overlay = create_walkability_overlay(obs, self.env.emulator)
                walkability_text = format_walkability_for_prompt(self.env.emulator)
            except Exception:
                overlay = obs
                walkability_text = None

            return ToolResult(
                success=True,
                message=f"Moved Pokemon #{pokemon_slot} to front of party.",
                screenshot=overlay,
                game_state=self._get_game_state(),
                walkability_text=walkability_text
            )

    def _execute_set_goal(self, params: dict) -> ToolResult:
        """Set a new main goal with optional template."""
        from src.agents.llm.knowledge import GOAL_TEMPLATES

        goal_description = params.get("goal", "")
        template = params.get("template", "")

        # If template specified, use it
        if template:
            if template not in GOAL_TEMPLATES:
                available = ", ".join(GOAL_TEMPLATES.keys())
                return ToolResult(
                    success=False,
                    message=f"Unknown template: '{template}'. Available: {available}"
                )

            goal = self.knowledge.set_main_goal(goal_description or template, template=template)
            tmpl = GOAL_TEMPLATES[template]
            subgoal_list = "\n".join(f"  {i+1}. {sg}" for i, sg in enumerate(tmpl.get("subgoals", [])))
            return ToolResult(
                success=True,
                message=f"Set goal: {goal.description}\n\nSubgoals:\n{subgoal_list}"
            )

        # Custom goal without template
        if not goal_description:
            return ToolResult(
                success=False,
                message="Provide either 'goal' description or 'template' name."
            )

        goal = self.knowledge.set_main_goal(goal_description)
        return ToolResult(
            success=True,
            message=f"Set goal: {goal.description}\n\nNo subgoals defined. Use add_subgoal to break it down."
        )

    def _execute_add_subgoal(self, params: dict) -> ToolResult:
        """Add a subgoal to the current active goal."""
        subgoal_desc = params.get("subgoal", "")

        if not subgoal_desc:
            return ToolResult(
                success=False,
                message="Provide a subgoal description."
            )

        subgoal = self.knowledge.add_subgoal(subgoal_desc)
        if subgoal is None:
            return ToolResult(
                success=False,
                message="No active goal to add subgoal to. Use set_goal first."
            )

        return ToolResult(
            success=True,
            message=f"Added subgoal: {subgoal.description}"
        )

    def _execute_complete_subgoal(self) -> ToolResult:
        """Mark the current subgoal as completed."""
        # Get current subgoal before completing for the message
        current = self.knowledge.goal_stack.get_current_subgoal()
        if current is None:
            return ToolResult(
                success=False,
                message="No active subgoal to complete."
            )

        completed_desc = current.description
        success = self.knowledge.complete_current_subgoal()

        if not success:
            return ToolResult(
                success=False,
                message="Failed to complete subgoal."
            )

        # Check if goal is now complete
        active = self.knowledge.goal_stack.get_active_goal()
        if active is None or active.status == "completed":
            return ToolResult(
                success=True,
                message=f"Completed subgoal: {completed_desc}\n\nGOAL COMPLETE! All subgoals finished."
            )

        # Get next subgoal
        next_sub = active.get_current_subgoal()
        progress = int(active.progress * 100)

        if next_sub:
            return ToolResult(
                success=True,
                message=f"Completed: {completed_desc}\nProgress: {progress}%\nNext: {next_sub.description}"
            )
        else:
            return ToolResult(
                success=True,
                message=f"Completed: {completed_desc}\nProgress: {progress}%"
            )

    def _execute_check_progress(self) -> ToolResult:
        """Check progress on current goal and subgoals."""
        progress = self.knowledge.get_goal_progress()

        if not progress.get("active"):
            return ToolResult(
                success=True,
                message="No active goal set. Use set_goal to set a goal."
            )

        lines = [
            f"GOAL: {progress['goal']}",
            f"Status: {progress['status']}",
            f"Progress: {progress['progress']}",
        ]

        if progress.get("current_subgoal"):
            lines.append(f"\nCurrent step: {progress['current_subgoal']}")

        if progress.get("completed_subgoals"):
            lines.append("\nCompleted:")
            for sg in progress["completed_subgoals"]:
                lines.append(f"  [x] {sg}")

        if progress.get("pending_subgoals"):
            lines.append("\nPending:")
            for sg in progress["pending_subgoals"]:
                lines.append(f"  [ ] {sg}")

        return ToolResult(
            success=True,
            message="\n".join(lines)
        )
