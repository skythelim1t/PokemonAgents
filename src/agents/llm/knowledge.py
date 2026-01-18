"""Hierarchical knowledge base for LLM agent."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Note:
    """A timestamped note with importance flag."""
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    vital: bool = False  # If True, never auto-prune
    category: str = "general"  # general, battle, navigation, item, npc

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "vital": self.vital,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        return cls(
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            vital=data.get("vital", False),
            category=data.get("category", "general"),
        )

    def matches_query(self, query: str) -> bool:
        """Check if note matches a search query."""
        query_lower = query.lower()
        return (
            query_lower in self.content.lower() or
            query_lower in self.category.lower()
        )


@dataclass
class MapTransition:
    """A recorded map transition (learned exit point)."""
    from_map_id: int
    to_map_id: int
    exit_world_coords: tuple[int, int]  # World coordinates where transition happened
    direction: str  # "north", "south", "east", "west"
    from_map_name: str = ""
    to_map_name: str = ""

    def to_dict(self) -> dict:
        return {
            "from_map_id": self.from_map_id,
            "to_map_id": self.to_map_id,
            "exit_world_coords": list(self.exit_world_coords),
            "direction": self.direction,
            "from_map_name": self.from_map_name,
            "to_map_name": self.to_map_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MapTransition":
        return cls(
            from_map_id=data.get("from_map_id", 0),
            to_map_id=data.get("to_map_id", 0),
            exit_world_coords=tuple(data.get("exit_world_coords", [0, 0])),
            direction=data.get("direction", "unknown"),
            from_map_name=data.get("from_map_name", ""),
            to_map_name=data.get("to_map_name", ""),
        )


@dataclass
class WorldState:
    """Knowledge about the game world."""
    current_location: str = "Unknown"
    current_map_id: int = 0
    visited_locations: list[str] = field(default_factory=list)
    visited_map_ids: set[int] = field(default_factory=set)
    # Learned map transitions (exit points between maps)
    map_transitions: list[MapTransition] = field(default_factory=list)
    story_flags: dict[str, bool] = field(default_factory=lambda: {
        "got_starter": False,
        "got_pokedex": False,
        "delivered_parcel": False,
        "beat_brock": False,
        "beat_misty": False,
        "beat_surge": False,
        "beat_erika": False,
        "beat_koga": False,
        "beat_sabrina": False,
        "beat_blaine": False,
        "beat_giovanni": False,
    })
    discovered_items: list[str] = field(default_factory=list)

    # Map names lookup
    MAP_NAMES: dict[int, str] = field(default_factory=lambda: {
        0: "Pallet Town",
        1: "Viridian City",
        2: "Pewter City",
        3: "Cerulean City",
        12: "Route 1",
        13: "Route 2",
        14: "Route 3",
        33: "Route 22",
        37: "Red's House 1F",
        38: "Red's House 2F",
        39: "Blue's House",
        40: "Oak's Lab",
        41: "Viridian Pokemon Center",
        42: "Viridian Pokemart",
        43: "Viridian School",
        44: "Viridian House",
        45: "Viridian Gym",
        51: "Pewter Pokemon Center",
        52: "Pewter Pokemart",
        53: "Pewter Museum 1F",
        54: "Pewter Gym",
        55: "Pewter House 1",
        56: "Pewter House 2",
        59: "Mt. Moon 1F",
        60: "Mt. Moon B1F",
        61: "Mt. Moon B2F",
        62: "Cerulean Pokemon Center",
        63: "Cerulean Gym",
        64: "Cerulean Bike Shop",
        65: "Cerulean Pokemart",
    })

    def update_location(self, map_id: int) -> None:
        """Update current location and track visited."""
        self.current_map_id = map_id
        self.current_location = self.MAP_NAMES.get(map_id, f"Unknown Area ({map_id})")

        if map_id not in self.visited_map_ids:
            self.visited_map_ids.add(map_id)
            if self.current_location not in self.visited_locations:
                self.visited_locations.append(self.current_location)

    def set_flag(self, flag: str, value: bool = True) -> None:
        """Set a story flag."""
        self.story_flags[flag] = value

    def record_transition(
        self,
        from_map_id: int,
        to_map_id: int,
        exit_coords: tuple[int, int],
        direction: str,
    ) -> MapTransition:
        """
        Record a map transition for future reference.

        Args:
            from_map_id: Map ID we're leaving
            to_map_id: Map ID we're entering
            exit_coords: World coordinates where the transition happened
            direction: Which direction we exited ("north", "south", "east", "west")

        Returns:
            The recorded MapTransition
        """
        # Check if we already have this transition
        for t in self.map_transitions:
            if t.from_map_id == from_map_id and t.to_map_id == to_map_id:
                # Update existing transition if coords are different
                if t.exit_world_coords != exit_coords:
                    t.exit_world_coords = exit_coords
                    t.direction = direction
                return t

        # Create new transition
        transition = MapTransition(
            from_map_id=from_map_id,
            to_map_id=to_map_id,
            exit_world_coords=exit_coords,
            direction=direction,
            from_map_name=self.MAP_NAMES.get(from_map_id, f"Map {from_map_id}"),
            to_map_name=self.MAP_NAMES.get(to_map_id, f"Map {to_map_id}"),
        )
        self.map_transitions.append(transition)
        logger.info(f"Learned transition: {transition.from_map_name} -> {transition.to_map_name} via {direction} at {exit_coords}")
        return transition

    def get_exits_from_map(self, map_id: int) -> list[MapTransition]:
        """Get all known exits from a specific map."""
        return [t for t in self.map_transitions if t.from_map_id == map_id]

    def get_transition_to_map(self, from_map_id: int, to_map_id: int) -> MapTransition | None:
        """Get a specific transition between two maps."""
        for t in self.map_transitions:
            if t.from_map_id == from_map_id and t.to_map_id == to_map_id:
                return t
        return None

    def to_dict(self) -> dict:
        return {
            "current_location": self.current_location,
            "current_map_id": self.current_map_id,
            "visited_locations": self.visited_locations,
            "visited_map_ids": list(self.visited_map_ids),
            "map_transitions": [t.to_dict() for t in self.map_transitions],
            "story_flags": self.story_flags,
            "discovered_items": self.discovered_items,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldState":
        ws = cls()
        ws.current_location = data.get("current_location", "Unknown")
        ws.current_map_id = data.get("current_map_id", 0)
        ws.visited_locations = data.get("visited_locations", [])
        ws.visited_map_ids = set(data.get("visited_map_ids", []))
        ws.map_transitions = [
            MapTransition.from_dict(t) for t in data.get("map_transitions", [])
        ]
        ws.story_flags = data.get("story_flags", ws.story_flags)
        ws.discovered_items = data.get("discovered_items", [])
        return ws


@dataclass
class PartyKnowledge:
    """Knowledge about the player's party."""
    pokemon: list[dict[str, Any]] = field(default_factory=list)
    # Type advantages we've learned (enemy_type -> our_best_pokemon)
    type_matchups: dict[str, str] = field(default_factory=dict)
    # Moves that worked well
    effective_moves: list[str] = field(default_factory=list)

    def update_pokemon(self, pokemon_data: list[dict]) -> None:
        """Update party pokemon info."""
        self.pokemon = pokemon_data

    def learn_matchup(self, enemy_type: str, our_pokemon: str) -> None:
        """Record a type matchup that worked."""
        self.type_matchups[enemy_type.lower()] = our_pokemon

    def add_effective_move(self, move: str) -> None:
        """Record a move that was effective."""
        if move not in self.effective_moves:
            self.effective_moves.append(move)

    def to_dict(self) -> dict:
        return {
            "pokemon": self.pokemon,
            "type_matchups": self.type_matchups,
            "effective_moves": self.effective_moves,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PartyKnowledge":
        pk = cls()
        pk.pokemon = data.get("pokemon", [])
        pk.type_matchups = data.get("type_matchups", {})
        pk.effective_moves = data.get("effective_moves", [])
        return pk


@dataclass
class Strategy:
    """A learned strategy."""
    name: str
    context: str  # When to use it
    steps: str  # What to do
    success_rate: float = 0.0
    times_used: int = 0


@dataclass
class Strategies:
    """Collection of learned strategies."""
    battle_strategies: dict[str, str] = field(default_factory=dict)  # opponent -> strategy
    area_strategies: dict[str, str] = field(default_factory=dict)  # location -> strategy
    grinding_spots: list[dict[str, Any]] = field(default_factory=list)

    def add_battle_strategy(self, opponent: str, strategy: str) -> None:
        """Add or update a battle strategy."""
        self.battle_strategies[opponent.lower()] = strategy

    def add_area_strategy(self, area: str, strategy: str) -> None:
        """Add or update an area strategy."""
        self.area_strategies[area.lower()] = strategy

    def add_grinding_spot(self, location: str, level_range: str, notes: str) -> None:
        """Add a grinding spot."""
        self.grinding_spots.append({
            "location": location,
            "level_range": level_range,
            "notes": notes,
        })

    def to_dict(self) -> dict:
        return {
            "battle_strategies": self.battle_strategies,
            "area_strategies": self.area_strategies,
            "grinding_spots": self.grinding_spots,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Strategies":
        s = cls()
        s.battle_strategies = data.get("battle_strategies", {})
        s.area_strategies = data.get("area_strategies", {})
        s.grinding_spots = data.get("grinding_spots", [])
        return s


@dataclass
class Failure:
    """A recorded failure to learn from."""
    step: int
    context: str
    action: str
    outcome: str
    lesson: str = ""


@dataclass
class Goal:
    """A hierarchical goal with subgoals and progress tracking."""
    id: str                          # Unique identifier
    description: str                 # "Beat Brock"
    subgoals: list["Goal"] = field(default_factory=list)
    status: str = "active"           # active, completed, blocked, abandoned
    progress: float = 0.0            # 0.0 to 1.0
    blocked_reason: str | None = None  # Why it's blocked (e.g., "Need CUT")
    completion_flag: str | None = None  # Story flag to auto-complete on
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict:
        """Serialize goal to dict."""
        return {
            "id": self.id,
            "description": self.description,
            "subgoals": [sg.to_dict() for sg in self.subgoals],
            "status": self.status,
            "progress": self.progress,
            "blocked_reason": self.blocked_reason,
            "completion_flag": self.completion_flag,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        """Deserialize goal from dict."""
        goal = cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            status=data.get("status", "active"),
            progress=data.get("progress", 0.0),
            blocked_reason=data.get("blocked_reason"),
            completion_flag=data.get("completion_flag"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
        )
        goal.subgoals = [Goal.from_dict(sg) for sg in data.get("subgoals", [])]
        return goal

    def get_current_subgoal(self) -> "Goal | None":
        """Get the first active subgoal."""
        for subgoal in self.subgoals:
            if subgoal.status == "active":
                return subgoal
        return None

    def update_progress(self) -> None:
        """Update progress based on completed subgoals."""
        if not self.subgoals:
            return
        completed = sum(1 for sg in self.subgoals if sg.status == "completed")
        self.progress = completed / len(self.subgoals)


@dataclass
class GoalStack:
    """Manages hierarchical goals."""
    goals: list[Goal] = field(default_factory=list)
    max_goals: int = 10

    def get_active_goal(self) -> Goal | None:
        """Get the current active main goal."""
        for goal in self.goals:
            if goal.status == "active":
                return goal
        return None

    def get_current_subgoal(self) -> Goal | None:
        """Get the current subgoal of the active goal."""
        active = self.get_active_goal()
        if active:
            return active.get_current_subgoal()
        return None

    def add_goal(self, description: str, subgoals: list[str] | None = None,
                 completion_flag: str | None = None) -> Goal:
        """Add a new main goal with optional subgoals."""
        import uuid
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        goal = Goal(
            id=goal_id,
            description=description,
            completion_flag=completion_flag,
        )
        if subgoals:
            for i, sg_desc in enumerate(subgoals):
                sg = Goal(
                    id=f"{goal_id}_sub{i}",
                    description=sg_desc,
                )
                goal.subgoals.append(sg)

        # Mark other goals as completed/abandoned if adding new main goal
        for existing in self.goals:
            if existing.status == "active":
                existing.status = "abandoned"

        self.goals.append(goal)

        # Enforce max goals
        if len(self.goals) > self.max_goals:
            # Remove oldest completed/abandoned goals
            self.goals = [g for g in self.goals if g.status == "active"] + \
                         [g for g in self.goals if g.status != "active"][-self.max_goals:]

        return goal

    def complete_current_subgoal(self) -> bool:
        """Mark the current subgoal as completed and advance."""
        active = self.get_active_goal()
        if not active:
            return False

        current = active.get_current_subgoal()
        if current:
            current.status = "completed"
            current.completed_at = datetime.now().isoformat()
            active.update_progress()
            logger.info(f"Completed subgoal: {current.description}")

            # Check if all subgoals done
            if active.progress >= 1.0:
                active.status = "completed"
                active.completed_at = datetime.now().isoformat()
                logger.info(f"Completed goal: {active.description}")
            return True
        return False

    def block_goal(self, reason: str) -> bool:
        """Mark the current goal as blocked."""
        active = self.get_active_goal()
        if active:
            active.status = "blocked"
            active.blocked_reason = reason
            logger.info(f"Blocked goal: {active.description} - {reason}")
            return True
        return False

    def to_dict(self) -> dict:
        """Serialize goal stack to dict."""
        return {
            "goals": [g.to_dict() for g in self.goals],
            "max_goals": self.max_goals,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalStack":
        """Deserialize goal stack from dict."""
        stack = cls(max_goals=data.get("max_goals", 10))
        stack.goals = [Goal.from_dict(g) for g in data.get("goals", [])]
        return stack


# Predefined goal templates for common objectives
GOAL_TEMPLATES: dict[str, dict] = {
    "beat_brock": {
        "description": "Beat Brock and earn the Boulder Badge",
        "subgoals": [
            "Leave Pallet Town and head north to Route 1",
            "Travel through Route 1 to Viridian City",
            "Continue north through Route 2 to Pewter City",
            "Train Pokemon to level 12+",
            "Enter Pewter Gym and defeat Brock",
        ],
        "completion_flag": "beat_brock",
    },
    "beat_misty": {
        "description": "Beat Misty and earn the Cascade Badge",
        "subgoals": [
            "Travel east from Pewter to Route 3",
            "Navigate through Mt. Moon",
            "Reach Cerulean City",
            "Train Pokemon to level 20+",
            "Enter Cerulean Gym and defeat Misty",
        ],
        "completion_flag": "beat_misty",
    },
    "beat_surge": {
        "description": "Beat Lt. Surge and earn the Thunder Badge",
        "subgoals": [
            "Get S.S. Anne ticket from Bill",
            "Travel south to Vermilion City",
            "Board S.S. Anne and help the Captain",
            "Get CUT HM and teach to a Pokemon",
            "Enter Vermilion Gym and defeat Lt. Surge",
        ],
        "completion_flag": "beat_surge",
    },
    "get_starter": {
        "description": "Get your first Pokemon from Oak's Lab",
        "subgoals": [
            "Go downstairs and leave your house",
            "Try to leave Pallet Town (Oak will stop you)",
            "Follow Oak to his lab",
            "Choose a starter Pokemon",
        ],
        "completion_flag": "got_starter",
    },
    "deliver_parcel": {
        "description": "Deliver Oak's Parcel and get the Pokedex",
        "subgoals": [
            "Travel north to Viridian City",
            "Visit the Pokemart and receive the parcel",
            "Return to Pallet Town",
            "Deliver parcel to Oak and receive Pokedex",
        ],
        "completion_flag": "delivered_parcel",
    },
    "train_pokemon": {
        "description": "Train Pokemon by battling wild Pokemon",
        "subgoals": [
            "Find a route with wild Pokemon",
            "Walk in tall grass to encounter wild Pokemon",
            "Battle and defeat wild Pokemon to gain XP",
            "Continue until Pokemon level up",
        ],
        "completion_flag": None,  # No auto-complete, manual check needed
    },
}


@dataclass
class KnowledgeBase:
    """
    Hierarchical knowledge base for the Pokemon agent.

    Organized into categories:
    - world_state: Location, visited places, story progress
    - party: Pokemon info, type matchups
    - strategies: Learned strategies for battles/areas
    - failures: Past mistakes to learn from
    - open_questions: Things to figure out
    - notes: General observations
    """

    # Structured knowledge
    world_state: WorldState = field(default_factory=WorldState)
    party: PartyKnowledge = field(default_factory=PartyKnowledge)
    strategies: Strategies = field(default_factory=Strategies)

    # Failures to learn from
    failures: list[dict[str, Any]] = field(default_factory=list)
    max_failures: int = 20

    # Open questions/goals
    open_questions: list[str] = field(default_factory=list)
    current_goal: str = "Explore and find wild Pokemon to battle"

    # General notes (with timestamps and vital flags)
    notes: list[Note] = field(default_factory=list)
    max_notes: int = 50  # Higher limit since we can prune non-vital

    # Hierarchical goal management
    goal_stack: GoalStack = field(default_factory=GoalStack)

    # ============= World State Methods =============

    def update_location(self, map_id: int) -> None:
        """Update current location."""
        self.world_state.update_location(map_id)

    def set_story_flag(self, flag: str, value: bool = True) -> None:
        """Set a story progress flag."""
        self.world_state.set_flag(flag, value)

    def get_map_name(self, map_id: int) -> str:
        """Get the name of a map."""
        return self.world_state.MAP_NAMES.get(map_id, f"Unknown Area ({map_id})")

    def record_map_transition(
        self,
        from_map_id: int,
        to_map_id: int,
        exit_coords: tuple[int, int],
        direction: str,
    ) -> None:
        """Record a map transition for future navigation."""
        self.world_state.record_transition(from_map_id, to_map_id, exit_coords, direction)

    def get_known_exits(self, map_id: int) -> list:
        """Get all known exits from a map."""
        return self.world_state.get_exits_from_map(map_id)

    def sync_from_game_state(self, info: dict) -> None:
        """
        Auto-sync story flags from actual game state.

        This detects progress automatically rather than relying on
        the LLM to manually call update_knowledge.

        Args:
            info: Game state dict from env._get_info()
        """
        # Detect got_starter from party count
        party_count = info.get("party_count", 0)
        if party_count > 0 and not self.world_state.story_flags.get("got_starter"):
            self.world_state.set_flag("got_starter", True)
            logger.info("Auto-detected: got_starter = True")

        # Detect got_pokedex from pokemon seen/caught
        # If player has seen any Pokemon, they have the Pokedex
        pokemon_seen = info.get("pokemon_seen", 0)
        pokemon_caught = info.get("pokemon_caught", 0)
        if (pokemon_seen > 0 or pokemon_caught > 0) and not self.world_state.story_flags.get("got_pokedex"):
            self.world_state.set_flag("got_pokedex", True)
            logger.info("Auto-detected: got_pokedex = True")

            # If you have the pokedex, you must have delivered the parcel
            # (Oak gives pokedex AFTER you deliver the parcel)
            if not self.world_state.story_flags.get("delivered_parcel"):
                self.world_state.set_flag("delivered_parcel", True)
                logger.info("Auto-detected: delivered_parcel = True (inferred from pokedex)")

        # Detect gym badges from bitfield
        # Bit 0: Boulder (Brock), Bit 1: Cascade (Misty), Bit 2: Thunder (Surge)
        # Bit 3: Rainbow (Erika), Bit 4: Soul (Koga), Bit 5: Marsh (Sabrina)
        # Bit 6: Volcano (Blaine), Bit 7: Earth (Giovanni)
        badges = info.get("badges_bitfield", 0)

        badge_flags = [
            ("beat_brock", 0),
            ("beat_misty", 1),
            ("beat_surge", 2),
            ("beat_erika", 3),
            ("beat_koga", 4),
            ("beat_sabrina", 5),
            ("beat_blaine", 6),
            ("beat_giovanni", 7),
        ]

        for flag_name, bit in badge_flags:
            has_badge = bool(badges & (1 << bit))
            if has_badge and not self.world_state.story_flags.get(flag_name):
                self.world_state.set_flag(flag_name, True)
                logger.info(f"Auto-detected: {flag_name} = True")

        # Auto-complete goals based on newly detected story flags
        self.check_goal_auto_completion()

    # ============= Party Methods =============

    def update_party(self, pokemon_data: list[dict]) -> None:
        """Update party pokemon info."""
        self.party.update_pokemon(pokemon_data)

    def learn_type_matchup(self, enemy_type: str, our_pokemon: str) -> None:
        """Record a type matchup that worked well."""
        self.party.learn_matchup(enemy_type, our_pokemon)

    # ============= Strategy Methods =============

    def add_battle_strategy(self, opponent: str, strategy: str) -> None:
        """Add a strategy for fighting a specific opponent."""
        self.strategies.add_battle_strategy(opponent, strategy)

    def add_area_strategy(self, area: str, strategy: str) -> None:
        """Add a strategy for navigating an area."""
        self.strategies.add_area_strategy(area, strategy)

    def add_grinding_spot(self, location: str, level_range: str, notes: str) -> None:
        """Record a good grinding spot."""
        self.strategies.add_grinding_spot(location, level_range, notes)

    # ============= Failure Recording =============

    def record_failure(self, step: int, context: str, action: str, outcome: str, lesson: str = "") -> None:
        """Record a failure to learn from."""
        self.failures.append({
            "step": step,
            "context": context,
            "action": action,
            "outcome": outcome,
            "lesson": lesson,
        })
        if len(self.failures) > self.max_failures:
            self.failures.pop(0)

    # ============= Questions/Goals =============

    def add_question(self, question: str) -> None:
        """Add an open question to investigate."""
        if question not in self.open_questions:
            self.open_questions.append(question)

    def resolve_question(self, question: str) -> None:
        """Remove a resolved question."""
        if question in self.open_questions:
            self.open_questions.remove(question)

    def set_goal(self, goal: str) -> None:
        """Set the current goal (legacy - for backwards compatibility)."""
        self.current_goal = goal

    # ============= Hierarchical Goals =============

    def set_main_goal(self, description: str, template: str | None = None) -> Goal:
        """
        Set a new main goal, optionally from a template.

        Args:
            description: Goal description (used if no template)
            template: Optional template key (beat_brock, beat_misty, etc.)

        Returns:
            The created Goal
        """
        if template and template in GOAL_TEMPLATES:
            tmpl = GOAL_TEMPLATES[template]
            goal = self.goal_stack.add_goal(
                description=tmpl["description"],
                subgoals=tmpl.get("subgoals"),
                completion_flag=tmpl.get("completion_flag"),
            )
        else:
            goal = self.goal_stack.add_goal(description=description)

        # Also update legacy current_goal for backwards compatibility
        self.current_goal = description
        return goal

    def add_subgoal(self, subgoal: str) -> Goal | None:
        """Add a subgoal to the current active goal."""
        active = self.goal_stack.get_active_goal()
        if not active:
            return None

        import uuid
        sg = Goal(
            id=f"{active.id}_sub{len(active.subgoals)}",
            description=subgoal,
        )
        active.subgoals.append(sg)
        return sg

    def complete_current_subgoal(self) -> bool:
        """Mark the current subgoal as completed."""
        return self.goal_stack.complete_current_subgoal()

    def get_goal_progress(self) -> dict[str, Any]:
        """Get detailed progress on current goal and subgoals."""
        active = self.goal_stack.get_active_goal()
        if not active:
            return {"active": False, "message": "No active goal set"}

        completed = [sg for sg in active.subgoals if sg.status == "completed"]
        pending = [sg for sg in active.subgoals if sg.status == "active"]
        current = active.get_current_subgoal()

        return {
            "active": True,
            "goal": active.description,
            "status": active.status,
            "progress": f"{int(active.progress * 100)}%",
            "current_subgoal": current.description if current else None,
            "completed_subgoals": [sg.description for sg in completed],
            "pending_subgoals": [sg.description for sg in pending],
        }

    def check_goal_auto_completion(self) -> list[str]:
        """
        Check if any goals should be auto-completed based on story flags.

        Returns:
            List of goal descriptions that were auto-completed
        """
        completed = []

        for goal in self.goal_stack.goals:
            if goal.status == "active" and goal.completion_flag:
                if self.world_state.story_flags.get(goal.completion_flag):
                    goal.status = "completed"
                    goal.completed_at = datetime.now().isoformat()
                    goal.progress = 1.0
                    # Mark all subgoals as completed too
                    for sg in goal.subgoals:
                        if sg.status == "active":
                            sg.status = "completed"
                            sg.completed_at = datetime.now().isoformat()
                    completed.append(goal.description)
                    logger.info(f"Auto-completed goal: {goal.description}")

        return completed

    # ============= Notes =============

    def add_note(self, content: str, vital: bool = False, category: str = "general") -> None:
        """Add a timestamped note."""
        # Check for duplicate content
        for note in self.notes:
            if note.content == content:
                return

        self.notes.append(Note(
            content=content,
            vital=vital,
            category=category,
        ))

        # Prune if over limit (keep vital notes)
        if len(self.notes) > self.max_notes:
            self._prune_notes()

    def _prune_notes(self) -> None:
        """Remove oldest non-vital notes to stay under limit."""
        while len(self.notes) > self.max_notes:
            # Find oldest non-vital note
            for i, note in enumerate(self.notes):
                if not note.vital:
                    self.notes.pop(i)
                    break
            else:
                # All notes are vital, remove oldest anyway
                self.notes.pop(0)
                break

    def remove_note(self, content: str) -> None:
        """Remove a note by content."""
        self.notes = [n for n in self.notes if n.content != content]

    # ============= Query Methods =============

    def query(self, query: str, section: str = "all", top_k: int = 5) -> list[str]:
        """
        Search knowledge base and return relevant entries.

        Args:
            query: Search term
            section: Which section to search (all, notes, strategies, failures, questions)
            top_k: Maximum results to return

        Returns:
            List of matching entries as strings
        """
        results = []
        query_lower = query.lower()

        if section in ("all", "notes"):
            for note in self.notes:
                if note.matches_query(query):
                    vital_marker = "[VITAL] " if note.vital else ""
                    results.append(f"{vital_marker}{note.content}")

        if section in ("all", "strategies"):
            for opponent, strategy in self.strategies.battle_strategies.items():
                if query_lower in opponent.lower() or query_lower in strategy.lower():
                    results.append(f"Battle strategy for {opponent}: {strategy}")
            for area, strategy in self.strategies.area_strategies.items():
                if query_lower in area.lower() or query_lower in strategy.lower():
                    results.append(f"Area strategy for {area}: {strategy}")

        if section in ("all", "failures"):
            for failure in self.failures:
                context = failure.get("context", "")
                outcome = failure.get("outcome", "")
                lesson = failure.get("lesson", "")
                if query_lower in context.lower() or query_lower in outcome.lower():
                    results.append(f"Failure: {context} -> {outcome}. Lesson: {lesson}")

        if section in ("all", "questions"):
            for question in self.open_questions:
                if query_lower in question.lower():
                    results.append(f"Open question: {question}")

        if section in ("all", "matchups"):
            for enemy_type, pokemon in self.party.type_matchups.items():
                if query_lower in enemy_type.lower() or query_lower in pokemon.lower():
                    results.append(f"Type matchup: {pokemon} is effective against {enemy_type}")

        return results[:top_k]

    def get_context_for_situation(self, situation: str) -> str:
        """
        Get relevant knowledge for a specific situation.

        Args:
            situation: Current situation (e.g., "battle", "exploring", "gym")

        Returns:
            Formatted relevant knowledge
        """
        lines = []

        if "battle" in situation.lower() or "fight" in situation.lower():
            # Return battle-relevant info
            if self.party.type_matchups:
                lines.append("Type matchups you know:")
                for enemy_type, pokemon in self.party.type_matchups.items():
                    lines.append(f"  - Use {pokemon} against {enemy_type}")
            if self.strategies.battle_strategies:
                lines.append("Battle strategies:")
                for opponent, strategy in list(self.strategies.battle_strategies.items())[-3:]:
                    lines.append(f"  - {opponent}: {strategy}")
            # Recent battle failures
            battle_failures = [f for f in self.failures if "battle" in f.get("context", "").lower()]
            if battle_failures:
                lines.append("Recent battle failures:")
                for f in battle_failures[-2:]:
                    lines.append(f"  - {f.get('context')}: {f.get('lesson', f.get('outcome'))}")

        elif "gym" in situation.lower():
            # Gym-specific strategies
            for opponent, strategy in self.strategies.battle_strategies.items():
                if "gym" in opponent.lower() or any(name in opponent.lower() for name in ["brock", "misty", "surge", "erika", "koga", "sabrina", "blaine", "giovanni"]):
                    lines.append(f"Strategy for {opponent}: {strategy}")

        elif "explore" in situation.lower() or "navigate" in situation.lower():
            # Navigation tips
            current_area = self.world_state.current_location.lower()
            if current_area in self.strategies.area_strategies:
                lines.append(f"Strategy for {current_area}: {self.strategies.area_strategies[current_area]}")
            # Relevant notes
            nav_notes = [n for n in self.notes if n.category == "navigation"]
            for note in nav_notes[-3:]:
                lines.append(f"Note: {note.content}")

        # Always include vital notes
        vital_notes = [n for n in self.notes if n.vital]
        if vital_notes:
            lines.append("Important notes:")
            for note in vital_notes[-5:]:
                lines.append(f"  - {note.content}")

        return "\n".join(lines) if lines else "No specific knowledge for this situation."

    # ============= Formatting for Prompts =============

    def format_for_prompt(self) -> str:
        """
        Format MINIMAL context for prompt. Full KB is queryable via query_knowledge tool.

        Only includes: location, goal (with subgoals), and vital notes.
        Claude should use query_knowledge for more details.
        """
        lines = []

        # Essential current state
        lines.append(f"Location: {self.world_state.current_location}")

        # Hierarchical goal display
        active_goal = self.goal_stack.get_active_goal()
        if active_goal:
            lines.append(f"GOAL: {active_goal.description}")
            current_sub = active_goal.get_current_subgoal()
            if current_sub:
                lines.append(f"  -> Current step: {current_sub.description}")
            # Show progress if there are subgoals
            if active_goal.subgoals:
                completed = [sg for sg in active_goal.subgoals if sg.status == "completed"]
                if completed:
                    lines.append(f"  -> Done: {', '.join(sg.description for sg in completed[-3:])}")
                lines.append(f"  -> Progress: {int(active_goal.progress * 100)}%")
        else:
            # Fall back to legacy current_goal
            lines.append(f"Goal: {self.current_goal}")

        # Story progress (only show achieved flags)
        achieved = [k for k, v in self.world_state.story_flags.items() if v]
        if achieved:
            lines.append(f"Progress: {', '.join(achieved)}")

        # Only vital notes (the really important stuff)
        vital_notes = [n for n in self.notes if n.vital]
        if vital_notes:
            lines.append("")
            lines.append("IMPORTANT (vital notes):")
            for note in vital_notes[-5:]:
                lines.append(f"  - {note.content}")

        # Hint about query tool
        lines.append("")
        lines.append("Use query_knowledge to recall strategies, past failures, or notes.")

        return "\n".join(lines)

    def get_summary_stats(self) -> str:
        """Get a brief summary of what's in the knowledge base."""
        return (
            f"KB contains: {len(self.notes)} notes ({len([n for n in self.notes if n.vital])} vital), "
            f"{len(self.strategies.battle_strategies)} battle strategies, "
            f"{len(self.strategies.area_strategies)} area strategies, "
            f"{len(self.failures)} failures, "
            f"{len(self.open_questions)} open questions"
        )

    # ============= Persistence =============

    def save(self, path: Path) -> None:
        """Save knowledge base to JSON file."""
        data = {
            "world_state": self.world_state.to_dict(),
            "party": self.party.to_dict(),
            "strategies": self.strategies.to_dict(),
            "failures": self.failures,
            "open_questions": self.open_questions,
            "current_goal": self.current_goal,
            "notes": [note.to_dict() for note in self.notes],
            "goal_stack": self.goal_stack.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved knowledge to {path}")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        """Load knowledge base from JSON file."""
        if not path.exists():
            logger.warning(f"Knowledge file not found: {path}, starting fresh")
            return cls()

        with open(path) as f:
            data = json.load(f)

        kb = cls()
        kb.world_state = WorldState.from_dict(data.get("world_state", {}))
        kb.party = PartyKnowledge.from_dict(data.get("party", {}))
        kb.strategies = Strategies.from_dict(data.get("strategies", {}))
        kb.failures = data.get("failures", [])
        kb.open_questions = data.get("open_questions", [])
        kb.current_goal = data.get("current_goal", kb.current_goal)

        # Deserialize notes - handle both old format (strings) and new format (dicts)
        raw_notes = data.get("notes", [])
        kb.notes = []
        for note_data in raw_notes:
            if isinstance(note_data, dict):
                kb.notes.append(Note.from_dict(note_data))
            elif isinstance(note_data, str):
                # Legacy format: plain string
                kb.notes.append(Note(content=note_data))

        # Load goal stack if present
        if "goal_stack" in data:
            kb.goal_stack = GoalStack.from_dict(data["goal_stack"])

        logger.info(f"Loaded knowledge from {path}")
        return kb

    @staticmethod
    def get_run_path(run_id: str) -> Path:
        """Get the file path for a run's knowledge base."""
        return Path("runs") / run_id / "knowledge.json"

    def reset(self) -> None:
        """Reset the knowledge base."""
        self.world_state = WorldState()
        self.party = PartyKnowledge()
        self.strategies = Strategies()
        self.failures = []
        self.open_questions = []
        self.current_goal = "Explore and find wild Pokemon to battle"
        self.notes = []
        self.goal_stack = GoalStack()
