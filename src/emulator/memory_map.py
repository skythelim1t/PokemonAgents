"""Memory addresses for Pokemon Red/Blue."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.emulator.pyboy_wrapper import EmulatorWrapper


@dataclass
class PokemonRedAddresses:
    """RAM addresses for Pokemon Red/Blue."""

    # Player position
    PLAYER_Y = 0xD361
    PLAYER_X = 0xD362
    MAP_ID = 0xD35E

    # Party data
    PARTY_COUNT = 0xD163
    PARTY_SPECIES_LIST = 0xD164  # Species IDs only (6 bytes + terminator)
    PARTY_START = 0xD16B  # Actual party Pokemon data starts here
    PARTY_SIZE = 44  # Bytes per Pokemon in party (0x2C)

    # Badges
    BADGES = 0xD356

    # Game state
    IN_BATTLE = 0xD057
    BATTLE_TYPE = 0xD05A  # 0 = wild, 1 = trainer

    # Money (3 bytes, BCD encoded)
    MONEY_START = 0xD347

    # Items
    ITEM_COUNT = 0xD31D
    ITEMS_START = 0xD31E

    # Pokemon stats in party (offsets from party Pokemon start)
    # See: https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_in_Generation_I
    POKEMON_SPECIES = 0x00
    POKEMON_HP_CURRENT = 0x01  # 2 bytes, big-endian
    POKEMON_BOX_LEVEL = 0x03  # Level when deposited (1 byte)
    POKEMON_STATUS = 0x04  # Status condition (1 byte)
    POKEMON_TYPE1 = 0x05
    POKEMON_TYPE2 = 0x06
    POKEMON_MOVES = 0x08  # 4 bytes (4 moves)
    POKEMON_LEVEL = 0x21  # Actual level (1 byte) = 33 decimal
    POKEMON_HP_MAX = 0x22  # 2 bytes, big-endian = 34 decimal
    POKEMON_ATTACK = 0x24  # 2 bytes = 36 decimal
    POKEMON_DEFENSE = 0x26  # 2 bytes = 38 decimal
    POKEMON_SPEED = 0x28  # 2 bytes = 40 decimal
    POKEMON_SPECIAL = 0x2A  # 2 bytes = 42 decimal

    # Event flags
    EVENT_FLAGS_START = 0xD747
    EVENT_FLAGS_END = 0xD886

    # Battle data
    ENEMY_HP = 0xCFE6  # 2 bytes, current enemy HP in battle
    ENEMY_HP_MAX = 0xCFF4  # 2 bytes, max enemy HP
    ENEMY_LEVEL = 0xCFF3  # Enemy Pokemon level in battle (in enemy mon struct)
    ENEMY_SPECIES = 0xCFE5  # Enemy Pokemon species ID
    BATTLE_TURN_COUNT = 0xCCD5  # Number of turns in current battle

    # Battle phase detection
    BATTLE_TYPE = 0xD057  # 0=not in battle, 1=wild, 2=trainer
    LOW_HEALTH_ALARM = 0xCFE7  # Low health alarm disable flag
    ACTION_SELECTION_FLAG = 0xCC2B  # Battle action selection state

    # Pokedex (bitfields - each byte = 8 Pokemon)
    POKEDEX_OWNED_START = 0xD2F7  # 19 bytes for 151 Pokemon
    POKEDEX_SEEN_START = 0xD30A  # 19 bytes for 151 Pokemon

    # Player battle Pokemon (active in battle)
    PLAYER_BATTLE_SPECIES = 0xD014  # Species ID of active Pokemon
    PLAYER_BATTLE_HP = 0xD015  # 2 bytes, current HP of active Pokemon
    PLAYER_BATTLE_MOVES = 0xD01C  # 4 bytes, move IDs (1-165)
    PLAYER_BATTLE_LEVEL = 0xD022  # Level of active Pokemon
    PLAYER_BATTLE_HP_MAX = 0xD023  # 2 bytes, max HP of active Pokemon

    # Menu/Text state
    CURRENT_MENU_ITEM = 0xCC26  # Current selected menu item (0-indexed)
    MAX_MENU_ITEM = 0xCC28  # Number of menu items - 1
    TEXT_BOX_ID = 0xCF93  # Type of text box (0 = none)
    MENU_JOYPAD_POLLING = 0xCC24  # Menu state flag
    YES_NO_CURSOR = 0xCC26  # Cursor position in yes/no box (0=yes, 1=no)

    # Battle menu state
    BATTLE_MENU_CURSOR = 0xCC2B  # Main battle menu cursor (FIGHT/BAG/POKEMON/RUN)
    MOVE_MENU_CURSOR = 0xCC2E  # Move selection cursor position (0-3)
    BATTLE_MENU_TYPE = 0xD127  # Current battle menu: 0=main, other=submenu
    WHOSE_TURN = 0xD062  # 0=player turn, 1=enemy turn
    ACTION_SELECTED = 0xCCDC  # Battle action already selected flag
    FIGHT_MENU_OPEN = 0xCC35  # Non-zero when FIGHT/move menu is open
    POKEMON_MENU_IN_BATTLE = 0xD07D  # Non-zero when Pokemon menu open in battle

    # Input detection
    JOYPAD_SIMULATION = 0xCCD3  # Simulated joypad (0 = accepting input)
    TEXT_DELAY_COUNTER = 0xCFC4  # Text printing delay (0 = text done)
    WALK_COUNTER = 0xCFC5  # Walking animation counter
    OVERWORLD_FLAGS = 0xD730  # Bit 0: don't accept START, Bit 5: don't accept input

    # Environment detection (indoor/outdoor/cave)
    TILESET_ID = 0xD527  # Current tileset ID (determines environment type)
    NUMBER_OF_WARPS = 0xD4E1  # Count of warps/exits on current map
    WARP_ENTRIES = 0xD4E2  # Warp data start (4 bytes per warp: Y, X, dest_warp, dest_map)

    # NOTE: Player sprite screen position is now obtained via PyBoy's sprite API
    # instead of memory addresses, which were unreliable (returned wrong values)


# Tileset classifications for environment detection
# Based on Pokemon Red disassembly: https://github.com/pret/pokered
OUTDOOR_TILESETS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
INDOOR_TILESETS = {12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 25, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 67}
CAVE_TILESETS = {15, 40, 46}


class GameState:
    """Read game state from emulator memory."""

    def __init__(self, emulator: "EmulatorWrapper") -> None:
        self.emulator = emulator
        self.addr = PokemonRedAddresses()

    def get_player_position(self) -> tuple[int, int]:
        """Get player X, Y position on current map (world coordinates)."""
        x = self.emulator.read_memory(self.addr.PLAYER_X)
        y = self.emulator.read_memory(self.addr.PLAYER_Y)
        return x, y

    def get_player_screen_position(self) -> tuple[int, int]:
        """
        Get player's position on screen as game tile coordinates.

        Returns (x, y) where x is 0-9 and y is 0-8 in the 10x9 game tile grid.
        Uses PyBoy's sprite API - sprite coordinates are already screen pixels
        (PyBoy handles OAM offset internally).
        """
        try:
            # Use PyBoy sprite API - sprite 0 is the player
            pixel_x, pixel_y = self.emulator.get_sprite_position(0)

            # Use sprite center for tile calculation (8x8 sprite, add 4 to get center)
            # Then divide by 16 (tile size) to get tile coordinates
            tile_x = (pixel_x + 4) // 16
            tile_y = (pixel_y + 4) // 16

            # Clamp to valid screen bounds (0-9 for x, 0-8 for y)
            tile_x = max(0, min(tile_x, 9))
            tile_y = max(0, min(tile_y, 8))

            return tile_x, tile_y
        except Exception:
            # Fallback to center if sprite API fails
            return 4, 4

    def get_map_id(self) -> int:
        """Get the current map ID."""
        return self.emulator.read_memory(self.addr.MAP_ID)

    def get_environment_type(self) -> str:
        """
        Get current environment type based on tileset.

        Returns:
            'outdoors', 'indoors', or 'cave'
        """
        tileset = self.emulator.read_memory(self.addr.TILESET_ID)
        if tileset in OUTDOOR_TILESETS:
            return "outdoors"
        elif tileset in CAVE_TILESETS:
            return "cave"
        else:
            return "indoors"  # Default to indoors for unknown tilesets

    def get_warp_locations(self) -> list[dict]:
        """
        Get all warp/exit locations on the current map.

        Returns:
            List of dicts with x, y coordinates and destination map ID
        """
        num_warps = self.emulator.read_memory(self.addr.NUMBER_OF_WARPS)
        warps = []
        for i in range(min(num_warps, 32)):  # Cap at 32 to avoid reading garbage
            offset = self.addr.WARP_ENTRIES + (i * 4)
            y = self.emulator.read_memory(offset)
            x = self.emulator.read_memory(offset + 1)
            dest_warp = self.emulator.read_memory(offset + 2)
            dest_map = self.emulator.read_memory(offset + 3)
            warps.append({
                "x": x,
                "y": y,
                "dest_map_id": dest_map,
                "dest_warp_id": dest_warp,
            })
        return warps

    def get_party_count(self) -> int:
        """Get the number of Pokemon in the party."""
        return self.emulator.read_memory(self.addr.PARTY_COUNT)

    def get_badges(self) -> int:
        """Get badges as a bitfield (8 bits for 8 badges)."""
        return self.emulator.read_memory(self.addr.BADGES)

    def get_badge_count(self) -> int:
        """Get the number of badges earned."""
        badges = self.get_badges()
        return bin(badges).count("1")

    def is_in_battle(self) -> bool:
        """Check if currently in a battle."""
        return self.emulator.read_memory(self.addr.IN_BATTLE) != 0

    def is_battle_menu_ready(self) -> bool:
        """
        Check if the battle menu is ready for action selection.

        Returns False during battle intro ("A wild X appeared!") and
        during move animations/text. Returns True when player can
        select FIGHT/BAG/POKEMON/RUN.
        """
        if not self.is_in_battle():
            return False

        # Check if we're in a text box or dialogue (battle intro, move text, etc.)
        text_box = self.emulator.read_memory(self.addr.TEXT_BOX_ID)
        if text_box > 0:
            return False  # Still showing text, not at menu

        # Check if menu is in battle action selection mode
        # When at the main battle menu, max_menu_item should be 3 (4 options: FIGHT/BAG/POKEMON/RUN)
        max_menu = self.emulator.read_memory(self.addr.MAX_MENU_ITEM)
        if max_menu == 3:
            return True  # At main battle menu with 4 options

        # Also check for FIGHT menu (4 moves)
        if max_menu >= 0 and max_menu <= 3:
            menu_active = self.emulator.read_memory(self.addr.MENU_JOYPAD_POLLING)
            if menu_active > 0:
                return True  # Some battle menu is active

        return False

    def get_party_levels(self) -> list[int]:
        """Get the levels of all Pokemon in the party."""
        count = self.get_party_count()
        levels = []
        for i in range(count):
            offset = self.addr.PARTY_START + (i * self.addr.PARTY_SIZE)
            level = self.emulator.read_memory(offset + self.addr.POKEMON_LEVEL)
            levels.append(level)
        return levels

    def get_party_hp(self) -> list[tuple[int, int]]:
        """Get (current_hp, max_hp) for each Pokemon in party."""
        count = self.get_party_count()
        hp_list = []
        for i in range(count):
            offset = self.addr.PARTY_START + (i * self.addr.PARTY_SIZE)
            # HP is stored as 2 bytes, big endian
            current_hp = (
                self.emulator.read_memory(offset + self.addr.POKEMON_HP_CURRENT) << 8
            ) + self.emulator.read_memory(offset + self.addr.POKEMON_HP_CURRENT + 1)
            max_hp = (
                self.emulator.read_memory(offset + self.addr.POKEMON_HP_MAX) << 8
            ) + self.emulator.read_memory(offset + self.addr.POKEMON_HP_MAX + 1)
            hp_list.append((current_hp, max_hp))
        return hp_list

    def get_total_level(self) -> int:
        """Get the sum of all party Pokemon levels."""
        return sum(self.get_party_levels())

    def get_money(self) -> int:
        """Get the player's money (BCD decoded)."""
        # Money is stored as 3 bytes in BCD format
        bytes_data = self.emulator.read_memory_range(self.addr.MONEY_START, 3)
        money = 0
        for byte in bytes_data:
            money = money * 100 + ((byte >> 4) * 10) + (byte & 0x0F)
        return money

    def get_enemy_hp(self) -> tuple[int, int]:
        """Get (current_hp, max_hp) of enemy Pokemon in battle."""
        current = (
            self.emulator.read_memory(self.addr.ENEMY_HP) << 8
        ) + self.emulator.read_memory(self.addr.ENEMY_HP + 1)
        max_hp = (
            self.emulator.read_memory(self.addr.ENEMY_HP_MAX) << 8
        ) + self.emulator.read_memory(self.addr.ENEMY_HP_MAX + 1)
        return current, max_hp

    def get_enemy_level(self) -> int:
        """Get the level of the enemy Pokemon in battle."""
        return self.emulator.read_memory(self.addr.ENEMY_LEVEL)

    def get_pokedex_owned_count(self) -> int:
        """Get the number of Pokemon owned (caught) in the Pokedex."""
        count = 0
        for i in range(19):  # 19 bytes cover 152 Pokemon (151 + padding)
            byte = self.emulator.read_memory(self.addr.POKEDEX_OWNED_START + i)
            count += bin(byte).count("1")
        return count

    def get_pokedex_seen_count(self) -> int:
        """Get the number of Pokemon seen in the Pokedex."""
        count = 0
        for i in range(19):
            byte = self.emulator.read_memory(self.addr.POKEDEX_SEEN_START + i)
            count += bin(byte).count("1")
        return count

    def get_total_party_hp(self) -> tuple[int, int]:
        """Get (current_total_hp, max_total_hp) across all party Pokemon."""
        hp_list = self.get_party_hp()
        current = sum(hp for hp, _ in hp_list)
        max_hp = sum(max_hp for _, max_hp in hp_list)
        return current, max_hp

    def get_party_alive_count(self) -> int:
        """Get number of party Pokemon with HP > 0."""
        return sum(1 for hp, _ in self.get_party_hp() if hp > 0)

    def get_menu_state(self) -> dict[str, int]:
        """Get current menu/dialogue state."""
        return {
            "text_box_id": self.emulator.read_memory(self.addr.TEXT_BOX_ID),
            "current_menu_item": self.emulator.read_memory(self.addr.CURRENT_MENU_ITEM),
            "max_menu_item": self.emulator.read_memory(self.addr.MAX_MENU_ITEM),
            "menu_active": self.emulator.read_memory(self.addr.MENU_JOYPAD_POLLING),
        }

    def is_in_menu_or_dialogue(self) -> bool:
        """Check if a menu or dialogue box is active."""
        menu = self.get_menu_state()
        # Only use text_box_id - MENU_JOYPAD_POLLING can have stale non-zero values
        # when walking around, giving false positives
        return menu["text_box_id"] > 0

    def has_menu_choice(self) -> bool:
        """Check if there's a menu with multiple choices."""
        menu = self.get_menu_state()
        # Only count as having choices if there's actually a text box visible
        # max_menu_item can retain stale values from previous menus
        return menu["text_box_id"] > 0 and menu["max_menu_item"] > 0

    def is_waiting_for_input(self) -> bool:
        """
        Check if the game is waiting for player input.

        Returns True when input is needed (menus, dialogue, battle decisions).
        Returns False during animations, text scrolling, walking, etc.
        """
        # Check if joypad simulation is active (cutscenes, forced movement)
        joypad_sim = self.emulator.read_memory(self.addr.JOYPAD_SIMULATION)
        if joypad_sim != 0:
            return False  # Game is simulating input, don't need player input

        # Check overworld flags - bit 5 means input disabled
        overworld_flags = self.emulator.read_memory(self.addr.OVERWORLD_FLAGS)
        if overworld_flags & 0x20:  # Bit 5 set
            return False  # Input disabled

        # Check if walking animation is in progress
        walk_counter = self.emulator.read_memory(self.addr.WALK_COUNTER)
        if walk_counter > 0:
            return False  # Still walking

        # Check if text is still printing
        text_delay = self.emulator.read_memory(self.addr.TEXT_DELAY_COUNTER)
        if text_delay > 0:
            return False  # Text still printing

        # Check if menu is active or text box is showing (these need input)
        menu_state = self.get_menu_state()
        if menu_state["menu_active"] > 0 or menu_state["text_box_id"] > 0:
            return True  # Menu or text box waiting for input

        # Default: assume input is needed (safer to ask than miss input)
        return True

    def get_active_pokemon_info(self) -> dict[str, any]:
        """Get info about the player's active Pokemon in battle."""
        species_id = self.emulator.read_memory(self.addr.PLAYER_BATTLE_SPECIES)
        level = self.emulator.read_memory(self.addr.PLAYER_BATTLE_LEVEL)

        current_hp = (
            self.emulator.read_memory(self.addr.PLAYER_BATTLE_HP) << 8
        ) + self.emulator.read_memory(self.addr.PLAYER_BATTLE_HP + 1)
        max_hp = (
            self.emulator.read_memory(self.addr.PLAYER_BATTLE_HP_MAX) << 8
        ) + self.emulator.read_memory(self.addr.PLAYER_BATTLE_HP_MAX + 1)

        # Read the 4 move IDs
        moves = []
        for i in range(4):
            move_id = self.emulator.read_memory(self.addr.PLAYER_BATTLE_MOVES + i)
            if move_id > 0:  # 0 means no move in that slot
                moves.append(move_id)

        return {
            "species_id": species_id,
            "species_name": get_pokemon_name(species_id),
            "level": level,
            "hp": current_hp,
            "max_hp": max_hp,
            "moves": [get_move_name(m) for m in moves],
            "move_ids": moves,
        }

    def get_enemy_pokemon_info(self) -> dict[str, any]:
        """Get info about the enemy Pokemon in battle."""
        species_id = self.emulator.read_memory(self.addr.ENEMY_SPECIES)
        level = self.emulator.read_memory(self.addr.ENEMY_LEVEL)
        current_hp, max_hp = self.get_enemy_hp()

        return {
            "species_id": species_id,
            "species_name": get_pokemon_name(species_id),
            "level": level,
            "hp": current_hp,
            "max_hp": max_hp,
        }

    def get_battle_turn_count(self) -> int:
        """Get the number of turns in the current battle."""
        return self.emulator.read_memory(self.addr.BATTLE_TURN_COUNT)

    def is_trainer_battle(self) -> bool:
        """Check if current battle is a trainer battle (not wild)."""
        battle_type = self.emulator.read_memory(self.addr.BATTLE_TYPE)
        return battle_type == 2  # 0=not in battle, 1=wild, 2=trainer

    def get_item_count(self) -> int:
        """Get the number of items in the bag."""
        return self.emulator.read_memory(self.addr.ITEM_COUNT)

    def get_event_flags(self) -> set[int]:
        """
        Get all set event flags as a set of bit indices.

        Event flags track story progress (got starter, beat gym, etc.)
        Returns set of indices (0-1023) for all set flags.
        """
        flags = set()
        flag_range = self.addr.EVENT_FLAGS_END - self.addr.EVENT_FLAGS_START
        for byte_idx in range(flag_range):
            byte_val = self.emulator.read_memory(self.addr.EVENT_FLAGS_START + byte_idx)
            for bit in range(8):
                if byte_val & (1 << bit):
                    flags.add(byte_idx * 8 + bit)
        return flags


# Pokemon name lookup (Gen 1 internal IDs are weird, not 1-151 order)
# This maps the internal species ID to name
POKEMON_NAMES = {
    0: "???",
    1: "RHYDON", 2: "KANGASKHAN", 3: "NIDORAN♂", 4: "CLEFAIRY", 5: "SPEAROW",
    6: "VOLTORB", 7: "NIDOKING", 8: "SLOWBRO", 9: "IVYSAUR", 10: "EXEGGUTOR",
    11: "LICKITUNG", 12: "EXEGGCUTE", 13: "GRIMER", 14: "GENGAR", 15: "NIDORAN♀",
    16: "NIDOQUEEN", 17: "CUBONE", 18: "RHYHORN", 19: "LAPRAS", 20: "ARCANINE",
    21: "MEW", 22: "GYARADOS", 23: "SHELLDER", 24: "TENTACOOL", 25: "GASTLY",
    26: "SCYTHER", 27: "STARYU", 28: "BLASTOISE", 29: "PINSIR", 30: "TANGELA",
    33: "GROWLITHE", 34: "ONIX", 35: "FEAROW", 36: "PIDGEY", 37: "SLOWPOKE",
    38: "KADABRA", 39: "GRAVELER", 40: "CHANSEY", 41: "MACHOKE", 42: "MR. MIME",
    43: "HITMONLEE", 44: "HITMONCHAN", 45: "ARBOK", 46: "PARASECT", 47: "PSYDUCK",
    48: "DROWZEE", 49: "GOLEM", 51: "MAGMAR", 53: "ELECTABUZZ", 54: "MAGNETON",
    55: "KOFFING", 57: "MANKEY", 58: "SEEL", 59: "DIGLETT", 60: "TAUROS",
    64: "FARFETCH'D", 65: "VENONAT", 66: "DRAGONITE", 70: "DODUO", 71: "POLIWAG",
    72: "JYNX", 73: "MOLTRES", 74: "ARTICUNO", 75: "ZAPDOS", 76: "DITTO",
    77: "MEOWTH", 78: "KRABBY", 82: "VULPIX", 83: "NINETALES", 84: "PIKACHU",
    85: "RAICHU", 88: "DRATINI", 89: "DRAGONAIR", 90: "KABUTO", 91: "KABUTOPS",
    92: "HORSEA", 93: "SEADRA", 96: "SANDSHREW", 97: "SANDSLASH", 98: "OMANYTE",
    99: "OMASTAR", 100: "JIGGLYPUFF", 101: "WIGGLYTUFF", 102: "EEVEE",
    103: "FLAREON", 104: "JOLTEON", 105: "VAPOREON", 106: "MACHOP", 107: "ZUBAT",
    108: "EKANS", 109: "PARAS", 110: "POLIWHIRL", 111: "POLIWRATH", 112: "WEEDLE",
    113: "KAKUNA", 114: "BEEDRILL", 116: "DODRIO", 117: "PRIMEAPE", 118: "DUGTRIO",
    119: "VENOMOTH", 120: "DEWGONG", 123: "CATERPIE", 124: "METAPOD",
    125: "BUTTERFREE", 126: "MACHAMP", 128: "GOLDUCK", 129: "HYPNO", 130: "GOLBAT",
    131: "MEWTWO", 132: "SNORLAX", 133: "MAGIKARP", 136: "MUK", 138: "KINGLER",
    139: "CLOYSTER", 141: "ELECTRODE", 142: "CLEFABLE", 143: "WEEZING",
    144: "PERSIAN", 145: "MAROWAK", 147: "HAUNTER", 148: "ABRA", 149: "ALAKAZAM",
    150: "PIDGEOTTO", 151: "PIDGEOT", 152: "STARMIE", 153: "BULBASAUR",
    154: "VENUSAUR", 155: "TENTACRUEL", 157: "GOLDEEN", 158: "SEAKING",
    163: "PONYTA", 164: "RAPIDASH", 165: "RATTATA", 166: "RATICATE",
    167: "NIDORINO", 168: "NIDORINA", 169: "GEODUDE", 170: "PORYGON",
    171: "AERODACTYL", 173: "MAGNEMITE", 176: "CHARMANDER", 177: "SQUIRTLE",
    178: "CHARMELEON", 179: "WARTORTLE", 180: "CHARIZARD", 185: "ODDISH",
    186: "GLOOM", 187: "VILEPLUME", 188: "BELLSPROUT", 189: "WEEPINBELL",
    190: "VICTREEBEL",
}


def get_pokemon_name(species_id: int) -> str:
    """Get Pokemon name from species ID."""
    return POKEMON_NAMES.get(species_id, f"Pokemon#{species_id}")


# Move names (Gen 1, ID 1-165)
MOVE_NAMES = {
    0: "---",
    1: "POUND", 2: "KARATE CHOP", 3: "DOUBLESLAP", 4: "COMET PUNCH", 5: "MEGA PUNCH",
    6: "PAY DAY", 7: "FIRE PUNCH", 8: "ICE PUNCH", 9: "THUNDERPUNCH", 10: "SCRATCH",
    11: "VICEGRIP", 12: "GUILLOTINE", 13: "RAZOR WIND", 14: "SWORDS DANCE", 15: "CUT",
    16: "GUST", 17: "WING ATTACK", 18: "WHIRLWIND", 19: "FLY", 20: "BIND",
    21: "SLAM", 22: "VINE WHIP", 23: "STOMP", 24: "DOUBLE KICK", 25: "MEGA KICK",
    26: "JUMP KICK", 27: "ROLLING KICK", 28: "SAND-ATTACK", 29: "HEADBUTT", 30: "HORN ATTACK",
    31: "FURY ATTACK", 32: "HORN DRILL", 33: "TACKLE", 34: "BODY SLAM", 35: "WRAP",
    36: "TAKE DOWN", 37: "THRASH", 38: "DOUBLE-EDGE", 39: "TAIL WHIP", 40: "POISON STING",
    41: "TWINEEDLE", 42: "PIN MISSILE", 43: "LEER", 44: "BITE", 45: "GROWL",
    46: "ROAR", 47: "SING", 48: "SUPERSONIC", 49: "SONICBOOM", 50: "DISABLE",
    51: "ACID", 52: "EMBER", 53: "FLAMETHROWER", 54: "MIST", 55: "WATER GUN",
    56: "HYDRO PUMP", 57: "SURF", 58: "ICE BEAM", 59: "BLIZZARD", 60: "PSYBEAM",
    61: "BUBBLEBEAM", 62: "AURORA BEAM", 63: "HYPER BEAM", 64: "PECK", 65: "DRILL PECK",
    66: "SUBMISSION", 67: "LOW KICK", 68: "COUNTER", 69: "SEISMIC TOSS", 70: "STRENGTH",
    71: "ABSORB", 72: "MEGA DRAIN", 73: "LEECH SEED", 74: "GROWTH", 75: "RAZOR LEAF",
    76: "SOLARBEAM", 77: "POISONPOWDER", 78: "STUN SPORE", 79: "SLEEP POWDER", 80: "PETAL DANCE",
    81: "STRING SHOT", 82: "DRAGON RAGE", 83: "FIRE SPIN", 84: "THUNDERSHOCK", 85: "THUNDERBOLT",
    86: "THUNDER WAVE", 87: "THUNDER", 88: "ROCK THROW", 89: "EARTHQUAKE", 90: "FISSURE",
    91: "DIG", 92: "TOXIC", 93: "CONFUSION", 94: "PSYCHIC", 95: "HYPNOSIS",
    96: "MEDITATE", 97: "AGILITY", 98: "QUICK ATTACK", 99: "RAGE", 100: "TELEPORT",
    101: "NIGHT SHADE", 102: "MIMIC", 103: "SCREECH", 104: "DOUBLE TEAM", 105: "RECOVER",
    106: "HARDEN", 107: "MINIMIZE", 108: "SMOKESCREEN", 109: "CONFUSE RAY", 110: "WITHDRAW",
    111: "DEFENSE CURL", 112: "BARRIER", 113: "LIGHT SCREEN", 114: "HAZE", 115: "REFLECT",
    116: "FOCUS ENERGY", 117: "BIDE", 118: "METRONOME", 119: "MIRROR MOVE", 120: "SELFDESTRUCT",
    121: "EGG BOMB", 122: "LICK", 123: "SMOG", 124: "SLUDGE", 125: "BONE CLUB",
    126: "FIRE BLAST", 127: "WATERFALL", 128: "CLAMP", 129: "SWIFT", 130: "SKULL BASH",
    131: "SPIKE CANNON", 132: "CONSTRICT", 133: "AMNESIA", 134: "KINESIS", 135: "SOFTBOILED",
    136: "HI JUMP KICK", 137: "GLARE", 138: "DREAM EATER", 139: "POISON GAS", 140: "BARRAGE",
    141: "LEECH LIFE", 142: "LOVELY KISS", 143: "SKY ATTACK", 144: "TRANSFORM", 145: "BUBBLE",
    146: "DIZZY PUNCH", 147: "SPORE", 148: "FLASH", 149: "PSYWAVE", 150: "SPLASH",
    151: "ACID ARMOR", 152: "CRABHAMMER", 153: "EXPLOSION", 154: "FURY SWIPES", 155: "BONEMERANG",
    156: "REST", 157: "ROCK SLIDE", 158: "HYPER FANG", 159: "SHARPEN", 160: "CONVERSION",
    161: "TRI ATTACK", 162: "SUPER FANG", 163: "SLASH", 164: "SUBSTITUTE", 165: "STRUGGLE",
}


def get_move_name(move_id: int) -> str:
    """Get move name from move ID."""
    return MOVE_NAMES.get(move_id, f"Move#{move_id}")
