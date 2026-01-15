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
    ENEMY_LEVEL = 0xCFE8  # Enemy Pokemon level in battle
    ENEMY_SPECIES = 0xCFE5  # Enemy Pokemon species ID
    BATTLE_TURN_COUNT = 0xCCD5  # Number of turns in current battle

    # Pokedex (bitfields - each byte = 8 Pokemon)
    POKEDEX_OWNED_START = 0xD2F7  # 19 bytes for 151 Pokemon
    POKEDEX_SEEN_START = 0xD30A  # 19 bytes for 151 Pokemon

    # Player battle Pokemon (active in battle)
    PLAYER_BATTLE_HP = 0xD015  # 2 bytes, current HP of active Pokemon
    PLAYER_BATTLE_HP_MAX = 0xD023  # 2 bytes, max HP of active Pokemon


class GameState:
    """Read game state from emulator memory."""

    def __init__(self, emulator: "EmulatorWrapper") -> None:
        self.emulator = emulator
        self.addr = PokemonRedAddresses()

    def get_player_position(self) -> tuple[int, int]:
        """Get player X, Y position on current map."""
        x = self.emulator.read_memory(self.addr.PLAYER_X)
        y = self.emulator.read_memory(self.addr.PLAYER_Y)
        return x, y

    def get_map_id(self) -> int:
        """Get the current map ID."""
        return self.emulator.read_memory(self.addr.MAP_ID)

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
