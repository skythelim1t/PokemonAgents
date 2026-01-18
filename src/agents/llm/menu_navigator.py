"""Menu navigation framework for Pokemon Red.

Provides robust menu navigation with cursor tracking, menu state detection,
and verified actions. This is the foundation for all menu-related tools.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.pokemon_env import PokemonEnv

logger = logging.getLogger(__name__)

# Memory addresses for menu state
CURRENT_MENU_ITEM = 0xCC26  # Current cursor position (0-indexed)
MAX_MENU_ITEM = 0xCC28      # Number of menu items - 1
TEXT_BOX_ID = 0xCF93        # Menu/dialogue active (>0 = active)
TEXT_DELAY_COUNTER = 0xCFC4 # Text printing delay (0 = done)
OVERWORLD_FLAGS = 0xD730    # Bit 5: don't accept input

# Button indices
BUTTON_A = 0
BUTTON_B = 1
BUTTON_START = 2
BUTTON_SELECT = 3
BUTTON_UP = 4
BUTTON_DOWN = 5
BUTTON_LEFT = 6
BUTTON_RIGHT = 7

# Pokemon Red START menu structure (0-indexed)
START_MENU_OPTIONS = {
    "pokemon": 0,
    "item": 1,
    "player": 2,  # Player name/status
    "save": 3,
    "option": 4,
    "exit": 5,
}


class MenuNavigator:
    """Robust menu navigation with cursor tracking and verification."""

    def __init__(self, env: "PokemonEnv"):
        """
        Initialize the menu navigator.

        Args:
            env: The Pokemon environment
        """
        self.env = env

    @property
    def emulator(self):
        """Get emulator from env dynamically (avoids stale reference issues)."""
        return self.env.emulator

    # === State Detection ===

    def is_menu_open(self) -> bool:
        """Check if any menu/dialogue is active."""
        text_box_id = self.emulator.read_memory(TEXT_BOX_ID)
        return text_box_id > 0

    def get_cursor_position(self) -> int:
        """Get current cursor position (0-indexed)."""
        return self.emulator.read_memory(CURRENT_MENU_ITEM)

    def get_menu_size(self) -> int:
        """Get number of menu options."""
        max_item = self.emulator.read_memory(MAX_MENU_ITEM)
        return max_item + 1

    def is_text_done(self) -> bool:
        """Check if text has finished printing."""
        text_delay = self.emulator.read_memory(TEXT_DELAY_COUNTER)
        return text_delay == 0

    def is_input_accepted(self) -> bool:
        """Check if game is accepting input."""
        flags = self.emulator.read_memory(OVERWORLD_FLAGS)
        # Bit 5 = don't accept input
        return (flags & 0x20) == 0

    def get_menu_state(self) -> dict:
        """Get full menu state for debugging."""
        return {
            "menu_open": self.is_menu_open(),
            "cursor": self.get_cursor_position(),
            "menu_size": self.get_menu_size(),
            "text_done": self.is_text_done(),
            "text_box_id": self.emulator.read_memory(TEXT_BOX_ID),
        }

    # === Wait Functions ===

    def wait_for_menu(self, timeout: int = 60) -> bool:
        """
        Wait until menu is open and ready for input.

        Args:
            timeout: Maximum frames to wait

        Returns:
            True if menu opened, False if timeout
        """
        for _ in range(timeout):
            if self.is_menu_open() and self.is_text_done():
                return True
            self.emulator.tick(1)
        return False

    def wait_for_menu_close(self, timeout: int = 60) -> bool:
        """
        Wait until menu is closed.

        Args:
            timeout: Maximum frames to wait

        Returns:
            True if menu closed, False if timeout
        """
        for _ in range(timeout):
            if not self.is_menu_open():
                return True
            self.emulator.tick(1)
        return False

    def wait_for_text(self, timeout: int = 120) -> bool:
        """
        Wait for text to finish printing.

        Args:
            timeout: Maximum frames to wait

        Returns:
            True if text finished, False if timeout
        """
        for _ in range(timeout):
            if self.is_text_done():
                return True
            self.emulator.tick(1)
        return False

    # === Basic Navigation ===

    def press_button(self, button: int, wait_frames: int = 8) -> None:
        """
        Press a button and wait.

        Args:
            button: Button index (use BUTTON_* constants)
            wait_frames: Frames to wait after press
        """
        self.env.step(button)
        if wait_frames > 0:
            self.emulator.tick(wait_frames)

    def move_cursor_up(self) -> int:
        """Move cursor up and return new position."""
        self.press_button(BUTTON_UP)
        return self.get_cursor_position()

    def move_cursor_down(self) -> int:
        """Move cursor down and return new position."""
        self.press_button(BUTTON_DOWN)
        return self.get_cursor_position()

    def move_cursor_to(self, target: int, max_attempts: int = 10) -> bool:
        """
        Move cursor to target position with verification.

        Args:
            target: Target cursor position (0-indexed)
            max_attempts: Maximum cursor moves before giving up

        Returns:
            True if cursor reached target, False otherwise
        """
        for _ in range(max_attempts):
            current = self.get_cursor_position()
            if current == target:
                return True

            if current < target:
                self.move_cursor_down()
            else:
                self.move_cursor_up()

        return self.get_cursor_position() == target

    def select_current(self) -> bool:
        """Press A to select current option and wait for response."""
        self.press_button(BUTTON_A)
        # Wait a bit for menu to respond
        self.emulator.tick(4)
        return True

    def cancel(self) -> bool:
        """Press B to cancel/go back."""
        self.press_button(BUTTON_B)
        return True

    # === Menu Open/Close ===

    def open_start_menu(self) -> bool:
        """
        Open START menu with verification.

        Returns:
            True if menu opened successfully
        """
        if self.is_menu_open():
            logger.debug("Menu already open")
            return True

        logger.debug("Opening START menu")
        self.press_button(BUTTON_START, wait_frames=0)
        success = self.wait_for_menu(timeout=30)

        if success:
            logger.debug(f"START menu opened, cursor at {self.get_cursor_position()}")
        else:
            logger.warning("Failed to open START menu")

        return success

    def close_menu(self, max_presses: int = 5) -> bool:
        """
        Close all menus by pressing B repeatedly.

        Args:
            max_presses: Maximum B presses

        Returns:
            True if menu closed
        """
        for i in range(max_presses):
            if not self.is_menu_open():
                logger.debug(f"Menu closed after {i} B presses")
                return True
            self.press_button(BUTTON_B, wait_frames=12)

        closed = not self.is_menu_open()
        if not closed:
            logger.warning(f"Menu still open after {max_presses} B presses")
        return closed

    # === High-Level Navigation ===

    def navigate_start_menu(self, option: str) -> bool:
        """
        Open START menu and navigate to specified option.

        Args:
            option: Menu option name ("pokemon", "item", "save", "option", "exit")

        Returns:
            True if successfully navigated to option
        """
        option = option.lower()
        if option not in START_MENU_OPTIONS:
            logger.error(f"Unknown START menu option: {option}")
            return False

        target_index = START_MENU_OPTIONS[option]
        logger.debug(f"Navigating to START menu option '{option}' (index {target_index})")

        if not self.open_start_menu():
            return False

        if not self.move_cursor_to(target_index):
            logger.warning(f"Failed to move cursor to {target_index}")
            return False

        logger.debug(f"Cursor now at {self.get_cursor_position()}, target was {target_index}")
        return True

    def select_start_menu_option(self, option: str) -> bool:
        """
        Navigate to and select a START menu option.

        Args:
            option: Menu option name

        Returns:
            True if successfully selected
        """
        if not self.navigate_start_menu(option):
            return False

        self.select_current()
        return self.wait_for_menu(timeout=30)

    # === Pokemon Menu Navigation ===

    def open_pokemon_menu(self) -> bool:
        """Open the Pokemon party menu."""
        return self.select_start_menu_option("pokemon")

    def navigate_to_pokemon(self, slot: int) -> bool:
        """
        Navigate to a specific Pokemon in the party menu.

        Args:
            slot: Pokemon slot (1-6)

        Returns:
            True if successfully navigated
        """
        if slot < 1 or slot > 6:
            logger.error(f"Invalid Pokemon slot: {slot}")
            return False

        # Pokemon slots are 0-indexed in the menu
        target = slot - 1
        return self.move_cursor_to(target)

    # === Item Menu Navigation ===

    def open_item_menu(self) -> bool:
        """Open the Item/Bag menu."""
        return self.select_start_menu_option("item")

    # === Save Menu ===

    def open_save_menu(self) -> bool:
        """Open the Save menu."""
        return self.select_start_menu_option("save")

    # === Utility ===

    def advance_dialogue(self, max_presses: int = 10) -> bool:
        """
        Advance through dialogue by pressing A.

        Args:
            max_presses: Maximum A presses

        Returns:
            True if dialogue finished (menu closed or different state)
        """
        initial_state = self.emulator.read_memory(TEXT_BOX_ID)

        for i in range(max_presses):
            if not self.is_menu_open():
                logger.debug(f"Dialogue finished after {i} A presses")
                return True

            self.press_button(BUTTON_A, wait_frames=16)

            # Check if state changed significantly
            new_state = self.emulator.read_memory(TEXT_BOX_ID)
            if new_state == 0:
                return True

        return not self.is_menu_open()
