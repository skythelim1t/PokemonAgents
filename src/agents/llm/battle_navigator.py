"""Battle menu navigation framework for Pokemon Red.

Provides robust battle menu navigation with cursor tracking, menu state detection,
and verified actions. Handles the main battle menu, move selection, Pokemon switching,
and item usage during battle.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.pokemon_env import PokemonEnv

logger = logging.getLogger(__name__)

# Memory addresses for battle menu state
BATTLE_MENU_CURSOR = 0xCC2B  # Main battle menu cursor position
MOVE_MENU_CURSOR = 0xCC2E  # Move selection cursor (0-3 in 2x2 grid)
CURRENT_MENU_ITEM = 0xCC26  # Generic menu cursor
MAX_MENU_ITEM = 0xCC28  # Number of menu items - 1
TEXT_BOX_ID = 0xCF93  # Text box active (>0 = active)
TEXT_DELAY_COUNTER = 0xCFC4  # Text printing delay (0 = done)
IN_BATTLE = 0xD057  # Battle flag (non-zero = in battle)
WHOSE_TURN = 0xD062  # 0 = player turn
FIGHT_MENU_OPEN = 0xCC35  # Non-zero when move selection is open

# Button indices
BUTTON_A = 0
BUTTON_B = 1
BUTTON_UP = 4
BUTTON_DOWN = 5
BUTTON_LEFT = 6
BUTTON_RIGHT = 7

# Battle menu options (2x2 grid layout)
# [FIGHT]  [BAG]
# [POKEMON][RUN]
BATTLE_MENU_OPTIONS = {
    "fight": 0,
    "bag": 1,
    "pokemon": 2,
    "run": 3,
}

# Move slots in 2x2 grid
# [Move 1] [Move 2]
# [Move 3] [Move 4]
MOVE_POSITIONS = {
    1: (0, 0),  # top-left
    2: (1, 0),  # top-right
    3: (0, 1),  # bottom-left
    4: (1, 1),  # bottom-right
}


class BattleMenuNavigator:
    """Robust battle menu navigation with cursor tracking and verification."""

    def __init__(self, env: "PokemonEnv"):
        """
        Initialize the battle menu navigator.

        Args:
            env: The Pokemon environment
        """
        self.env = env
        # Optional callback for rendering during waits (set by spectator)
        self._render_callback: callable | None = None

    def set_render_callback(self, callback: callable) -> None:
        """Set a callback to render the screen during waits."""
        self._render_callback = callback

    def _maybe_render(self) -> None:
        """Call the render callback if set."""
        if self._render_callback:
            try:
                self._render_callback()
            except Exception:
                pass  # Don't let render errors break battle logic

    @property
    def emulator(self):
        """Get emulator from env dynamically (avoids stale reference issues)."""
        return self.env.emulator

    # === State Detection ===

    def is_in_battle(self) -> bool:
        """Check if currently in a battle."""
        return self.emulator.read_memory(IN_BATTLE) != 0

    def is_player_turn(self) -> bool:
        """Check if it's the player's turn."""
        if not self.is_in_battle():
            return False
        # During player turn, we should be able to input
        return self.is_text_done()

    def is_text_done(self) -> bool:
        """Check if text has finished printing."""
        text_delay = self.emulator.read_memory(TEXT_DELAY_COUNTER)
        return text_delay == 0

    def is_menu_open(self) -> bool:
        """Check if any menu/dialogue is active."""
        text_box_id = self.emulator.read_memory(TEXT_BOX_ID)
        return text_box_id > 0

    def is_battle_menu_ready(self) -> bool:
        """
        Check if the main battle menu is ready for input (FIGHT/BAG/POKEMON/RUN).

        In battle, text_box_id > 0 means text is displaying (NOT ready).
        Ready when text is done AND we have exactly 4 menu options AND menu accepts input.
        """
        if not self.is_in_battle():
            return False

        # If text box is showing, we're not at the menu yet
        text_box = self.emulator.read_memory(TEXT_BOX_ID)
        if text_box > 0:
            return False

        # Text delay must be 0 (text finished printing)
        text_delay = self.emulator.read_memory(TEXT_DELAY_COUNTER)
        if text_delay > 0:
            return False

        # Check if we have exactly 4 options (FIGHT/BAG/POKEMON/RUN)
        max_menu = self.emulator.read_memory(MAX_MENU_ITEM)
        if max_menu != 3:
            return False

        # Check if menu is accepting input (wMenuJoypad > 0)
        # This helps avoid false positives during transitions
        menu_joypad = self.emulator.read_memory(0xCC24)
        return menu_joypad > 0

    def is_fight_menu_open(self) -> bool:
        """Check if the move selection menu is open."""
        # Only rely on the FIGHT_MENU_OPEN flag - the fallback logic was causing
        # false positives when at the main battle menu (0xCC24 is set for any menu)
        fight_menu = self.emulator.read_memory(FIGHT_MENU_OPEN)
        return fight_menu != 0

    def get_battle_menu_cursor(self) -> int:
        """Get cursor position in main battle menu (0-3)."""
        return self.emulator.read_memory(BATTLE_MENU_CURSOR)

    def get_move_cursor(self) -> int:
        """Get cursor position in move menu (0-3)."""
        return self.emulator.read_memory(MOVE_MENU_CURSOR)

    def get_menu_cursor(self) -> int:
        """Get generic menu cursor position."""
        return self.emulator.read_memory(CURRENT_MENU_ITEM)

    def get_menu_size(self) -> int:
        """Get number of menu options."""
        max_item = self.emulator.read_memory(MAX_MENU_ITEM)
        return max_item + 1

    def get_battle_state(self) -> dict:
        """Get full battle menu state for debugging."""
        return {
            "in_battle": self.is_in_battle(),
            "battle_menu_ready": self.is_battle_menu_ready(),
            "text_box_id": self.emulator.read_memory(TEXT_BOX_ID),
            "text_delay": self.emulator.read_memory(TEXT_DELAY_COUNTER),
            "max_menu_item": self.emulator.read_memory(MAX_MENU_ITEM),
            "menu_joypad": self.emulator.read_memory(0xCC24),
            "battle_cursor": self.get_battle_menu_cursor(),
            "move_cursor": self.get_move_cursor(),
        }

    # === Wait Functions ===

    def wait_for_battle_menu(self, timeout: int = 300) -> bool:
        """
        Wait until the main battle menu is ready for input.

        Works for both:
        - First turn (advances through "Wild X appeared!", "Go POKEMON!")
        - Subsequent turns (waits for enemy turn to finish, advances result text)

        Args:
            timeout: Maximum frames to wait

        Returns:
            True if battle menu is ready, False if timeout
        """
        if not self.is_in_battle():
            logger.debug("Not in battle")
            return False

        # Check if already at menu (common case for subsequent turns)
        if self.is_battle_menu_ready():
            logger.debug("Already at battle menu")
            return True

        # Wait and advance through any text/animations until menu is ready
        a_presses = 0
        frames_without_progress = 0

        for frame in range(timeout):
            # Check if menu is ready
            if self.is_battle_menu_ready():
                logger.debug(f"Battle menu ready after {frame} frames, {a_presses} A presses")
                return True

            # Check if battle ended
            if not self.is_in_battle():
                logger.debug(f"Battle ended at frame {frame}")
                return False

            # Get current state
            text_box = self.emulator.read_memory(TEXT_BOX_ID)
            text_delay = self.emulator.read_memory(TEXT_DELAY_COUNTER)
            max_menu = self.emulator.read_memory(MAX_MENU_ITEM)
            menu_joypad = self.emulator.read_memory(0xCC24)

            # Log state periodically
            if frame % 50 == 0:
                logger.debug(f"wait_for_battle_menu frame {frame}: text_box={text_box}, text_delay={text_delay}, max_menu={max_menu}, menu_joypad=0x{menu_joypad:02X}")

            if text_box > 0:
                # Text is showing - wait for it to finish printing, then advance
                if text_delay == 0:
                    self.press_a(wait_frames=8)
                    a_presses += 1
                    frames_without_progress = 0
                else:
                    # Text still printing
                    self.emulator.tick(2)
            else:
                # No text box - might be in animation or transitioning
                self.emulator.tick(2)
                frames_without_progress += 1

                # If stuck with no text for a while, try pressing A
                # (might need to dismiss a result or continue)
                if frames_without_progress > 30:
                    self.press_a(wait_frames=4)
                    a_presses += 1
                    frames_without_progress = 0

            # Render periodically so spectator shows progress (every 10 frames)
            if frame % 10 == 0:
                self._maybe_render()

        # Debug info on timeout
        state = self.get_battle_state()
        logger.warning(f"Battle menu timeout after {timeout} frames, {a_presses} A presses. State: {state}")
        return False

    def wait_for_move_menu(self, timeout: int = 60) -> bool:
        """
        Wait until the move selection menu is open.

        Args:
            timeout: Maximum frames to wait

        Returns:
            True if move menu opened, False if timeout
        """
        for frame in range(timeout):
            fight_flag = self.emulator.read_memory(FIGHT_MENU_OPEN)
            text_done = self.is_text_done()
            if frame % 20 == 0:
                logger.debug(f"wait_for_move_menu frame {frame}: FIGHT_MENU_OPEN=0x{fight_flag:02X}, text_done={text_done}")
            if fight_flag != 0 and text_done:
                logger.debug(f"wait_for_move_menu: menu opened at frame {frame}")
                return True
            self.emulator.tick(1)
        logger.warning(f"wait_for_move_menu: timeout after {timeout} frames, FIGHT_MENU_OPEN=0x{self.emulator.read_memory(FIGHT_MENU_OPEN):02X}")
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

    def wait_for_battle_end(self, timeout: int = 300) -> bool:
        """
        Wait until battle ends.

        Args:
            timeout: Maximum frames to wait

        Returns:
            True if battle ended, False if timeout
        """
        for _ in range(timeout):
            if not self.is_in_battle():
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

    def press_a(self, wait_frames: int = 8) -> None:
        """Press A button."""
        self.press_button(BUTTON_A, wait_frames)

    def press_b(self, wait_frames: int = 8) -> None:
        """Press B button."""
        self.press_button(BUTTON_B, wait_frames)

    # === Main Battle Menu Navigation ===

    def navigate_to_battle_option(self, option: str) -> bool:
        """
        Navigate to a main battle menu option (FIGHT/BAG/POKEMON/RUN).

        The battle menu is a 2x2 grid:
        [FIGHT]  [BAG]
        [POKEMON][RUN]

        Args:
            option: One of "fight", "bag", "pokemon", "run"

        Returns:
            True if successfully navigated
        """
        option = option.lower()
        if option not in BATTLE_MENU_OPTIONS:
            logger.error(f"Unknown battle option: {option}")
            return False

        target = BATTLE_MENU_OPTIONS[option]
        target_x = target % 2  # 0=left, 1=right
        target_y = target // 2  # 0=top, 1=bottom

        logger.debug(f"Navigating to battle option '{option}' (pos {target})")

        # Navigate in the 2x2 grid
        for attempt in range(6):
            current = self.get_battle_menu_cursor()
            current_x = current % 2
            current_y = current // 2

            if current == target:
                logger.debug(f"Reached battle option '{option}'")
                return True

            # Move horizontally
            if current_x < target_x:
                self.press_button(BUTTON_RIGHT)
            elif current_x > target_x:
                self.press_button(BUTTON_LEFT)
            # Move vertically
            elif current_y < target_y:
                self.press_button(BUTTON_DOWN)
            elif current_y > target_y:
                self.press_button(BUTTON_UP)

        return self.get_battle_menu_cursor() == target

    def select_fight(self) -> bool:
        """
        Select FIGHT to open move menu.

        Returns:
            True if move menu opened successfully
        """
        logger.debug("select_fight: navigating to FIGHT option")
        if not self.navigate_to_battle_option("fight"):
            logger.warning("select_fight: failed to navigate to FIGHT")
            return False

        logger.debug("select_fight: pressing A to open move menu")
        self.press_a()

        # Wait for move menu to open - if the memory flag isn't reliable,
        # we still proceed after a reasonable wait
        result = self.wait_for_move_menu(timeout=60)
        if not result:
            # The FIGHT_MENU_OPEN flag may not be set reliably in all cases.
            # Wait a bit more and assume we're at the move menu.
            logger.debug("select_fight: FIGHT_MENU_OPEN not set, waiting extra frames")
            self.emulator.tick(30)
            # Check if we're still in battle and text is done - likely at move menu
            if self.is_in_battle() and self.is_text_done():
                logger.debug("select_fight: assuming move menu is open")
                return True
        logger.debug(f"select_fight: result={result}")
        return result

    def select_bag(self) -> bool:
        """
        Select BAG/ITEM option.

        Returns:
            True if bag menu opened
        """
        if not self.navigate_to_battle_option("bag"):
            return False

        self.press_a()
        self.emulator.tick(20)  # Wait for bag menu
        return True

    def select_pokemon(self) -> bool:
        """
        Select POKEMON option to switch.

        Returns:
            True if Pokemon menu opened
        """
        if not self.navigate_to_battle_option("pokemon"):
            return False

        self.press_a()
        self.emulator.tick(20)  # Wait for party screen
        return True

    def select_run(self) -> bool:
        """
        Select RUN option to flee.

        Returns:
            True if run was selected
        """
        logger.debug("select_run: navigating to RUN option")
        if not self.navigate_to_battle_option("run"):
            logger.warning("select_run: failed to navigate to RUN")
            return False

        logger.debug("select_run: pressing A to run")
        self.press_a()

        # Wait for run action to process (game shows "Got away safely!" or "Can't escape!")
        self.emulator.tick(30)
        logger.debug("select_run: run action submitted")
        return True

    # === Move Selection ===

    def navigate_to_move(self, move_slot: int) -> bool:
        """
        Navigate to a specific move in the move menu (2x2 grid).

        Move layout:
        [Move 1] [Move 2]
        [Move 3] [Move 4]

        Args:
            move_slot: Move slot 1-4

        Returns:
            True if successfully navigated to move
        """
        if move_slot < 1 or move_slot > 4:
            logger.error(f"Invalid move slot: {move_slot}")
            return False

        target_x, target_y = MOVE_POSITIONS[move_slot]

        logger.debug(f"Navigating to move slot {move_slot} at ({target_x}, {target_y})")

        for attempt in range(6):
            current = self.get_move_cursor()
            current_x = current % 2
            current_y = current // 2

            # Check if at target (convert target coords to cursor value)
            target_cursor = target_y * 2 + target_x
            if current == target_cursor:
                logger.debug(f"Reached move slot {move_slot}")
                return True

            # Move horizontally
            if current_x < target_x:
                self.press_button(BUTTON_RIGHT)
            elif current_x > target_x:
                self.press_button(BUTTON_LEFT)
            # Move vertically
            elif current_y < target_y:
                self.press_button(BUTTON_DOWN)
            elif current_y > target_y:
                self.press_button(BUTTON_UP)

        # Final check
        current = self.get_move_cursor()
        target_cursor = target_y * 2 + target_x
        return current == target_cursor

    def select_move(self, move_slot: int) -> bool:
        """
        Select a move to use in battle.

        Args:
            move_slot: Move slot 1-4

        Returns:
            True if move was selected
        """
        # First, make sure we're at the FIGHT menu
        fight_menu_flag = self.emulator.read_memory(FIGHT_MENU_OPEN)
        logger.debug(f"FIGHT_MENU_OPEN=0x{fight_menu_flag:02X}, is_fight_menu_open={self.is_fight_menu_open()}")

        if not self.is_fight_menu_open():
            logger.debug("Fight menu not open, calling select_fight()")
            if not self.select_fight():
                logger.warning("Failed to open FIGHT menu")
                return False
        else:
            logger.debug("Fight menu already open (skipping select_fight)")

        # Navigate to the move
        if not self.navigate_to_move(move_slot):
            logger.warning(f"Failed to navigate to move {move_slot}")
            return False

        # Select the move
        self.press_a()
        logger.info(f"Selected move {move_slot}, waiting for attack animation...")
        return True

    # === Pokemon Selection (in battle) ===

    def navigate_to_party_pokemon(self, slot: int) -> bool:
        """
        Navigate to a Pokemon in the party menu during battle.

        Args:
            slot: Pokemon slot 1-6

        Returns:
            True if successfully navigated
        """
        if slot < 1 or slot > 6:
            logger.error(f"Invalid Pokemon slot: {slot}")
            return False

        target = slot - 1  # Convert to 0-indexed

        for attempt in range(10):
            current = self.get_menu_cursor()
            if current == target:
                return True

            if current < target:
                self.press_button(BUTTON_DOWN)
            else:
                self.press_button(BUTTON_UP)

        return self.get_menu_cursor() == target

    def switch_pokemon(self, slot: int) -> bool:
        """
        Switch to a different Pokemon during battle.

        Args:
            slot: Pokemon slot 1-6

        Returns:
            True if switch was initiated
        """
        # Open Pokemon menu
        if not self.select_pokemon():
            return False

        # Navigate to the Pokemon
        if not self.navigate_to_party_pokemon(slot):
            self.press_b()  # Cancel and close menu
            return False

        # Select the Pokemon (this brings up submenu)
        self.press_a()
        self.emulator.tick(10)

        # Select SWITCH (first option in submenu)
        self.press_a()
        logger.debug(f"Initiated switch to Pokemon {slot}")
        return True

    # === High-Level Actions ===

    def execute_attack(self, move_slot: int) -> bool:
        """
        Execute an attack in battle with full verification.

        Args:
            move_slot: Which move to use (1-4)

        Returns:
            True if attack was executed
        """
        if not self.is_in_battle():
            logger.error("Not in battle!")
            return False

        # Wait for our turn
        if not self.wait_for_battle_menu(timeout=60):
            logger.warning("Timed out waiting for battle menu")
            return False

        # Select the move
        if not self.select_move(move_slot):
            return False

        logger.info(f"Executed attack with move slot {move_slot}")
        return True

    def execute_run(self) -> bool:
        """
        Attempt to run from battle with full verification.

        Returns:
            True if run was attempted
        """
        if not self.is_in_battle():
            logger.error("Not in battle!")
            return False

        # Wait for our turn
        if not self.wait_for_battle_menu(timeout=60):
            logger.warning("Timed out waiting for battle menu")
            return False

        # Select RUN
        if not self.select_run():
            return False

        logger.info("Attempted to run from battle")
        return True

    def execute_switch(self, slot: int) -> bool:
        """
        Switch Pokemon during battle with full verification.

        Args:
            slot: Pokemon slot to switch to (1-6)

        Returns:
            True if switch was initiated
        """
        if not self.is_in_battle():
            logger.error("Not in battle!")
            return False

        # Wait for our turn
        if not self.wait_for_battle_menu(timeout=60):
            logger.warning("Timed out waiting for battle menu")
            return False

        # Switch to the Pokemon
        if not self.switch_pokemon(slot):
            return False

        logger.info(f"Switching to Pokemon slot {slot}")
        return True

    # === Utility ===

    def advance_battle_text(self, max_presses: int = 10) -> bool:
        """
        Advance through battle text/dialogue by pressing A.

        Args:
            max_presses: Maximum A presses

        Returns:
            True if text was advanced
        """
        for i in range(max_presses):
            if not self.is_menu_open():
                return True

            # Check if we're at a battle menu (don't advance past it)
            if self.is_text_done() and self.get_menu_size() == 4:
                # Likely at main battle menu
                return True

            self.press_a(wait_frames=16)
            # Render so spectator shows the text advancing
            self._maybe_render()

        return not self.is_menu_open() or self.is_text_done()

    def cancel_to_battle_menu(self, max_presses: int = 5) -> bool:
        """
        Press B to cancel back to main battle menu.

        Args:
            max_presses: Maximum B presses

        Returns:
            True if back at main battle menu
        """
        for i in range(max_presses):
            # Check if fight menu is closed (we're back at main)
            if not self.is_fight_menu_open():
                self.emulator.tick(5)
                return True
            self.press_b(wait_frames=12)

        return not self.is_fight_menu_open()
