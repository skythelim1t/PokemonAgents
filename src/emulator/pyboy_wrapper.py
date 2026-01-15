"""PyBoy emulator wrapper for Pokemon games."""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pyboy import PyBoy

# Button mappings
BUTTONS = ["a", "b", "start", "select", "up", "down", "left", "right"]
BUTTON_TO_INDEX = {btn: idx for idx, btn in enumerate(BUTTONS)}


class EmulatorWrapper:
    """Wrapper around PyBoy emulator for Pokemon games."""

    def __init__(
        self,
        rom_path: Path | str,
        headless: bool = True,
        speed: int = 0,
    ) -> None:
        """
        Initialize the emulator.

        Args:
            rom_path: Path to the ROM file (.gb or .gbc)
            headless: If True, run without display window
            speed: Emulation speed (0 = unlimited, 1 = normal, 2 = 2x, etc.)
        """
        self.rom_path = Path(rom_path)
        if not self.rom_path.exists():
            raise FileNotFoundError(f"ROM not found: {self.rom_path}")

        window_type: Literal["null", "SDL2"] = "null" if headless else "SDL2"
        self.pyboy = PyBoy(str(self.rom_path), window=window_type)
        self.pyboy.set_emulation_speed(speed)

    def tick(self, count: int = 1) -> None:
        """Advance the emulator by the specified number of frames."""
        for _ in range(count):
            self.pyboy.tick()

    def press_button(self, button: str, duration: int = 8) -> None:
        """
        Press a button for the specified duration (in frames).

        Args:
            button: One of 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'
            duration: How many frames to hold the button
        """
        if button not in BUTTONS:
            raise ValueError(f"Invalid button: {button}. Must be one of {BUTTONS}")

        self.pyboy.button_press(button)
        self.tick(duration)
        self.pyboy.button_release(button)

    def press_button_sequence(
        self, buttons: list[str], duration: int = 8, gap: int = 4
    ) -> None:
        """
        Press a sequence of buttons.

        Args:
            buttons: List of button names to press in order
            duration: How many frames to hold each button
            gap: How many frames to wait between button presses
        """
        for button in buttons:
            self.press_button(button, duration)
            self.tick(gap)

    def get_screen(self) -> NDArray[np.uint8]:
        """Get the current screen as a numpy array (H, W, 3) RGB."""
        # PyBoy returns RGBA, we convert to RGB
        rgba = np.array(self.pyboy.screen.image)
        return rgba[:, :, :3]  # Drop alpha channel

    def get_screen_grayscale(self) -> NDArray[np.uint8]:
        """Get the current screen as grayscale (H, W)."""
        screen = self.get_screen()
        return np.dot(screen[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def read_memory(self, address: int) -> int:
        """Read a single byte from memory."""
        return self.pyboy.memory[address]

    def read_memory_range(self, start: int, length: int) -> list[int]:
        """Read a range of bytes from memory."""
        return [self.pyboy.memory[start + i] for i in range(length)]

    def save_state(self, path: Path | str) -> None:
        """Save the current emulator state to a file."""
        path = Path(path)
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, path: Path | str) -> None:
        """Load an emulator state from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Save state not found: {path}")
        with open(path, "rb") as f:
            self.pyboy.load_state(f)

    def close(self) -> None:
        """Clean up and close the emulator."""
        self.pyboy.stop()

    @property
    def screen_shape(self) -> tuple[int, int, int]:
        """Return the screen dimensions (H, W, C)."""
        return (144, 160, 3)  # Game Boy screen is 160x144

    def __enter__(self) -> "EmulatorWrapper":
        return self

    def __exit__(self, *args) -> None:
        self.close()
