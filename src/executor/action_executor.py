"""Executor layer for translating strategic actions to button sequences."""

from typing import Iterator

from src.agents.llm.actions import StrategicAction, ACTION_TO_BUTTONS, SIMPLE_ACTION_TO_BUTTON


class ActionExecutor:
    """
    Translates strategic actions into button press sequences.

    For initial implementation, this provides simple 1:1 mappings.
    Can be extended later for complex multi-step sequences.
    """

    def __init__(self, simple_mode: bool = True) -> None:
        """
        Initialize the executor.

        Args:
            simple_mode: If True, use simple 1:1 mappings.
                        If False, use full multi-button sequences.
        """
        self.simple_mode = simple_mode
        self._current_sequence: list[int] = []
        self._sequence_index: int = 0
        self._current_action: StrategicAction | None = None

    def get_buttons(self, action: StrategicAction) -> list[int]:
        """
        Get the button sequence for a strategic action.

        Args:
            action: The strategic action to execute

        Returns:
            List of button indices to press
        """
        if self.simple_mode:
            return [SIMPLE_ACTION_TO_BUTTON.get(action, 0)]
        else:
            return ACTION_TO_BUTTONS.get(action, [0])

    def execute(self, action: StrategicAction) -> Iterator[int]:
        """
        Generator that yields button presses for an action.

        Yields one button at a time for use in step-by-step execution.
        """
        buttons = self.get_buttons(action)
        for button in buttons:
            yield button

    def start_sequence(self, action: StrategicAction) -> None:
        """Start a new button sequence for the given action."""
        self._current_action = action
        self._current_sequence = self.get_buttons(action)
        self._sequence_index = 0

    def next_button(self) -> int | None:
        """
        Get the next button in the current sequence.

        Returns:
            Button index, or None if sequence complete.
        """
        if self._sequence_index >= len(self._current_sequence):
            return None

        button = self._current_sequence[self._sequence_index]
        self._sequence_index += 1
        return button

    @property
    def sequence_complete(self) -> bool:
        """Check if current sequence is complete."""
        return self._sequence_index >= len(self._current_sequence)

    @property
    def current_action(self) -> StrategicAction | None:
        """Get the current action being executed."""
        return self._current_action

    def reset(self) -> None:
        """Reset the executor state."""
        self._current_sequence = []
        self._sequence_index = 0
        self._current_action = None
