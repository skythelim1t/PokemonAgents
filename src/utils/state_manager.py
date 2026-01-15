"""Utility for creating and managing save states."""

import argparse
from pathlib import Path

from src.emulator.pyboy_wrapper import EmulatorWrapper
from src.emulator.memory_map import GameState


def interactive_state_creator(rom_path: Path, save_path: Path) -> None:
    """
    Run the game interactively and save state when ready.

    Controls:
    - Arrow keys: Move
    - Z: A button
    - X: B button
    - Enter: Start
    - Backspace: Select
    - S: Save state
    - Q: Quit
    """
    print("=" * 60)
    print("Interactive State Creator")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow keys : D-pad")
    print("  Z          : A button")
    print("  X          : B button")
    print("  Enter      : Start")
    print("  Backspace  : Select")
    print("  S          : Save state to file")
    print("  Q          : Quit")
    print("\nPlay until you're ready, then press S to save.")
    print("=" * 60)

    # Use SDL2 window for interactive play
    emulator = EmulatorWrapper(rom_path, headless=False, speed=1)
    game_state = GameState(emulator)

    print("\nGame started! The PyBoy window should appear.")
    print("Use the controls listed above to play.")
    print("Press S in this terminal when ready to save.\n")

    import sys
    import select
    import tty
    import termios

    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        running = True
        while running:
            # Tick the emulator
            emulator.tick()

            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1).lower()

                if char == 's':
                    # Save state
                    emulator.save_state(save_path)

                    # Print game info
                    x, y = game_state.get_player_position()
                    badges = game_state.get_badge_count()
                    party_count = game_state.get_party_count()
                    levels = game_state.get_party_levels()
                    map_id = game_state.get_map_id()

                    print("\n" + "=" * 60)
                    print(f"STATE SAVED: {save_path}")
                    print("=" * 60)
                    print(f"  Map ID: {map_id}")
                    print(f"  Position: ({x}, {y})")
                    print(f"  Badges: {badges}")
                    print(f"  Party: {party_count} Pokemon")
                    print(f"  Levels: {levels}")
                    print("=" * 60 + "\n")

                elif char == 'q':
                    print("\nQuitting...")
                    running = False

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        emulator.close()

    print("Done!")


def view_state_info(rom_path: Path, state_path: Path) -> None:
    """Load a state and display its info."""
    if not state_path.exists():
        print(f"State file not found: {state_path}")
        return

    emulator = EmulatorWrapper(rom_path, headless=True)
    emulator.load_state(state_path)
    game_state = GameState(emulator)

    x, y = game_state.get_player_position()
    badges = game_state.get_badge_count()
    party_count = game_state.get_party_count()
    levels = game_state.get_party_levels()
    map_id = game_state.get_map_id()
    money = game_state.get_money()
    hp_list = game_state.get_party_hp()

    print("=" * 60)
    print(f"State: {state_path}")
    print("=" * 60)
    print(f"  Map ID: {map_id}")
    print(f"  Position: ({x}, {y})")
    print(f"  Badges: {badges}/8")
    print(f"  Money: ${money}")
    print(f"  Party: {party_count} Pokemon")
    for i, (level, (hp, max_hp)) in enumerate(zip(levels, hp_list)):
        print(f"    #{i+1}: Level {level}, HP {hp}/{max_hp}")
    print("=" * 60)

    emulator.close()


def main():
    parser = argparse.ArgumentParser(description="Pokemon Save State Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create state command
    create_parser = subparsers.add_parser("create", help="Create a new save state interactively")
    create_parser.add_argument("--rom", type=Path, required=True, help="Path to ROM file")
    create_parser.add_argument("--output", type=Path, default=Path("saves/init.state"),
                               help="Output state file path")

    # View state command
    view_parser = subparsers.add_parser("view", help="View info about a save state")
    view_parser.add_argument("--rom", type=Path, required=True, help="Path to ROM file")
    view_parser.add_argument("--state", type=Path, required=True, help="State file to view")

    args = parser.parse_args()

    if args.command == "create":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        interactive_state_creator(args.rom, args.output)
    elif args.command == "view":
        view_state_info(args.rom, args.state)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
