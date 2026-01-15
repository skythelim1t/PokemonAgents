"""Run a trained RL model on Pokemon Red."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.environment.pokemon_env import PokemonRedEnv


def run_model(
    model_path: Path,
    rom_path: Path,
    init_state: Path | None = None,
    num_episodes: int = 1,
    max_steps: int = 10000,
    deterministic: bool = True,
    render: bool = True,
) -> None:
    """
    Run a trained model.

    Args:
        model_path: Path to the saved model (.zip file)
        rom_path: Path to the ROM file
        init_state: Optional initial save state
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode
        deterministic: Use deterministic actions
        render: Show the game window
    """
    print("=" * 60)
    print("Running Trained Model")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"ROM: {rom_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = PPO.load(str(model_path))

    # Create environment
    env = PokemonRedEnv(
        rom_path=rom_path,
        init_state=init_state,
        headless=not render,
        action_freq=24,
        max_steps=max_steps,
    )

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs, info = env.reset()
        total_reward = 0.0
        step = 0

        while step < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Print progress every 500 steps
            if step % 500 == 0:
                print(
                    f"Step {step:5d} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Badges: {info.get('badges', 0)} | "
                    f"Level: {info.get('total_level', 0):3d} | "
                    f"Maps: {info.get('maps_visited', 0):3d}"
                )

            if terminated or truncated:
                break

        print(f"\nEpisode finished:")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Badges: {info.get('badges', 0)}")
        print(f"  Party level: {info.get('total_level', 0)}")
        print(f"  Maps visited: {info.get('maps_visited', 0)}")
        print(f"  Coords visited: {info.get('coords_visited', 0)}")

    env.close()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Run a trained RL model")
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--rom", type=Path, required=True, help="Path to Pokemon Red ROM"
    )
    parser.add_argument(
        "--state", type=Path, default=None, help="Initial save state"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Max steps per episode"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Run without display"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic actions"
    )

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model not found at {args.model}")
        return

    if not args.rom.exists():
        print(f"Error: ROM not found at {args.rom}")
        return

    run_model(
        model_path=args.model,
        rom_path=args.rom,
        init_state=args.state,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
