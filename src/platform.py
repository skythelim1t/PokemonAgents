"""Main platform orchestrator for running Pokemon agents."""

import argparse
import logging
from pathlib import Path

from src.agents.random_agent import RandomAgent
from src.environment.pokemon_env import PokemonRedEnv

logger = logging.getLogger(__name__)


def run_random_agent(
    rom_path: Path,
    num_steps: int = 1000,
    headless: bool = True,
    init_state: Path | None = None,
    speed: int = 0,
) -> None:
    """
    Run a random agent in the Pokemon environment.

    Args:
        rom_path: Path to the ROM file
        num_steps: Number of steps to run
        headless: Run without display window
        init_state: Optional initial save state
        speed: Emulation speed (0=unlimited, 1=normal, 2=2x, etc.)
    """
    env = PokemonRedEnv(
        rom_path=rom_path,
        init_state=init_state,
        headless=headless,
        action_freq=24,
        max_steps=num_steps,
        speed=speed,
    )

    agent = RandomAgent(num_actions=env.action_space.n, seed=42)

    print(f"Starting random agent for {num_steps} steps...")
    print(f"ROM: {rom_path}")
    print(f"Headless: {headless}")
    print("-" * 50)

    obs, info = env.reset()
    agent.reset()

    total_reward = 0.0
    step = 0

    try:
        while step < num_steps:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Print progress every 100 steps
            if step % 100 == 0:
                print(
                    f"Step {step:5d} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Badges: {info.get('badges', 0)} | "
                    f"Level: {info.get('total_level', 0):3d} | "
                    f"Maps: {info.get('maps_visited', 0):3d} | "
                    f"Battle: {info.get('in_battle', False)}"
                )

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                break

            step += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        env.close()

    print("-" * 50)
    print(f"Final Stats:")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Badges: {info.get('badges', 0)}")
    print(f"  Total Level: {info.get('total_level', 0)}")
    print(f"  Maps Visited: {info.get('maps_visited', 0)}")
    print(f"  Coords Visited: {info.get('coords_visited', 0)}")


def run_llm_agent(
    rom_path: Path,
    num_steps: int = 1000,
    headless: bool = True,
    init_state: Path | None = None,
    speed: int = 1,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    use_vision: bool = False,
    action_skip: int = 10,
) -> None:
    """
    Run an LLM agent in the Pokemon environment.

    Args:
        rom_path: Path to the ROM file
        num_steps: Number of steps to run
        headless: Run without display window
        init_state: Optional initial save state
        speed: Emulation speed (0=unlimited, 1=normal, 2=2x, etc.)
        provider: LLM provider (anthropic, openai, bedrock)
        model: Model name/ID
        use_vision: Include screen images in prompts
        action_skip: Only call LLM every N frames
    """
    from src.agents.llm_agent import LLMAgent, LLMAgentConfig

    env = PokemonRedEnv(
        rom_path=rom_path,
        init_state=init_state,
        headless=headless,
        action_freq=24,
        max_steps=num_steps,
        speed=speed,
    )

    config = LLMAgentConfig(
        provider=provider,
        model=model,
        use_vision=use_vision,
        action_skip=action_skip,
    )
    agent = LLMAgent(config)

    print(f"Starting LLM agent for {num_steps} steps...")
    print(f"ROM: {rom_path}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Vision: {use_vision}")
    print(f"Action skip: {action_skip}")
    print("-" * 50)

    obs, info = env.reset()
    agent.reset()

    total_reward = 0.0
    step = 0

    try:
        while step < num_steps:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, obs, terminated or truncated, info)
            total_reward += reward

            # Print progress every 10 steps (LLM is slower)
            if step % 10 == 0:
                print(
                    f"Step {step:5d} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Action: {agent.last_action_name:12s} | "
                    f"Badges: {info.get('badges', 0)} | "
                    f"Battle: {info.get('in_battle', False)} | "
                    f"Tokens: {agent.tokens_used}"
                )

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                break

            step += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        env.close()

    print("-" * 50)
    print(f"Final Stats:")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Badges: {info.get('badges', 0)}")
    print(f"  Total Level: {info.get('total_level', 0)}")
    print(f"  Maps Visited: {info.get('maps_visited', 0)}")
    print(f"  Total Tokens Used: {agent.tokens_used}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pokemon Agent Platform")
    parser.add_argument(
        "--rom",
        type=Path,
        required=True,
        help="Path to the Pokemon ROM file (.gb or .gbc)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Path to initial save state (optional)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps to run (default: 1000)",
    )
    parser.add_argument(
        "--spectate",
        action="store_true",
        help="Run with display window (spectate mode)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display window (default)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=0,
        help="Emulation speed (0=unlimited, 1=normal, 2=2x, etc.)",
    )

    # Agent selection
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "llm"],
        help="Agent type to run (default: random)",
    )

    # LLM-specific options
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "bedrock"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Model name/ID (default: claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Include screen images in LLM prompts",
    )
    parser.add_argument(
        "--action-skip",
        type=int,
        default=10,
        help="Only call LLM every N frames (default: 10)",
    )

    args = parser.parse_args()

    # Default to headless unless spectate is specified
    headless = not args.spectate

    if not args.rom.exists():
        print(f"Error: ROM file not found: {args.rom}")
        print("\nPlease place your Pokemon Red ROM in the roms/ directory.")
        print("Expected filename: pokemon_red.gb or PokemonRed.gb")
        return

    if args.agent == "llm":
        # For LLM agent, default to normal speed if spectating
        speed = args.speed if args.speed > 0 else (1 if args.spectate else 0)
        run_llm_agent(
            rom_path=args.rom,
            num_steps=args.steps,
            headless=headless,
            init_state=args.state,
            speed=speed,
            provider=args.provider,
            model=args.model,
            use_vision=args.vision,
            action_skip=args.action_skip,
        )
    else:
        run_random_agent(
            rom_path=args.rom,
            num_steps=args.steps,
            headless=headless,
            init_state=args.state,
            speed=args.speed,
        )


if __name__ == "__main__":
    main()
