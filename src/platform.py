"""Main platform orchestrator for running Pokemon agents."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.agents.random_agent import RandomAgent
from src.environment.pokemon_env import PokemonRedEnv

logger = logging.getLogger(__name__)


def create_recorder(args, agent_type: str):
    """Create a recorder if --record flag is set."""
    if not args.record:
        return None

    from src.recording.recorder import PlaythroughRecorder

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.record_dir / f"{timestamp}_{agent_type}"

    recorder = PlaythroughRecorder(
        output_dir=output_dir,
        fps=10,  # Reasonable playback speed
        scale=2,  # 2x upscale for watchable video
    )
    recorder.start(
        rom_path=str(args.rom),
        agent_type=agent_type,
        init_state=str(args.state) if args.state else None,
    )

    return recorder


def run_random_agent(
    rom_path: Path,
    num_steps: int = 1000,
    headless: bool = True,
    init_state: Path | None = None,
    speed: int = 0,
    recorder=None,
) -> None:
    """
    Run a random agent in the Pokemon environment.

    Args:
        rom_path: Path to the ROM file
        num_steps: Number of steps to run
        headless: Run without display window
        init_state: Optional initial save state
        speed: Emulation speed (0=unlimited, 1=normal, 2=2x, etc.)
        recorder: Optional PlaythroughRecorder for recording gameplay
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
    if recorder:
        print(f"Recording to: {recorder.output_dir}")
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

            # Record step if recorder is active
            if recorder:
                recorder.record_step(step, action, obs, reward, info)

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
        if recorder:
            recorder.stop()

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
    log_conversation: bool = False,
    recorder=None,
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
        log_conversation: Print LLM prompts and responses
        recorder: Optional PlaythroughRecorder for recording gameplay
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
        log_conversation=log_conversation,
    )
    agent = LLMAgent(config)

    print(f"Starting LLM agent for {num_steps} steps...")
    print(f"ROM: {rom_path}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Vision: {use_vision}")
    print(f"Headless: {headless}")
    if recorder:
        print(f"Recording to: {recorder.output_dir}")
    print("-" * 50)

    obs, info = env.reset()
    agent.reset()

    # For spectate mode, run some extra frames to ensure window is visible
    if not headless and env.emulator is not None:
        print("Running initial frames for display...")
        env.emulator.tick(30)

    total_reward = 0.0
    step = 0

    try:
        while step < num_steps:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, obs, terminated or truncated, info)
            total_reward += reward

            # Record step if recorder is active
            if recorder:
                recorder.record_step(step, action, obs, reward, info)

            # For spectate mode, run extra frames to keep display responsive
            if not headless and env.emulator is not None:
                env.emulator.tick(2)

            # Print progress every 20 steps
            if step % 20 == 0:
                exec_state = "BUSY" if agent.executor.is_busy else "IDLE"
                print(
                    f"Step {step:5d} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Action: {agent.last_action_name:15s} | "
                    f"Exec: {exec_state:4s} | "
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
        if recorder:
            recorder.stop()

    print("-" * 50)
    print(f"Final Stats:")
    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Badges: {info.get('badges', 0)}")
    print(f"  Total Level: {info.get('total_level', 0)}")
    print(f"  Maps Visited: {info.get('maps_visited', 0)}")
    print(f"  Total Tokens Used: {agent.tokens_used}")


def run_tool_agent(
    rom_path: Path,
    num_steps: int = 1000,
    headless: bool = True,
    init_state: Path | None = None,
    speed: int = 1,
    provider: str = "bedrock",
    model: str = "anthropic.claude-3-haiku-20240307-v1:0",
    log_conversation: bool = False,
    run_id: str | None = None,
    recorder=None,
) -> None:
    """
    Run the tool-based LLM agent.

    This agent uses Claude's function calling API instead of text parsing.

    Args:
        rom_path: Path to the ROM file
        num_steps: Number of steps to run
        headless: Run without display window
        init_state: Optional initial save state
        speed: Emulation speed
        provider: LLM provider
        model: Model name/ID
        log_conversation: Print tool calls and responses
        run_id: Optional run ID for loading/saving knowledge
        recorder: Optional PlaythroughRecorder for recording gameplay
    """
    from src.agents.tool_agent import ToolAgent, ToolAgentConfig
    from src.agents.llm.knowledge import KnowledgeBase

    env = PokemonRedEnv(
        rom_path=rom_path,
        init_state=init_state,
        headless=headless,
        action_freq=24,
        max_steps=num_steps,
        speed=speed,
    )

    config = ToolAgentConfig(
        provider=provider,
        model=model,
        log_conversation=log_conversation,
    )

    # Load or create knowledge base
    if run_id:
        knowledge_path = KnowledgeBase.get_run_path(run_id)
        knowledge = KnowledgeBase.load(knowledge_path)
        print(f"Resuming run: {run_id}")
    else:
        knowledge = None  # Agent will create fresh one

    agent = ToolAgent(config, knowledge=knowledge)
    agent.set_env(env)

    print(f"Starting Tool Agent for {num_steps} steps...")
    print(f"ROM: {rom_path}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Mode: Tool-based (function calling)")
    print(f"Run ID: {run_id or 'new session'}")
    print(f"Headless: {headless}")
    if recorder:
        print(f"Recording to: {recorder.output_dir}")
    print("-" * 50)

    obs, info = env.reset()
    agent.reset()

    if not headless and env.emulator is not None:
        print("Running initial frames for display...")
        env.emulator.tick(30)

    total_reward = 0.0
    step = 0

    try:
        while step < num_steps:
            # Tool agent handles everything internally
            agent.act(obs, info)

            # Get updated state
            obs = env.emulator.get_screen() if env.emulator else obs
            info = env._get_info()

            # Record step if recorder is active (tool agent doesn't track rewards)
            if recorder:
                recorder.record_step(step, 0, obs, 0.0, info)

            # For spectate mode
            if not headless and env.emulator is not None:
                env.emulator.tick(2)

            # Print progress every 10 steps
            if step % 10 == 0:
                print(
                    f"Step {step:5d} | "
                    f"Last Tool: {agent.last_action_name:30s} | "
                    f"Badges: {info.get('badges', 0)} | "
                    f"Battle: {info.get('in_battle', False)} | "
                    f"Tokens: {agent.tokens_used}"
                )

                # Auto-save knowledge every 10 steps if run_id provided
                if run_id:
                    knowledge_path = KnowledgeBase.get_run_path(run_id)
                    agent.knowledge.save(knowledge_path)

            step += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        env.close()
        if recorder:
            recorder.stop()

        # Save knowledge if run_id provided
        if run_id:
            knowledge_path = KnowledgeBase.get_run_path(run_id)
            agent.knowledge.save(knowledge_path)
            print(f"Saved knowledge to {knowledge_path}")

    print("-" * 50)
    print(f"Final Stats:")
    print(f"  Total Steps: {step}")
    print(f"  Final Badges: {info.get('badges', 0)}")
    print(f"  Total Tokens Used: {agent.tokens_used}")
    if run_id:
        print(f"  Run ID: {run_id} (use --run-id {run_id} to resume)")


def main() -> None:
    """Main entry point."""
    # Configure logging to see debug output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

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
        choices=["random", "llm", "tool"],
        help="Agent type to run: random, llm (text parsing), tool (function calling)",
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
        "--log-conversation",
        action="store_true",
        help="Log LLM prompts and responses (verbose)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for resuming a previous session (loads/saves knowledge)",
    )

    # Recording options
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record playthrough to MP4 video + training data JSON",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory for recordings (default: recordings/)",
    )

    args = parser.parse_args()

    # Default to headless unless spectate is specified
    headless = not args.spectate

    if not args.rom.exists():
        print(f"Error: ROM file not found: {args.rom}")
        print("\nPlease place your Pokemon Red ROM in the roms/ directory.")
        print("Expected filename: pokemon_red.gb or PokemonRed.gb")
        return

    if args.agent == "tool":
        # Tool-based agent with function calling
        speed = args.speed if args.speed > 0 else (1 if args.spectate else 0)
        recorder = create_recorder(args, "tool")
        run_tool_agent(
            rom_path=args.rom,
            num_steps=args.steps,
            headless=headless,
            init_state=args.state,
            speed=speed,
            provider=args.provider,
            model=args.model,
            log_conversation=args.log_conversation,
            run_id=args.run_id,
            recorder=recorder,
        )
    elif args.agent == "llm":
        # Text-parsing LLM agent
        speed = args.speed if args.speed > 0 else (1 if args.spectate else 0)
        recorder = create_recorder(args, "llm")
        run_llm_agent(
            rom_path=args.rom,
            num_steps=args.steps,
            headless=headless,
            init_state=args.state,
            speed=speed,
            provider=args.provider,
            model=args.model,
            use_vision=args.vision,
            log_conversation=args.log_conversation,
            recorder=recorder,
        )
    else:
        recorder = create_recorder(args, "random")
        run_random_agent(
            rom_path=args.rom,
            num_steps=args.steps,
            headless=headless,
            init_state=args.state,
            speed=args.speed,
            recorder=recorder,
        )


if __name__ == "__main__":
    main()
