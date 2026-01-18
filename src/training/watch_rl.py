"""Watch a trained RL agent play Pokemon Red."""

import argparse
import time
from pathlib import Path

import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from src.environment.pokemon_env import PokemonRedEnv


def watch(
    model_path: Path,
    rom_path: Path,
    init_state: Path | None = None,
    speed: float = 1.0,
    max_steps: int = 10000,
    agent: str = "ppo",
) -> None:
    """
    Watch a trained RL agent play.

    Args:
        model_path: Path to the trained model (.zip file)
        rom_path: Path to the ROM file
        init_state: Optional initial save state
        speed: Playback speed (1.0 = normal, 2.0 = 2x, etc.)
        max_steps: Maximum steps to run
        agent: Agent type - 'ppo' or 'recurrent'
    """
    print("=" * 60)
    print("Pokemon Red RL Playback")
    print("=" * 60)
    print(f"Agent: {agent.upper()}")
    print(f"Model: {model_path}")
    print(f"ROM: {rom_path}")
    print(f"Speed: {speed}x")
    print("=" * 60)
    print("\nControls:")
    print("  Q / ESC - Quit")
    print("  SPACE   - Pause/Resume")
    print("  +/-     - Speed up/down")
    print("=" * 60)

    # Load model first to check its expected observation space
    print("\nLoading model...")
    if agent == "recurrent":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(model_path))
    else:
        model = PPO.load(str(model_path))
    expected_shape = model.observation_space.shape
    print(f"Model loaded! Expects observation shape: {expected_shape}")

    # Determine if model expects downscaled (36x40) or full resolution (144x160)
    # Shape is (C, H, W) - check height dimension
    expected_height = expected_shape[1]
    downscale = expected_height < 100  # 36 vs 144
    print(f"Using downscale={downscale} (height={expected_height})")

    # Create environment with display
    def make_env():
        return PokemonRedEnv(
            rom_path=rom_path,
            init_state=init_state,
            headless=False,  # Enable display
            action_freq=24,
            max_steps=max_steps,
            downscale=downscale,
        )

    env = DummyVecEnv([make_env])

    # If model expects stacked frames (channels > 3), add frame stacking
    if expected_shape[0] > 3:
        n_stack = expected_shape[0] // 3
        print(f"Adding frame stacking with n_stack={n_stack}")
        env = VecFrameStack(env, n_stack=n_stack)

    # Apply VecTransposeImage to match training env (HWC -> CHW)
    env = VecTransposeImage(env)

    # Setup pygame display
    pygame.init()
    scale = 4
    screen = pygame.display.set_mode((160 * scale, 144 * scale))
    pygame.display.set_caption("Pokemon Red - RL Agent")
    clock = pygame.time.Clock()

    # Get the underlying PyBoy instance for rendering
    # Navigate through wrappers: VecTransposeImage -> VecFrameStack -> DummyVecEnv -> PokemonRedEnv
    def get_underlying_env(vec_env):
        """Unwrap to get the actual PokemonRedEnv."""
        while hasattr(vec_env, 'venv'):
            vec_env = vec_env.venv
        return vec_env.envs[0]

    underlying_env = get_underlying_env(env)

    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    paused = False
    running = True
    lstm_states = None  # For RecurrentPPO
    # Use stochastic for recurrent to prevent getting stuck in loops
    use_deterministic = agent != "recurrent"

    print(f"\nStarting playback... (deterministic={use_deterministic})\n")

    try:
        while running and step < max_steps:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        speed = min(speed * 2, 16)
                        print(f"Speed: {speed}x")
                    elif event.key == pygame.K_MINUS:
                        speed = max(speed / 2, 0.25)
                        print(f"Speed: {speed}x")

            if paused:
                clock.tick(30)
                continue

            # Get action from model (pass lstm_states for RecurrentPPO)
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=use_deterministic)

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step += 1

            # Render to pygame
            frame = underlying_env.emulator.get_screen()
            # frame is (144, 160, 3) RGB
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, (160 * scale, 144 * scale))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Status update every 100 steps
            if step % 100 == 0:
                info_dict = info[0] if info else {}
                badges = info_dict.get("badges", 0)
                pos = info_dict.get("position", (0, 0))
                print(f"Step {step:5d} | Reward: {total_reward:8.1f} | Badges: {badges} | Pos: {pos}")

            # Control speed
            if speed > 0:
                clock.tick(60 * speed)

            if done[0]:
                print(f"\nEpisode finished at step {step}")
                print(f"Total reward: {total_reward:.1f}")
                obs = env.reset()
                total_reward = 0
                lstm_states = None  # Reset LSTM states for new episode

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        env.close()
        pygame.quit()

    print("\n" + "=" * 60)
    print("Playback ended")
    print(f"Total steps: {step}")
    print(f"Final reward: {total_reward:.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Watch trained RL agent play")
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to trained model (.zip)"
    )
    parser.add_argument(
        "--rom", type=Path, required=True, help="Path to Pokemon Red ROM"
    )
    parser.add_argument(
        "--state", type=Path, default=None, help="Initial save state"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed (default: 1.0)"
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Max steps to run"
    )
    parser.add_argument(
        "--agent", type=str, default="ppo", choices=["ppo", "recurrent"],
        help="Agent type: 'ppo' or 'recurrent' (PPO with LSTM)"
    )

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model not found at {args.model}")
        return

    if not args.rom.exists():
        print(f"Error: ROM not found at {args.rom}")
        return

    watch(
        model_path=args.model,
        rom_path=args.rom,
        init_state=args.state,
        speed=args.speed,
        max_steps=args.steps,
        agent=args.agent,
    )


if __name__ == "__main__":
    main()
