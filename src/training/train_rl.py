"""Train a reinforcement learning agent on Pokemon Red."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    is_vecenv_wrapped,
)

from src.environment.pokemon_env import PokemonRedEnv


class SpectateCallback(BaseCallback):
    """Callback that renders training progress in a pygame window."""

    def __init__(
        self,
        rom_path: Path,
        init_state: Path | None,
        max_steps: int,
        n_stack: int = 4,
        render_freq: int = 1,
        scale: int = 4,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.rom_path = rom_path
        self.init_state = init_state
        self.max_steps = max_steps
        self.n_stack = n_stack
        self.render_freq = render_freq
        self.scale = scale

        # Will be initialized on first call
        self.spectate_env = None
        self.screen = None
        self.clock = None
        self.obs = None
        self.episode_reward = 0
        self.episode_steps = 0

    def _init_spectate(self) -> None:
        """Initialize spectate environment and pygame."""
        # Create a visible environment (not in subprocess)
        def make_spectate_env():
            return PokemonRedEnv(
                rom_path=self.rom_path,
                init_state=self.init_state,
                headless=True,  # We'll render via pygame ourselves
                action_freq=24,
                max_steps=self.max_steps,
                downscale=True,
            )

        self.spectate_env = DummyVecEnv([make_spectate_env])
        self.spectate_env = VecFrameStack(self.spectate_env, n_stack=self.n_stack)
        self.spectate_env = VecTransposeImage(self.spectate_env)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((160 * self.scale, 144 * self.scale))
        pygame.display.set_caption("Pokemon Red - RL Training (Live)")
        self.clock = pygame.time.Clock()

        # Reset environment
        self.obs = self.spectate_env.reset()
        self.episode_reward = 0
        self.episode_steps = 0

    def _on_training_start(self) -> None:
        """Initialize on training start."""
        self._init_spectate()

    def _on_step(self) -> bool:
        """Called after each training step."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Stop training
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False

        # Run spectate env with current policy
        if self.n_calls % self.render_freq == 0:
            # Get action from current policy
            action, _ = self.model.predict(self.obs, deterministic=False)

            # Step environment
            self.obs, reward, done, info = self.spectate_env.step(action)
            self.episode_reward += reward[0]
            self.episode_steps += 1

            # Get frame from underlying environment and render
            underlying_env = self.spectate_env.venv.envs[0]
            if underlying_env.emulator is not None:
                frame = underlying_env.emulator.get_screen()
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                surface = pygame.transform.scale(
                    surface, (160 * self.scale, 144 * self.scale)
                )
                self.screen.blit(surface, (0, 0))

                # Draw stats overlay
                font = pygame.font.Font(None, 24)
                stats = [
                    f"Step: {self.num_timesteps:,}",
                    f"Ep Reward: {self.episode_reward:.1f}",
                    f"Ep Steps: {self.episode_steps}",
                ]
                for i, text in enumerate(stats):
                    surf = font.render(text, True, (255, 255, 255))
                    # Draw shadow for readability
                    shadow = font.render(text, True, (0, 0, 0))
                    self.screen.blit(shadow, (6, 6 + i * 20))
                    self.screen.blit(surf, (5, 5 + i * 20))

                pygame.display.flip()

            # Reset if episode done
            if done[0]:
                self.obs = self.spectate_env.reset()
                self.episode_reward = 0
                self.episode_steps = 0

            # Limit frame rate
            self.clock.tick(60)

        return True

    def _on_training_end(self) -> None:
        """Cleanup on training end."""
        if self.spectate_env is not None:
            self.spectate_env.close()
        pygame.quit()


def make_env(rom_path: Path, init_state: Path | None, max_steps: int):
    """Factory function to create environment instances."""
    def _init():
        return PokemonRedEnv(
            rom_path=rom_path,
            init_state=init_state,
            headless=True,
            action_freq=24,
            max_steps=max_steps,
            downscale=True,  # 36x40 observations for faster training
        )
    return _init


def train(
    rom_path: Path,
    init_state: Path | None = None,
    output_dir: Path = Path("models"),
    total_timesteps: int = 1_000_000,
    n_envs: int = 24,
    max_steps: int = 2048,
    learning_rate: float = 2.5e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 4,
    gamma: float = 0.997,
    checkpoint_freq: int = 10_000,
    eval_freq: int = 25_000,
    spectate: bool = False,
    agent: str = "ppo",
) -> None:
    """
    Train a PPO agent on Pokemon Red.

    Args:
        rom_path: Path to the ROM file
        init_state: Optional initial save state
        output_dir: Directory to save models and logs
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        max_steps: Max steps per episode
        learning_rate: Learning rate for PPO
        n_steps: Steps per environment before update
        batch_size: Minibatch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        spectate: Show a live pygame window during training
        agent: RL algorithm - 'ppo' or 'recurrent' (PPO with LSTM)
    """
    # Determine n_stack based on agent type
    # RecurrentPPO uses LSTM for memory, so less frame stacking needed
    n_stack = 1 if agent == "recurrent" else 4

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{agent}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Pokemon Red RL Training")
    print("=" * 60)
    print(f"Agent: {agent.upper()}")
    print(f"ROM: {rom_path}")
    print(f"Initial state: {init_state}")
    print(f"Output directory: {run_dir}")
    print(f"Parallel environments: {n_envs}")
    print(f"Frame stack: {n_stack}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 60)

    # Create vectorized environment
    # Use DummyVecEnv for recurrent (avoids multiprocessing issues with LSTM)
    # Use SubprocVecEnv for standard PPO (faster with parallel processes)
    print("\nCreating training environments...")
    if agent == "recurrent":
        # DummyVecEnv runs in single process - more stable for LSTM
        actual_n_envs = min(n_envs, 16)
        print(f"RecurrentPPO: Using {actual_n_envs} envs with DummyVecEnv")
        env = make_vec_env(
            make_env(rom_path, init_state, max_steps),
            n_envs=actual_n_envs,
            vec_env_cls=DummyVecEnv,
        )
    else:
        actual_n_envs = n_envs
        env = make_vec_env(
            make_env(rom_path, init_state, max_steps),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )

    # Stack frames for temporal information
    env = VecFrameStack(env, n_stack=n_stack)

    # Create evaluation environment (single env)
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        make_env(rom_path, init_state, max_steps),
        n_envs=1,
        vec_env_cls=DummyVecEnv,  # Always use DummyVecEnv for eval (single env)
    )
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)  # Must match training env
    # Apply VecTransposeImage to match what SB3 auto-applies to training env
    eval_env = VecTransposeImage(eval_env)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // actual_n_envs,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_pokemon",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # For RecurrentPPO, deterministic=False gives more representative eval
    # (deterministic can cause agent to get stuck in loops without exploration)
    eval_deterministic = agent != "recurrent"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=eval_freq // actual_n_envs,
        n_eval_episodes=3,
        deterministic=eval_deterministic,
    )

    callback_list = [checkpoint_callback, eval_callback]

    # Add spectate callback if enabled
    if spectate:
        spectate_callback = SpectateCallback(
            rom_path=rom_path,
            init_state=init_state,
            max_steps=max_steps,
            n_stack=n_stack,
            render_freq=1,
            scale=4,
        )
        callback_list.append(spectate_callback)
        print("Spectate mode: ENABLED (pygame window)")

    callbacks = CallbackList(callback_list)

    # Create RL agent
    # Use MPS (Apple Silicon GPU) if available, otherwise CPU
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if agent == "recurrent":
        from sb3_contrib import RecurrentPPO
        print(f"Creating RecurrentPPO agent on device: {device}")
        # batch_size must be divisible by actual_n_envs for RecurrentPPO
        recurrent_batch_size = 128  # 128 is divisible by 16
        model = RecurrentPPO(
            "CnnLstmPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=recurrent_batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=0.05,  # Entropy bonus for more exploration
            verbose=1,
            tensorboard_log=str(log_dir),
            device=device,
        )
    else:
        print(f"Creating PPO agent on device: {device}")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=0.05,  # Entropy bonus for more exploration
            verbose=1,
            tensorboard_log=str(log_dir),
            device=device,
        )

    print(f"\nPolicy architecture:")
    print(model.policy)

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save final model
    final_path = run_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")

    # Cleanup
    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Best model: {run_dir / 'best_model'}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"\nTo view logs: tensorboard --logdir {log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on Pokemon Red")
    parser.add_argument(
        "--rom", type=Path, required=True, help="Path to Pokemon Red ROM"
    )
    parser.add_argument(
        "--state", type=Path, default=None, help="Initial save state"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models"), help="Output directory"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--envs", type=int, default=24, help="Number of parallel environments"
    )
    parser.add_argument(
        "--max-steps", type=int, default=2048, help="Max steps per episode"
    )
    parser.add_argument(
        "--lr", type=float, default=2.5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=10_000, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--spectate", action="store_true", help="Show live pygame window during training"
    )
    parser.add_argument(
        "--agent", type=str, default="ppo", choices=["ppo", "recurrent"],
        help="RL algorithm: 'ppo' (standard) or 'recurrent' (PPO with LSTM memory)"
    )

    args = parser.parse_args()

    if not args.rom.exists():
        print(f"Error: ROM not found at {args.rom}")
        return

    train(
        rom_path=args.rom,
        init_state=args.state,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        checkpoint_freq=args.checkpoint_freq,
        spectate=args.spectate,
        agent=args.agent,
    )


if __name__ == "__main__":
    main()
