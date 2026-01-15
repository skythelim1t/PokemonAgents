"""Train a reinforcement learning agent on Pokemon Red."""

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from src.environment.pokemon_env import PokemonRedEnv


def make_env(rom_path: Path, init_state: Path | None, max_steps: int):
    """Factory function to create environment instances."""
    def _init():
        return PokemonRedEnv(
            rom_path=rom_path,
            init_state=init_state,
            headless=True,
            action_freq=24,
            max_steps=max_steps,
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
    n_steps: int = 128,
    batch_size: int = 256,
    n_epochs: int = 4,
    gamma: float = 0.99,
    checkpoint_freq: int = 10_000,
    eval_freq: int = 25_000,
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
    """
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"ppo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Pokemon Red RL Training")
    print("=" * 60)
    print(f"ROM: {rom_path}")
    print(f"Initial state: {init_state}")
    print(f"Output directory: {run_dir}")
    print(f"Parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 60)

    # Create vectorized environment
    print("\nCreating training environments...")
    env = make_vec_env(
        make_env(rom_path, init_state, max_steps),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Optional: Stack 4 frames for temporal information
    # env = VecFrameStack(env, n_stack=4)

    # Create evaluation environment (single env, same wrapper type as training)
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        make_env(rom_path, init_state, max_steps),
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_pokemon",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=3,
        deterministic=True,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Create PPO agent
    # Use MPS (Apple Silicon GPU) if available, otherwise CPU
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Creating PPO agent on device: {device}")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
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
    )


if __name__ == "__main__":
    main()
