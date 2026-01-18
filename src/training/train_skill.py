"""
Unified skill training script.
Trains individual RL skills for the hybrid agent.
"""
import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Try to import RecurrentPPO
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False

from src.environment.skill_envs import (
    NavigateEnv,
    BattleEnv,
    HealEnv,
    PCEnv,
)


SKILL_ENVS = {
    "navigate": NavigateEnv,
    "battle": BattleEnv,
    "heal": HealEnv,
    "pc_manage": PCEnv,
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_env(skill_name: str, config: dict, rank: int = 0):
    """Create environment factory for vectorized envs."""
    def _init():
        env_class = SKILL_ENVS[skill_name]
        env_kwargs = config.get("env_kwargs", {})
        env = env_class(
            rom_path=config["rom_path"],
            init_state=config.get("init_state"),
            **env_kwargs
        )
        return env
    return _init


def train_skill(config_path: str):
    """Train a skill based on config file."""
    config = load_config(config_path)

    skill_name = config["skill_name"]
    print(f"Training skill: {skill_name}")

    # Create output directory
    output_dir = Path(config.get("output_dir", f"models/skills/{skill_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vectorized environments
    n_envs = config.get("n_envs", 4)
    use_subproc = config.get("use_subproc", False)

    env_fns = [make_env(skill_name, config, i) for i in range(n_envs)]

    if use_subproc:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Create eval environment
    eval_env = DummyVecEnv([make_env(skill_name, config, 0)])

    # Algorithm selection
    algorithm = config.get("algorithm", "PPO")
    policy = config.get("policy", "CnnPolicy")

    # Training hyperparameters
    hyperparams = config.get("hyperparams", {})

    if algorithm == "RecurrentPPO" and HAS_RECURRENT:
        model = RecurrentPPO(
            policy=policy,
            env=vec_env,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
            **hyperparams
        )
    else:
        if algorithm == "RecurrentPPO" and not HAS_RECURRENT:
            print("Warning: RecurrentPPO not available, falling back to PPO")
        model = PPO(
            policy=policy,
            env=vec_env,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
            **hyperparams
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 10000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix=skill_name,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=config.get("eval_freq", 5000),
        n_eval_episodes=config.get("n_eval_episodes", 5),
        deterministic=True,
    )

    # Train
    total_timesteps = config.get("total_timesteps", 100_000)
    print(f"Training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(str(output_dir / "final_model"))
    print(f"Training complete. Models saved to {output_dir}")

    vec_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL skills for hybrid agent")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to skill config YAML file"
    )
    args = parser.parse_args()

    train_skill(args.config)


if __name__ == "__main__":
    main()
