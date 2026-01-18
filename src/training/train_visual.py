"""Train RL agent with visual spectator display."""

import argparse
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.pokemon_env import PokemonRedEnv
from src.emulator.pyboy_wrapper import BUTTONS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GRAY = (128, 128, 128)


class VisualCallback(BaseCallback):
    """Callback that renders training to a pygame window."""

    def __init__(self, env: PokemonRedEnv, scale: int = 3):
        super().__init__()
        self.env = env
        self.scale = scale
        self.game_width = 160
        self.game_height = 144
        self.sidebar_width = 220

        # Initialize pygame
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.game_width * scale + self.sidebar_width, self.game_height * scale)
        )
        pygame.display.set_caption("Pokemon RL Training (Live)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

        self.running = True
        self.speed = 1  # 0=max, 1=normal
        self.episode_rewards = []
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        """Called after each step."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif event.key == pygame.K_0:
                    self.speed = 0
                elif event.key == pygame.K_1:
                    self.speed = 1
                elif event.key == pygame.K_2:
                    self.speed = 2

        # Track episode reward
        self.current_ep_reward += self.locals.get("rewards", [0])[0]

        # Check for episode end
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.current_ep_reward = 0.0

        # Render
        self._render()

        # Speed control
        if self.speed == 1:
            self.clock.tick(30)
        elif self.speed == 2:
            self.clock.tick(60)
        # speed=0 means no limit

        return self.running

    def _render(self):
        """Render game and stats."""
        self.display.fill(BLACK)

        # Get current observation from environment
        if hasattr(self.env, '_get_observation'):
            screen = self.env._get_observation()
        else:
            screen = np.zeros((144, 160, 3), dtype=np.uint8)

        # Draw game screen
        surface = pygame.surfarray.make_surface(screen.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            surface, (self.game_width * self.scale, self.game_height * self.scale)
        )
        self.display.blit(scaled, (0, 0))

        # Draw sidebar
        self._render_sidebar()

        pygame.display.flip()

    def _render_sidebar(self):
        """Render training stats sidebar."""
        x = self.game_width * self.scale + 10
        y = 10

        def text(s, color=WHITE, small=False):
            nonlocal y
            font = self.small_font if small else self.font
            surf = font.render(s, True, color)
            self.display.blit(surf, (x, y))
            y += 20 if small else 24

        text("RL TRAINING", GREEN)
        y += 5

        # Training stats
        text(f"Steps: {self.num_timesteps:,}")
        text(f"Episodes: {len(self.episode_rewards)}")

        # Rewards
        y += 5
        text("REWARDS", GREEN)
        text(f"Current: {self.current_ep_reward:.2f}")
        if self.episode_rewards:
            text(f"Last Ep: {self.episode_rewards[-1]:.2f}")
            avg = np.mean(self.episode_rewards[-10:])
            text(f"Avg(10): {avg:.2f}")

        # Game state from info
        y += 5
        text("GAME STATE", GREEN)
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        text(f"Badges: {info.get('badges', 0)}/8")
        text(f"Level: {info.get('total_level', 0)}")
        text(f"Maps: {info.get('maps_visited', 0)}")
        text(f"Coords: {info.get('coords_visited', 0)}")

        if info.get("in_battle"):
            text("IN BATTLE!", RED)

        # Last action
        y += 5
        actions = self.locals.get("actions", [0])
        action = int(actions[0]) if len(actions) > 0 else 0
        text(f"Action: {BUTTONS[action]}", BLUE)

        # Speed indicator
        y += 10
        speed_txt = ["MAX", "1x", "2x"][self.speed]
        text(f"Speed: {speed_txt}", GRAY, small=True)

        # Controls
        y = self.game_height * self.scale - 50
        text("Q=Quit 0/1/2=Speed", GRAY, small=True)

    def _on_training_end(self):
        """Clean up pygame."""
        pygame.quit()


def train_visual(
    rom_path: Path,
    init_state: Path | None = None,
    total_timesteps: int = 100_000,
    scale: int = 3,
    output_path: Path = Path("models/visual_trained.zip"),
) -> None:
    """
    Train PPO while displaying the game visually.
    """
    print("=" * 60)
    print("Pokemon RL Training (Visual)")
    print("=" * 60)
    print(f"ROM: {rom_path}")
    print(f"Timesteps: {total_timesteps:,}")
    print("Controls: Q=quit, 0=max speed, 1=normal, 2=fast")
    print("=" * 60)

    # Create environment
    env = PokemonRedEnv(
        rom_path=rom_path,
        init_state=init_state,
        headless=True,
        action_freq=36,
        max_steps=4096,
    )

    # Create callback for visualization
    callback = VisualCallback(env, scale=scale)

    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,  # Entropy bonus to prevent action collapse
        verbose=1,
    )

    print("Starting training...\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted!")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"\nModel saved to: {output_path}")

    if callback.episode_rewards:
        print(f"Episodes completed: {len(callback.episode_rewards)}")
        print(f"Best episode reward: {max(callback.episode_rewards):.2f}")
        print(f"Average reward: {np.mean(callback.episode_rewards):.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL with visual display")
    parser.add_argument("--rom", type=Path, required=True, help="ROM path")
    parser.add_argument("--state", type=Path, default=None, help="Initial state")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training steps")
    parser.add_argument("--scale", type=int, default=3, help="Display scale")
    parser.add_argument("--output", type=Path, default=Path("models/visual_trained.zip"))

    args = parser.parse_args()
    train_visual(args.rom, args.state, args.timesteps, args.scale, args.output)


if __name__ == "__main__":
    main()
