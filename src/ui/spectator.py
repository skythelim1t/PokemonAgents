"""Pygame-based spectator UI for watching agents play."""

from pathlib import Path
from typing import Any

import numpy as np
import pygame
from numpy.typing import NDArray

from src.agents.base import BaseAgent
from src.agents.random_agent import RandomAgent
from src.environment.pokemon_env import PokemonRedEnv
from src.emulator.pyboy_wrapper import BUTTONS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GRAY = (128, 128, 128)


class SpectatorUI:
    """
    Pygame-based UI for watching agents play Pokemon.

    Features:
    - Real-time game display (scaled up)
    - Agent state overlay
    - Speed controls
    - Pause functionality
    """

    def __init__(
        self,
        env: PokemonRedEnv,
        agent: BaseAgent,
        scale: int = 3,
        target_fps: int = 60,
    ) -> None:
        """
        Initialize the spectator UI.

        Args:
            env: Pokemon environment
            agent: Agent to watch
            scale: Display scale factor (default 3x)
            target_fps: Target frames per second
        """
        self.env = env
        self.agent = agent
        self.scale = scale
        self.target_fps = target_fps

        # Game Boy screen is 160x144
        self.game_width = 160
        self.game_height = 144
        self.sidebar_width = 200

        # Calculate window dimensions
        self.display_width = (self.game_width * scale) + self.sidebar_width
        self.display_height = self.game_height * scale

        # State
        self.paused = False
        self.speed = 1  # 1 = normal, 2 = 2x, 0 = unlimited
        self.running = True
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action = 0

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Pokemon Agent Spectator")
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def _render_game(self, observation: NDArray[np.uint8]) -> None:
        """Render the game screen."""
        # Create surface from observation
        surface = pygame.surfarray.make_surface(observation.swapaxes(0, 1))

        # Scale up
        scaled = pygame.transform.scale(
            surface, (self.game_width * self.scale, self.game_height * self.scale)
        )

        # Blit to screen
        self.screen.blit(scaled, (0, 0))

    def _render_sidebar(self, info: dict[str, Any]) -> None:
        """Render the info sidebar."""
        x_offset = self.game_width * self.scale + 10
        y_offset = 10
        line_height = 22

        # Background
        sidebar_rect = pygame.Rect(
            self.game_width * self.scale, 0, self.sidebar_width, self.display_height
        )
        pygame.draw.rect(self.screen, (30, 30, 30), sidebar_rect)

        def draw_text(text: str, color: tuple[int, int, int] = WHITE) -> None:
            nonlocal y_offset
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += line_height

        def draw_small_text(text: str, color: tuple[int, int, int] = GRAY) -> None:
            nonlocal y_offset
            surface = self.small_font.render(text, True, color)
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += line_height - 4

        # Title
        draw_text("AGENT STATUS", GREEN)
        y_offset += 5

        # Step counter
        draw_text(f"Step: {self.step_count}")

        # Reward
        draw_text(f"Reward: {self.total_reward:.2f}")

        # Speed indicator
        speed_text = "MAX" if self.speed == 0 else f"{self.speed}x"
        draw_text(f"Speed: {speed_text}", BLUE if self.speed > 1 else WHITE)

        # Pause indicator
        if self.paused:
            draw_text("PAUSED", RED)
        else:
            y_offset += line_height

        y_offset += 10

        # Game state
        draw_text("GAME STATE", GREEN)
        y_offset += 5

        draw_text(f"Badges: {info.get('badges', 0)}/8")
        draw_text(f"Caught: {info.get('pokemon_caught', 0)}")
        draw_text(f"Level: {info.get('total_level', 0)}")

        # HP bar
        party_hp = info.get("party_hp", 0)
        party_max = info.get("party_max_hp", 1)
        hp_pct = party_hp / party_max if party_max > 0 else 0
        hp_color = GREEN if hp_pct > 0.5 else (255, 255, 0) if hp_pct > 0.2 else RED
        draw_text(f"HP: {party_hp}/{party_max}", hp_color)

        if info.get("in_battle", False):
            draw_text("IN BATTLE!", RED)
            enemy_hp = info.get("enemy_hp", 0)
            enemy_max = info.get("enemy_max_hp", 1)
            draw_small_text(f"Enemy: {enemy_hp}/{enemy_max} Lv{info.get('enemy_level', 0)}", RED)
        else:
            y_offset += line_height

        y_offset += 10

        # Exploration stats
        draw_text("EXPLORATION", GREEN)
        y_offset += 5
        draw_text(f"Maps: {info.get('maps_visited', 0)}")
        draw_text(f"Tiles: {info.get('coords_visited', 0)}")

        y_offset += 10

        # Last action
        draw_text("LAST ACTION", GREEN)
        y_offset += 5
        draw_text(f"{BUTTONS[self.last_action].upper()}", BLUE)

        # Controls help at bottom
        y_offset = self.display_height - 80
        draw_small_text("Controls:", GRAY)
        draw_small_text("SPACE - Pause", GRAY)
        draw_small_text("1/2/3/0 - Speed", GRAY)
        draw_small_text("Q/ESC - Quit", GRAY)

    def _handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.speed = 1
                elif event.key == pygame.K_2:
                    self.speed = 2
                elif event.key == pygame.K_3:
                    self.speed = 4
                elif event.key == pygame.K_0:
                    self.speed = 0  # Unlimited

    def run(self, max_steps: int = 10000) -> None:
        """
        Run the spectator UI.

        Args:
            max_steps: Maximum steps before stopping
        """
        obs, info = self.env.reset()
        self.agent.reset()
        self.step_count = 0
        self.total_reward = 0.0

        print("Spectator UI started. Press Q or ESC to quit.")
        print("Controls: SPACE=Pause, 1/2/3=Speed, 0=Max Speed")

        while self.running and self.step_count < max_steps:
            self._handle_events()

            if not self.paused:
                # Agent takes action
                action = self.agent.act(obs, info)
                self.last_action = action

                obs, reward, terminated, truncated, info = self.env.step(action)
                self.total_reward += reward
                self.step_count += 1

                if terminated or truncated:
                    print(f"Episode ended at step {self.step_count}")
                    obs, info = self.env.reset()
                    self.agent.reset()

            # Render
            self.screen.fill(BLACK)
            self._render_game(obs)
            self._render_sidebar(info)
            pygame.display.flip()

            # Control speed
            if self.speed > 0:
                self.clock.tick(self.target_fps * self.speed)

        pygame.quit()
        print(f"\nFinal stats: {self.step_count} steps, {self.total_reward:.2f} reward")


def run_spectator(
    rom_path: Path,
    init_state: Path | None = None,
    max_steps: int = 10000,
    scale: int = 3,
) -> None:
    """
    Launch the spectator UI with a random agent.

    Args:
        rom_path: Path to the ROM file
        init_state: Optional initial save state
        max_steps: Maximum steps to run
        scale: Display scale factor
    """
    env = PokemonRedEnv(
        rom_path=rom_path,
        init_state=init_state,
        headless=True,  # We handle display ourselves
        action_freq=24,
        max_steps=max_steps,
    )

    agent = RandomAgent(num_actions=env.action_space.n, seed=42)

    ui = SpectatorUI(env, agent, scale=scale)

    try:
        ui.run(max_steps=max_steps)
    finally:
        env.close()


def run_spectator_with_model(
    rom_path: Path,
    model_path: Path,
    init_state: Path | None = None,
    max_steps: int = 10000,
    scale: int = 3,
    deterministic: bool = True,
) -> None:
    """
    Launch the spectator UI with a trained PPO model.

    Args:
        rom_path: Path to the ROM file
        model_path: Path to trained model (.zip)
        init_state: Optional initial save state
        max_steps: Maximum steps to run
        scale: Display scale factor
        deterministic: Use deterministic actions
    """
    from src.agents.ppo_agent import PPOAgent

    env = PokemonRedEnv(
        rom_path=rom_path,
        init_state=init_state,
        headless=True,
        action_freq=24,
        max_steps=max_steps,
    )

    agent = PPOAgent(model_path, deterministic=deterministic)

    ui = SpectatorUI(env, agent, scale=scale)

    try:
        ui.run(max_steps=max_steps)
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch an agent play Pokemon")
    parser.add_argument("rom", type=Path, help="Path to ROM file")
    parser.add_argument("--state", type=Path, default=None, help="Initial save state")
    parser.add_argument("--model", type=Path, default=None, help="Trained model path")
    parser.add_argument("--scale", type=int, default=3, help="Display scale")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions")

    args = parser.parse_args()

    if args.model:
        run_spectator_with_model(
            rom_path=args.rom,
            model_path=args.model,
            init_state=args.state,
            max_steps=args.max_steps,
            scale=args.scale,
            deterministic=not args.stochastic,
        )
    else:
        run_spectator(
            rom_path=args.rom,
            init_state=args.state,
            max_steps=args.max_steps,
            scale=args.scale,
        )
