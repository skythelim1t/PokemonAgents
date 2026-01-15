"""Gymnasium environment for Pokemon Red."""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.emulator.memory_map import GameState
from src.emulator.pyboy_wrapper import BUTTONS, EmulatorWrapper


class PokemonRedEnv(gym.Env):
    """
    Gymnasium environment for Pokemon Red.

    Observation: RGB screen image (144, 160, 3)
    Action: Discrete(8) - one of the 8 Game Boy buttons
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: Path | str,
        init_state: Path | str | None = None,
        headless: bool = True,
        action_freq: int = 36,  # ~0.6 sec per action at 60fps, enough for tile movement
        max_steps: int = 2048,
        render_mode: str | None = None,
        speed: int = 0,
    ) -> None:
        """
        Initialize the Pokemon Red environment.

        Args:
            rom_path: Path to the Pokemon Red ROM file
            init_state: Optional path to an initial save state (to skip intro)
            headless: Run without display window
            action_freq: Frames to run per action (hold button + release)
            max_steps: Maximum steps per episode
            render_mode: 'human' for display, 'rgb_array' for array output
            speed: Emulation speed (0=unlimited, 1=normal, 2=2x, etc.)
        """
        super().__init__()

        self._original_rom_path = Path(rom_path)
        self._original_init_state = Path(init_state) if init_state else None
        self.headless = headless
        self.action_freq = action_freq
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.speed = speed

        # Create isolated temp directory for this env instance (avoids file contention)
        self._temp_dir = Path(tempfile.mkdtemp(prefix=f"pokemon_env_{uuid.uuid4().hex[:8]}_"))
        self.rom_path = self._temp_dir / self._original_rom_path.name
        shutil.copy(self._original_rom_path, self.rom_path)
        if self._original_init_state:
            self.init_state = self._temp_dir / self._original_init_state.name
            shutil.copy(self._original_init_state, self.init_state)
        else:
            self.init_state = None

        # Will be initialized in reset()
        self.emulator: EmulatorWrapper | None = None
        self.game_state: GameState | None = None
        self.step_count = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(BUTTONS))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(144, 160, 3), dtype=np.uint8
        )

        # Tracking for rewards
        self._prev_badges = 0
        self._prev_total_level = 0
        self._prev_map_id = 0
        self._visited_maps: set[int] = set()
        self._visited_coords: set[tuple[int, int, int]] = set()  # (map, x, y)

        # Battle and HP tracking
        self._prev_party_hp = 0
        self._prev_enemy_hp = 0
        self._was_in_battle = False
        self._prev_pokemon_caught = 0
        self._prev_party_alive = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Close existing emulator if any
        if self.emulator is not None:
            self.emulator.close()

        # Create new emulator instance
        self.emulator = EmulatorWrapper(
            self.rom_path,
            headless=self.headless,
            speed=self.speed,
        )
        self.game_state = GameState(self.emulator)

        # Load initial state if provided
        if self.init_state and self.init_state.exists():
            self.emulator.load_state(self.init_state)
            # Run frames to let game settle after load
            self.emulator.tick(60)
        else:
            # Run a few frames to let the game initialize
            self.emulator.tick(60)

        # Reset tracking
        self.step_count = 0
        self._prev_badges = self.game_state.get_badge_count()
        self._prev_total_level = self.game_state.get_total_level()
        self._prev_map_id = self.game_state.get_map_id()
        self._visited_maps = {self._prev_map_id}
        x, y = self.game_state.get_player_position()
        self._visited_coords = {(self._prev_map_id, x, y)}

        # Reset battle tracking
        self._prev_party_hp, _ = self.game_state.get_total_party_hp()
        self._prev_enemy_hp = 0
        self._was_in_battle = self.game_state.is_in_battle()
        self._prev_pokemon_caught = self.game_state.get_pokedex_owned_count()
        self._prev_party_alive = self.game_state.get_party_alive_count()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action: Index of the button to press (0-7)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.emulator is None or self.game_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Execute the action
        button = BUTTONS[action]
        self._run_action(button)

        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = False  # Pokemon Red doesn't really "end"
        truncated = self.step_count >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _run_action(self, button: str) -> None:
        """Run a button press action for action_freq frames."""
        if self.emulator is None:
            return

        # Pokemon Red needs ~16 frames for a tile movement
        # Hold button briefly, then wait for the action to complete
        self.emulator.pyboy.button_press(button)
        self.emulator.tick(8)  # Hold for 8 frames
        self.emulator.pyboy.button_release(button)
        self.emulator.tick(self.action_freq - 8)  # Wait remaining frames

    def _get_observation(self) -> NDArray[np.uint8]:
        """Get the current screen as observation."""
        if self.emulator is None:
            return np.zeros((144, 160, 3), dtype=np.uint8)
        return self.emulator.get_screen()

    def _calculate_reward(self) -> float:
        """Calculate reward based on game progress."""
        if self.game_state is None:
            return 0.0

        reward = 0.0
        in_battle = self.game_state.is_in_battle()

        # ===================
        # BADGE REWARDS (highest priority)
        # ===================
        badges = self.game_state.get_badge_count()
        if badges > self._prev_badges:
            reward += 10.0 * (badges - self._prev_badges)
            self._prev_badges = badges

        # ===================
        # POKEMON CAUGHT REWARDS
        # ===================
        pokemon_caught = self.game_state.get_pokedex_owned_count()
        if pokemon_caught > self._prev_pokemon_caught:
            # Reward for catching new Pokemon
            reward += 5.0 * (pokemon_caught - self._prev_pokemon_caught)
            self._prev_pokemon_caught = pokemon_caught

        # ===================
        # BATTLE REWARDS
        # ===================
        if in_battle:
            enemy_hp, enemy_max_hp = self.game_state.get_enemy_hp()

            # Reward for damaging enemy (scaled by damage dealt)
            if self._prev_enemy_hp > 0 and enemy_hp < self._prev_enemy_hp:
                damage_dealt = self._prev_enemy_hp - enemy_hp
                # Normalize by max HP to give similar rewards regardless of enemy
                if enemy_max_hp > 0:
                    reward += 0.5 * (damage_dealt / enemy_max_hp)

            # Reward for defeating enemy (enemy HP dropped to 0)
            if self._was_in_battle and self._prev_enemy_hp > 0 and enemy_hp == 0:
                reward += 1.0  # Knocked out enemy

            self._prev_enemy_hp = enemy_hp

        # Battle ended - check outcome
        if self._was_in_battle and not in_battle:
            # Battle just ended
            party_alive = self.game_state.get_party_alive_count()

            # Penalty if we lost Pokemon (fainted)
            if party_alive < self._prev_party_alive:
                fainted = self._prev_party_alive - party_alive
                reward -= 1.0 * fainted  # Penalty per fainted Pokemon

            self._prev_party_alive = party_alive
            self._prev_enemy_hp = 0  # Reset enemy HP tracking

        # Track battle state
        self._was_in_battle = in_battle

        # ===================
        # HP/HEALING REWARDS
        # ===================
        current_party_hp, max_party_hp = self.game_state.get_total_party_hp()

        # Reward for healing (HP increased outside of battle)
        if not in_battle and current_party_hp > self._prev_party_hp:
            hp_healed = current_party_hp - self._prev_party_hp
            if max_party_hp > 0:
                reward += 0.2 * (hp_healed / max_party_hp)

        # Small penalty for losing HP (encourages avoiding damage)
        if current_party_hp < self._prev_party_hp:
            hp_lost = self._prev_party_hp - current_party_hp
            if max_party_hp > 0:
                reward -= 0.05 * (hp_lost / max_party_hp)

        self._prev_party_hp = current_party_hp

        # ===================
        # LEVELING REWARDS
        # ===================
        total_level = self.game_state.get_total_level()
        if total_level > self._prev_total_level:
            reward += 0.5 * (total_level - self._prev_total_level)
            self._prev_total_level = total_level

        # ===================
        # EXPLORATION REWARDS
        # ===================
        map_id = self.game_state.get_map_id()
        if map_id not in self._visited_maps:
            self._visited_maps.add(map_id)
            reward += 2.0  # New map discovered

        x, y = self.game_state.get_player_position()
        coord = (map_id, x, y)
        if coord not in self._visited_coords:
            self._visited_coords.add(coord)
            reward += 0.005  # Small reward for new tiles

        return reward

    def _get_info(self) -> dict[str, Any]:
        """Get additional info about the current state."""
        if self.game_state is None:
            return {}

        x, y = self.game_state.get_player_position()
        party_hp, party_max_hp = self.game_state.get_total_party_hp()

        info = {
            "step": self.step_count,
            "badges": self.game_state.get_badge_count(),
            "total_level": self.game_state.get_total_level(),
            "map_id": self.game_state.get_map_id(),
            "position": (x, y),
            "in_battle": self.game_state.is_in_battle(),
            "maps_visited": len(self._visited_maps),
            "coords_visited": len(self._visited_coords),
            "pokemon_caught": self.game_state.get_pokedex_owned_count(),
            "party_hp": party_hp,
            "party_max_hp": party_max_hp,
            "party_alive": self.game_state.get_party_alive_count(),
        }

        # Add battle info if in battle
        if info["in_battle"]:
            enemy_hp, enemy_max_hp = self.game_state.get_enemy_hp()
            info["enemy_hp"] = enemy_hp
            info["enemy_max_hp"] = enemy_max_hp
            info["enemy_level"] = self.game_state.get_enemy_level()

        return info

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            # For human rendering, the emulator window handles it
            # when headless=False
            pass
        return None

    def close(self) -> None:
        """Clean up resources."""
        if self.emulator is not None:
            self.emulator.close()
            self.emulator = None
        # Clean up temp directory
        if hasattr(self, "_temp_dir") and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
