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
from PIL import Image

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
        downscale: bool = False,  # Downscale observations for faster RL training
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
            downscale: If True, downscale observations to 36x40 (16x smaller) for faster training
        """
        super().__init__()

        self._original_rom_path = Path(rom_path)
        self._original_init_state = Path(init_state) if init_state else None
        self.headless = headless
        self.action_freq = action_freq
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.speed = speed
        self.downscale = downscale

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
        # Observation shape: full (144, 160, 3) or downscaled (36, 40, 3)
        obs_shape = (36, 40, 3) if self.downscale else (144, 160, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
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

        # Diminishing returns tracking (prevents grinding same area)
        self._battles_per_map: dict[int, int] = {}

        # New reward tracking
        self._prev_money = 0
        self._prev_pokemon_seen = 0
        self._prev_item_count = 0
        self._prev_event_flags: set[int] = set()
        self._battle_start_party_hp = 0  # Party HP when battle started
        self._battle_start_turn = 0  # Turn count when battle started
        self._is_trainer_battle = False  # Whether current battle is trainer
        self._battle_won = False  # Whether we won the battle (defeated enemy or caught)

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
        # Persist visited maps/coords across episodes (don't reset)
        self._visited_maps.add(self._prev_map_id)
        x, y = self.game_state.get_player_position()
        self._visited_coords.add((self._prev_map_id, x, y))

        # Reset battle tracking
        self._prev_party_hp, _ = self.game_state.get_total_party_hp()
        self._prev_enemy_hp = 0
        self._was_in_battle = self.game_state.is_in_battle()
        self._prev_pokemon_caught = self.game_state.get_pokedex_owned_count()
        self._prev_party_alive = self.game_state.get_party_alive_count()
        self._battles_per_map = {}

        # Reset new reward tracking
        self._prev_money = self.game_state.get_money()
        self._prev_pokemon_seen = self.game_state.get_pokedex_seen_count()
        self._prev_item_count = self.game_state.get_item_count()
        self._prev_event_flags = self.game_state.get_event_flags()
        self._battle_start_party_hp = 0
        self._battle_start_turn = 0
        self._is_trainer_battle = False
        self._battle_won = False

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
        """Get the current screen as observation, optionally downscaled."""
        if self.emulator is None:
            shape = (36, 40, 3) if self.downscale else (144, 160, 3)
            return np.zeros(shape, dtype=np.uint8)

        screen = self.emulator.get_screen()  # (144, 160, 3)

        if self.downscale:
            # Downscale to 40x36 (width x height) for faster training
            # Using BILINEAR for speed, LANCZOS for quality
            img = Image.fromarray(screen)
            img = img.resize((40, 36), Image.BILINEAR)
            return np.array(img, dtype=np.uint8)

        return screen

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
            self._battle_won = True  # Catching counts as winning

        # ===================
        # BATTLE REWARDS (with diminishing returns to prevent grinding)
        # ===================
        map_id = self.game_state.get_map_id()
        battles_here = self._battles_per_map.get(map_id, 0)
        # Diminishing multiplier: 1.0 -> 0.5 -> 0.33 -> 0.25 -> ... (floors at 0.1)
        diminish = max(0.1, 1.0 / (1 + battles_here * 0.2))

        # Battle just started - record initial state
        if in_battle and not self._was_in_battle:
            self._battle_start_party_hp, _ = self.game_state.get_total_party_hp()
            self._battle_start_turn = self.game_state.get_battle_turn_count()
            self._is_trainer_battle = self.game_state.is_trainer_battle()
            self._battle_won = False  # Reset for new battle

        if in_battle:
            enemy_hp, enemy_max_hp = self.game_state.get_enemy_hp()

            # Reward for damaging enemy (scaled by damage dealt, with diminishing returns)
            if self._prev_enemy_hp > 0 and enemy_hp < self._prev_enemy_hp:
                damage_dealt = self._prev_enemy_hp - enemy_hp
                # Normalize by max HP to give similar rewards regardless of enemy
                if enemy_max_hp > 0:
                    reward += 0.5 * diminish * (damage_dealt / enemy_max_hp)

                # OHKO bonus: enemy went from full HP to 0 in one hit
                if self._prev_enemy_hp == enemy_max_hp and enemy_hp == 0:
                    reward += 0.5  # OHKO bonus

            # Reward for defeating enemy (enemy HP dropped to 0)
            if self._was_in_battle and self._prev_enemy_hp > 0 and enemy_hp == 0:
                reward += 1.0 * diminish  # Knocked out enemy (diminishing)
                self._battles_per_map[map_id] = battles_here + 1  # Track battle
                self._battle_won = True  # Mark battle as won

            self._prev_enemy_hp = enemy_hp

        # Battle ended - check outcome
        if self._was_in_battle and not in_battle:
            # Battle just ended
            party_alive = self.game_state.get_party_alive_count()
            current_party_hp, _ = self.game_state.get_total_party_hp()

            # Penalty if we lost Pokemon (fainted)
            if party_alive < self._prev_party_alive:
                fainted = self._prev_party_alive - party_alive
                reward -= 1.0 * fainted  # Penalty per fainted Pokemon

            # Whiteout detection (all Pokemon fainted)
            if party_alive == 0:
                reward -= 5.0  # Whiteout penalty

            # Only give victory bonuses if we actually won (not if we fled)
            if self._battle_won:
                # Trainer battle bonus (more valuable than wild)
                if self._is_trainer_battle:
                    reward += 2.0  # Trainer battle won bonus

                # No damage taken bonus
                if current_party_hp >= self._battle_start_party_hp and self._battle_start_party_hp > 0:
                    reward += 0.5  # Won without taking damage

                # Efficient battle bonus (fewer turns = better)
                battle_turns = self.game_state.get_battle_turn_count() - self._battle_start_turn
                if battle_turns <= 2:
                    reward += 0.2 * (3 - battle_turns)  # +0.4 for 1 turn, +0.2 for 2 turns

            self._prev_party_alive = party_alive
            self._prev_enemy_hp = 0  # Reset enemy HP tracking
            self._is_trainer_battle = False
            self._battle_won = False  # Reset for next battle

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

        self._prev_party_hp = current_party_hp

        # ===================
        # LEVELING REWARDS (with diminishing returns to prevent grinding)
        # ===================
        total_level = self.game_state.get_total_level()
        if total_level > self._prev_total_level:
            reward += 0.5 * diminish * (total_level - self._prev_total_level)
            self._prev_total_level = total_level

        # ===================
        # EXPLORATION REWARDS (boosted to encourage progression over grinding)
        # ===================
        # Note: map_id already defined above in battle rewards section
        if map_id not in self._visited_maps:
            self._visited_maps.add(map_id)
            reward += 5.0  # New map discovered (was 2.0)

        x, y = self.game_state.get_player_position()
        coord = (map_id, x, y)
        if coord not in self._visited_coords:
            self._visited_coords.add(coord)
            reward += 0.005  # Reward for new tiles

        # ===================
        # MONEY REWARDS
        # ===================
        current_money = self.game_state.get_money()
        if current_money > self._prev_money:
            # Money gained (from battles, selling items, etc.)
            money_gained = current_money - self._prev_money
            reward += 0.01 * (money_gained / 100)  # +0.01 per $100
        self._prev_money = current_money


        # ===================
        # ITEM REWARDS
        # ===================
        item_count = self.game_state.get_item_count()
        if item_count > self._prev_item_count:
            # Gained items (pickup, purchase, gift)
            reward += 0.5 * (item_count - self._prev_item_count)
        self._prev_item_count = item_count

        # ===================
        # EVENT FLAG REWARDS (story progress)
        # ===================
        current_flags = self.game_state.get_event_flags()
        new_flags = current_flags - self._prev_event_flags
        if new_flags:
            reward += 3.0 * len(new_flags)  # +3.0 per new event flag
            self._prev_event_flags = current_flags

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
            "badges_bitfield": self.game_state.get_badges(),  # Raw bitfield for individual badge checks
            "total_level": self.game_state.get_total_level(),
            "map_id": self.game_state.get_map_id(),
            "position": (x, y),
            "in_battle": self.game_state.is_in_battle(),
            "maps_visited": len(self._visited_maps),
            "coords_visited": len(self._visited_coords),
            "pokemon_caught": self.game_state.get_pokedex_owned_count(),
            "pokemon_seen": self.game_state.get_pokedex_seen_count(),
            "party_count": self.game_state.get_party_count(),  # Number of Pokemon in party
            "party_hp": party_hp,
            "party_max_hp": party_max_hp,
            "party_alive": self.game_state.get_party_alive_count(),
        }

        # Add battle info if in battle
        if info["in_battle"]:
            # Check if battle menu is ready (not in intro/animation)
            info["battle_menu_ready"] = self.game_state.is_battle_menu_ready()

            enemy_hp, enemy_max_hp = self.game_state.get_enemy_hp()
            info["enemy_hp"] = enemy_hp
            info["enemy_max_hp"] = enemy_max_hp
            info["enemy_level"] = self.game_state.get_enemy_level()

            # Always read Pokemon info during battle (prompt will validate if it's ready)
            player_pokemon = self.game_state.get_active_pokemon_info()
            enemy_pokemon = self.game_state.get_enemy_pokemon_info()
            info["player_pokemon"] = player_pokemon
            info["enemy_pokemon"] = enemy_pokemon

        # Add menu/dialogue state
        menu_state = self.game_state.get_menu_state()
        info["menu_active"] = self.game_state.is_in_menu_or_dialogue()
        info["has_menu_choice"] = self.game_state.has_menu_choice()
        info["menu_cursor"] = menu_state["current_menu_item"]
        info["menu_options"] = menu_state["max_menu_item"] + 1 if menu_state["max_menu_item"] > 0 else 0

        # Input detection
        info["waiting_for_input"] = self.game_state.is_waiting_for_input()

        # Environment type (indoors/outdoors/cave) and exits
        info["environment_type"] = self.game_state.get_environment_type()
        if info["environment_type"] != "outdoors":
            warps = self.game_state.get_warp_locations()
            info["exits"] = [(w["x"], w["y"]) for w in warps]
        else:
            info["exits"] = []

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
