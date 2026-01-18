# Pokemon Agent Platform

## Project Overview

A local-first platform for building and running AI agents that play Pokemon ROM games. Supports multiple agent types including LLMs (strategic decision-making) and RL models (frame-by-frame control).

## Core Architecture

```
┌─────────────────────────────────────────────────┐
│  Emulator Layer (PyBoy)                         │
│  - Handles GB/GBC ROMs (Red, Blue, Yellow, etc.)│
│  - Exposes RAM, screen buffer, inputs           │
│  - Simple pip install, no compilation needed    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│  Agent Manager                                  │
│  - Swappable agent implementations              │
│  - Common observation/action interface          │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│  Spectator UI (Pygame)                          │
│  - Real-time game viewing                       │
│  - Agent state overlay                          │
│  - Speed controls, pause, hotkeys               │
└─────────────────────────────────────────────────┘
```

## Agent Types

### LLM Agents (Strategic)
- Make high-level decisions: "Go to Cerulean City", "Use Thunderbolt on Gyarados"
- Decisions translated to button sequences by an Executor layer
- Action space: `MOVE_TO`, `BATTLE_ACTION`, `CATCH`, `INTERACT`, etc.

### RL Agents (Tactical)
- Frame-by-frame or frame-skipped button outputs
- Raw action space: A, B, Up, Down, Left, Right, Start, Select
- Gymnasium-compatible environment interface
- **PPO**: Standard PPO with CNN policy and 4-frame stacking
- **RecurrentPPO**: PPO with LSTM memory (from sb3-contrib)

### Executor Layer
- Translates LLM strategic commands into button sequences
- Handles: menu navigation, pathfinding, battle execution, dialogue advancement
- Initially rule-based/scripted, can evolve to learned policies

## Tech Stack

- **Emulator**: PyBoy (Game Boy / Game Boy Color emulator)
- **RL Framework**: Gymnasium for environment interface
- **UI**: Pygame for local spectating (PyBoy also supports SDL2)
- **LLM Integration**: Abstract interface, implementations for OpenAI/Anthropic/local models
- **Target Game**: Pokemon Red (GB) - architecture supports Red/Blue/Yellow/Crystal

## Key Design Decisions

1. **Local-first**: No cloud dependencies for core functionality
2. **Agent-agnostic**: Common interface allows swapping LLM/RL/hybrid agents
3. **Spectate mode**: Toggle between headless fast execution and real-time viewing
4. **ROM-agnostic goal**: Start with Pokemon Red, architecture supports other GB/GBC Pokemon games

## Directory Structure

```
pokemon-agent-platform/
├── CLAUDE.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── platform.py              # CLI entry point (random/LLM agents)
│   ├── emulator/
│   │   ├── __init__.py
│   │   ├── pyboy_wrapper.py     # PyBoy abstraction
│   │   └── memory_map.py        # RAM addresses for game state
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract Agent interface
│   │   ├── random_agent.py      # Baseline random agent
│   │   ├── ppo_agent.py         # PPO model wrapper (inference)
│   │   ├── llm_agent.py         # LLM strategic agent (legacy)
│   │   ├── tool_agent.py        # Tool-based LLM agent (recommended)
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── actions.py       # Strategic action definitions
│   │       ├── providers.py     # Anthropic/OpenAI/Bedrock
│   │       ├── prompts.py       # Prompt templates
│   │       ├── tools.py         # Tool definitions & executor
│   │       ├── knowledge.py     # Persistent knowledge base
│   │       ├── menu_navigator.py    # Menu navigation with cursor tracking
│   │       ├── battle_navigator.py  # Battle menu navigation
│   │       └── walkability.py   # Tile walkability detection
│   ├── executor/
│   │   ├── __init__.py
│   │   └── action_executor.py   # Strategic action → button translator
│   ├── environment/
│   │   ├── __init__.py
│   │   └── pokemon_env.py       # Gymnasium environment wrapper
│   ├── training/
│   │   ├── train_rl.py          # RL training (PPO/RecurrentPPO, --spectate option)
│   │   ├── watch_rl.py          # Watch trained model play
│   │   ├── train_visual.py      # Training with live pygame display (legacy)
│   │   └── run_model.py         # Run trained model for evaluation
│   ├── ui/
│   │   ├── __init__.py
│   │   └── spectator.py         # Pygame spectator UI
│   └── utils/
│       └── state_manager.py     # Save state creation/inspection
├── models/                      # Trained models output directory
├── roms/                        # .gitignore'd, user provides ROMs
└── saves/                       # Save states
```

## Code Style

- Python 3.10+
- Type hints on all function signatures
- Docstrings for public methods
- Use `pathlib.Path` over string paths
- Prefer dataclasses or Pydantic for structured data
- Keep emulator-specific code isolated in `emulator/`

## Memory Map Notes

Pokemon Red/Blue key RAM addresses (see `src/emulator/memory_map.py` for full list):

**Player State:**
- `0xD361`: Player Y position (map)
- `0xD362`: Player X position (map)
- `0xD35E`: Current map ID
- `0xD163`: Party count
- `0xD356`: Badges (bitfield)

**Battle State:**
- `0xD057`: In-battle flag
- `0xCC2B`: Battle menu cursor (FIGHT/BAG/POKEMON/RUN)
- `0xCC2E`: Move menu cursor (0-3)
- `0xCFE6`: Enemy HP (2 bytes)

**Menu/Text State:**
- `0xCC26`: Current menu cursor position
- `0xCC28`: Max menu items - 1
- `0xCF93`: Text box ID (>0 = menu/dialogue active)
- `0xCFC4`: Text delay counter (0 = text done)

**Environment Detection:**
- `0xD527`: Tileset ID (determines indoor/outdoor/cave)
- `0xD4E1`: Number of warps on current map
- `0xD4E2`: Warp entries (4 bytes each: Y, X, dest_warp, dest_map)

**Tileset Classifications:**
- Outdoor: 0-11 (towns, routes)
- Indoor: 12-14, 17-25, 41-54, 67 (buildings)
- Cave: 15, 40, 46

## Development Phases

### Phase 1: Foundation (POC) ✅
- [x] Install PyBoy and verify Pokemon Red ROM loads
- [x] Create basic Gymnasium environment wrapper
- [x] Random agent that mashes buttons
- [x] Spectator UI with speed toggle
- [x] Initial save state (post-parcel delivery)

### Phase 2: Game State ✅
- [x] RAM reading for player position, party, badges
- [x] HP/level/stats memory reading
- [x] Battle state detection (in_battle, enemy HP, enemy level)
- [x] Pokedex counts (owned/seen)
- [x] Save state management

### Phase 3: Executor ✅
- [x] Strategic action definitions (StrategicAction enum)
- [x] Basic action executor (1:1 button mapping)
- [x] Menu navigation (MenuNavigator with cursor tracking)
- [x] Battle menu navigation (BattleMenuNavigator)
- [x] Dynamic environment detection (indoor/outdoor/cave via tileset)
- [x] Warp/exit detection for building navigation

### Phase 4: Agents ✅
- [x] PPO agent with CNN policy (stable-baselines3)
- [x] Full training pipeline with parallel environments
- [x] Multi-component reward shaping
- [x] LLM agent with strategic action space
- [x] Multi-provider support (Anthropic, OpenAI, Bedrock)

## Running the Platform

```bash
# Install dependencies
pip install -e .

# Run random agent with spectator mode (no learning, just watching)
python -m src.platform --rom roms/pokemon_red.gb --spectate --state saves/ready_to_play.state

# Run headless (fast, no display)
python -m src.platform --rom roms/pokemon_red.gb --headless
```

## RL Training

### Quick Start

```bash
# Activate virtual environment first
source .venv/bin/activate

# Train PPO agent (headless, 24 parallel environments, uses MPS on Apple Silicon)
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/route1_start.state

# Train PPO with live spectate window
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/route1_start.state --spectate

# Train RecurrentPPO (LSTM memory, 16 parallel environments)
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/route1_start.state --agent recurrent

# Train RecurrentPPO with spectate
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/route1_start.state --agent recurrent --spectate

# Watch a trained PPO model
python -m src.training.watch_rl --rom roms/pokemon_red.gb --model models/ppo_*/final_model.zip

# Watch a trained RecurrentPPO model
python -m src.training.watch_rl --rom roms/pokemon_red.gb --model models/recurrent_*/final_model.zip --agent recurrent
```

### Long Training Runs (tmux + caffeinate)

For overnight or multi-day training on macOS:

```bash
# Run in tmux with caffeinate to prevent sleep
tmux new -s pokemon "caffeinate -is bash -c 'source .venv/bin/activate && python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/route1_start.state --timesteps 10000000'"

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t pokemon
# Kill session: tmux kill-session -t pokemon

# Scroll up in tmux: Ctrl+B, then [ (use arrows/PgUp, q to exit)
```

### Training Options

```bash
python -m src.training.train_rl \
  --rom roms/pokemon_red.gb \
  --state saves/route1_start.state \
  --agent ppo \                  # ppo or recurrent (default: ppo)
  --envs 24 \                    # Number of parallel environments (default: 24, max 16 for recurrent)
  --timesteps 10000000 \         # Total training steps (default: 1M)
  --lr 0.00025 \                 # Learning rate
  --checkpoint-freq 10000 \      # Save checkpoint every N steps
  --spectate                     # Live pygame window during training
```

### Agent Comparison

| Feature | PPO | RecurrentPPO |
|---------|-----|--------------|
| Policy | CnnPolicy | CnnLstmPolicy |
| Temporal context | Frame stacking (4 frames) | LSTM memory |
| Parallel envs | 24 (SubprocVecEnv) | 16 (DummyVecEnv) |
| Batch size | 256 | 128 |
| n_steps | 2048 | 2048 |
| gamma | 0.997 | 0.997 |
| Best for | Fast training | Complex sequences |

### Training Duration Guide

| Timesteps | Duration (~24 envs) | Expected Learning |
|-----------|---------------------|-------------------|
| 1M | ~20 min | Basic sanity check |
| 10M | ~3 hours | Basic movement patterns |
| 50-100M | ~15-30 hours | Purposeful navigation, battle wins |
| 500M-1B | Days | Meaningful game progress |

### Hardware Optimization

- **Apple Silicon (M1/M2/M3)**: Automatically uses MPS (Metal Performance Shaders) for GPU acceleration
- **Parallel environments**: Default 24 envs, adjust based on CPU cores
  - 8-core: use 8-16 envs
  - 16-core: use 16-32 envs
- **RAM usage**: ~100MB per environment

### Monitoring Training

```bash
# TensorBoard (graphs of rewards, loss, etc.)
tensorboard --logdir models/ppo_*/logs
```

Key metrics to watch:
- **Episode reward**: Should trend upward over time
- **entropy_loss**: Slowly decreasing = agent becoming more decisive
- **explained_variance**: Closer to 1.0 = better value predictions

### Reward System

The environment provides rewards for game progress. **Exploration is persistent across episodes** to encourage discovering new areas.

| Event | Reward | Notes |
|-------|--------|-------|
| Earn badge | +10.0 | Per badge |
| Catch Pokemon | +5.0 | Per new Pokemon |
| Discover new map | +5.0 | Persists across episodes |
| Event flag triggered | +3.0 | Story progress |
| Trainer battle won | +2.0 | Bonus on top of battle rewards |
| Defeat enemy | +1.0 | With diminishing returns per map |
| Level up | +0.5 | With diminishing returns |
| Item gained | +0.5 | Per item |
| No damage taken | +0.5 | Battle won without HP loss |
| OHKO bonus | +0.5 | One-hit KO |
| Damage enemy | +0.5 × (damage/max_hp) | Normalized |
| Healing | +0.2 × (healed/max_hp) | Outside battle |
| Efficient battle | +0.2 to +0.4 | +0.4 for 1-turn, +0.2 for 2-turn |
| Explore new tile | +0.005 | Persists across episodes |
| Money gained | +0.01 per $100 | |
| Pokemon faints | -1.0 | Per fainted party member |
| Whiteout | -5.0 | All Pokemon fainted |
| Run away | 0 | No reward for fleeing |

### Training Output

Models are saved to `models/{agent}_YYYYMMDD_HHMMSS/` (e.g., `models/ppo_20240115_143022/` or `models/recurrent_20240115_143022/`):
- `checkpoints/` - Periodic checkpoints
- `best_model/` - Best performing model (by eval reward)
- `final_model.zip` - Final model after training
- `logs/` - TensorBoard logs

## Save State Management

Available save states in `saves/`:

| State | Description |
|-------|-------------|
| `route1_start.state` | On Route 1, ready for wild battles |
| `post_parcel.state` | After delivering Oak's parcel |
| `pallet_town_start.state` | In Pallet Town |
| `ready_to_play.state` | General starting point |

### Creating Fresh Save States

If you get PyBoy version warnings, regenerate save states:

```bash
python -c "
from src.emulator.pyboy_wrapper import EmulatorWrapper
from pathlib import Path

emu = EmulatorWrapper('roms/pokemon_red.gb', headless=True)
emu.load_state(Path('saves/route1_start.state'))
emu.tick(60)
emu.save_state(Path('saves/route1_fresh.state'))
emu.close()
print('Created fresh save state')
"
```

## LLM Agents

### Tool Agent (Recommended)

The Tool Agent uses function calling to interact with the game through discrete tools.

```bash
# Run Tool Agent with Bedrock
python -m src.platform --rom roms/pokemon_red.gb --agent tool \
  --provider bedrock --model anthropic.claude-3-haiku-20240307-v1:0 \
  --spectate --log-conversation

# Run for specific steps
python -m src.platform --rom roms/pokemon_red.gb --agent tool \
  --provider bedrock --model anthropic.claude-3-haiku-20240307-v1:0 \
  --steps 100 --spectate
```

**Available Tools:**
- `navigator(x, y)` - Walk to screen coordinates
- `attack(slot)` - Use move in battle (1-4)
- `run_away()` - Flee from battle
- `check_pokemon()` - View party status
- `interact()` - Press A (talk/interact)
- `wait(condition)` - Wait for game state

**Features:**
- Walkability grid overlay (shows passable tiles)
- Dynamic environment detection (indoor/outdoor/cave)
- Exit coordinates shown when inside buildings
- Battle menu navigation with verification

### Strategic LLM Agent (Legacy)

Run an LLM-powered agent that makes strategic decisions based on what it sees on screen.

### How It Works

The LLM agent uses **vision-based navigation**:

1. **Screenshot Analysis**: The agent receives the current game screen
2. **Visual Interpretation**: LLM identifies paths, tall grass, obstacles, NPCs, doors
3. **Direction Choice**: Agent chooses EXPLORE_UP/DOWN/LEFT/RIGHT based on what it sees
4. **Action Execution**: Executor translates strategic action to button presses

This approach scales to any location without hardcoded coordinates.

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  Screenshot │────▶│  LLM Call   │────▶│  Strategic   │────▶│ Buttons │
│  + State    │     │  (Vision)   │     │  Action      │     │ A/B/D-pad│
└─────────────┘     └─────────────┘     └──────────────┘     └─────────┘
                           │
                    Sees: paths, grass,
                    obstacles, NPCs
```

### Providers

| Provider | Models | Auth |
|----------|--------|------|
| `anthropic` | claude-3-5-sonnet, claude-3-haiku | `ANTHROPIC_API_KEY` |
| `openai` | gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` |
| `bedrock` | Claude via AWS | Default AWS credentials |

### Usage

```bash
# Install LLM dependencies
pip install -e ".[llm]"

# Run with vision (recommended)
python -m src.platform --rom roms/pokemon_red.gb --agent llm --vision --spectate

# Run with Bedrock + vision
python -m src.platform --rom roms/pokemon_red.gb --agent llm \
  --provider bedrock --model anthropic.claude-3-haiku-20240307-v1:0 \
  --vision --spectate

# Log prompts and responses for debugging
python -m src.platform --rom roms/pokemon_red.gb --agent llm --vision --log-conversation

# Run for specific number of steps
python -m src.platform --rom roms/pokemon_red.gb --agent llm --vision --steps 500
```

### Strategic Actions

**Navigation** (vision-based):
- `EXPLORE_UP` - Walk north (toward top of screen)
- `EXPLORE_DOWN` - Walk south (toward bottom of screen)
- `EXPLORE_LEFT` - Walk west
- `EXPLORE_RIGHT` - Walk east

**Interaction**:
- `INTERACT` - Press A (talk to NPC, read sign, advance dialogue)
- `CANCEL` - Press B (close menu, go back)

**Battle**:
- `ATTACK_1` through `ATTACK_4` - Use moves
- `RUN_AWAY` - Flee from battle

### Visual Recognition

The LLM is prompted to identify:
- **Paths**: Light tan/beige ground (safe to walk)
- **Tall Grass**: Dark green patches (wild Pokemon encounters)
- **Trees/Walls**: Obstacles (blocked)
- **Water**: Blue tiles (blocked)
- **Ledges**: Can jump down but not climb up
- **Doorways**: Dark rectangles on buildings (enter)
- **NPCs**: Characters to talk to

### Action Executor

The executor translates strategic actions to button sequences:
- Handles walking animations (waits for movement to complete)
- Detects obstacles and blocked directions
- Manages battle menu navigation
- Auto-advances dialogue when appropriate

## Useful Commands

```bash
# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core + RL dependencies
pip install -e ".[rl]"

# Install LLM dependencies (optional)
pip install -e ".[llm]"

# Or install the project
pip install -e .
```

## Resources

- [PyBoy Documentation](https://github.com/Baekalfen/PyBoy)
- [PyBoy API Reference](https://baekalfen.github.io/PyBoy/index.html)
- [Pokemon Red RAM Map](https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PokemonRedExperiments (reference implementation)](https://github.com/PWhiddy/PokemonRedExperiments)

## Technical Notes

### Parallel Environment Isolation

Each parallel environment uses its own copy of the ROM and save state files. This prevents file contention issues.

- **PPO**: Uses `SubprocVecEnv` (separate processes, 24 envs)
- **RecurrentPPO**: Uses `DummyVecEnv` (single process, 16 envs) to avoid SDL2/multiprocessing conflicts with LSTM

### Training Architecture

**PPO (default):**
```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  24 Parallel     │────▶│   CNN Policy     │────▶│  PPO Update  │
│  Environments    │◀────│  (4-frame stack) │◀────│  (Gradients) │
│  (SubprocVecEnv) │     │  (on MPS/GPU)    │     │              │
└──────────────────┘     └──────────────────┘     └──────────────┘
```

**RecurrentPPO:**
```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  16 Parallel     │────▶│  CNN + LSTM      │────▶│  PPO Update  │
│  Environments    │◀────│  (memory state)  │◀────│  (Gradients) │
│  (DummyVecEnv)   │     │  (on MPS/GPU)    │     │              │
└──────────────────┘     └──────────────────┘     └──────────────┘
```

Each environment:
- Temp ROM copy (isolated)
- 24 frames per action
- Screen → CNN features
- RAM → rewards

### Performance Benchmarks (M3 Max, 24 envs)

| Device | FPS | Time for 10M steps |
|--------|-----|-------------------|
| CPU    | ~130 | ~18 hours |
| MPS    | ~900 | ~3 hours |

## Future: GBA Support

The architecture is designed to support swapping emulators. Future work could add:
- mGBA/pygba for GBA games (Fire Red, Emerald, etc.)
- Requires building mGBA with Python bindings (complex setup)
- Same agent interface, different emulator wrapper
