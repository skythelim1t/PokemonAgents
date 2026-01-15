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
│   │   ├── llm_agent.py         # LLM strategic agent
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── actions.py       # Strategic action definitions
│   │       ├── providers.py     # Anthropic/OpenAI/Bedrock
│   │       └── prompts.py       # Prompt templates
│   ├── executor/
│   │   ├── __init__.py
│   │   └── action_executor.py   # Strategic action → button translator
│   ├── environment/
│   │   ├── __init__.py
│   │   └── pokemon_env.py       # Gymnasium environment wrapper
│   ├── training/
│   │   ├── train_rl.py          # Headless PPO training (24 parallel envs)
│   │   ├── train_visual.py      # Training with live pygame display
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

Pokemon Red/Blue key RAM addresses:
- `0xD361`: Player X position (map)
- `0xD362`: Player Y position (map)
- `0xD35E`: Current map ID
- `0xD163`: Party count
- `0xD164-D273`: Party Pokemon data
- `0xD356`: Badges (bitfield)
- `0xD057`: In-battle flag

These should be documented in `configs/pokemon_red.yaml` or `src/emulator/memory_map.py`.

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

### Phase 3: Executor ⚠️ (Basic)
- [x] Strategic action definitions (StrategicAction enum)
- [x] Basic action executor (1:1 button mapping)
- [ ] Menu navigation (Pokemon menu, bag, PC)
- [ ] Overworld pathfinding
- [ ] Complex multi-button sequences

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
# Train PPO agent (headless, 24 parallel environments, uses MPS on Apple Silicon)
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/ready_to_play.state

# Train with live visualization (slower, 1 environment)
python -m src.training.train_visual --rom roms/pokemon_red.gb --state saves/ready_to_play.state

# Run a trained model
python -m src.training.run_model --rom roms/pokemon_red.gb --state saves/ready_to_play.state --model models/ppo_*/final_model.zip
```

### Training Options

```bash
python -m src.training.train_rl \
  --rom roms/pokemon_red.gb \
  --state saves/ready_to_play.state \
  --envs 24 \                    # Number of parallel environments (default: 24)
  --timesteps 10000000 \         # Total training steps (default: 1M)
  --lr 0.00025                   # Learning rate
```

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

The environment provides rewards for game progress:

| Event | Reward |
|-------|--------|
| New badge | +10.0 |
| Catch Pokemon | +5.0 |
| Discover new map | +2.0 |
| Defeat enemy Pokemon | +1.0 |
| Level up | +0.5 per level |
| Damage enemy | +0.5 × (damage/max_hp) |
| Heal HP | +0.2 × (healed/max_hp) |
| Explore new tile | +0.005 |
| Pokemon faints | -1.0 per faint |
| Take damage | -0.05 × (lost/max_hp) |

### Training Output

Models are saved to `models/ppo_YYYYMMDD_HHMMSS/`:
- `checkpoints/` - Periodic checkpoints
- `best_model/` - Best performing model (by eval reward)
- `final_model.zip` - Final model after training
- `logs/` - TensorBoard logs

## LLM Agent

Run an LLM-powered agent that makes strategic decisions.

### Providers

| Provider | Models | Auth |
|----------|--------|------|
| `anthropic` | claude-3-5-sonnet, claude-3-haiku | `ANTHROPIC_API_KEY` |
| `openai` | gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` |
| `bedrock` | Claude, Llama, Mistral via AWS | Default AWS credentials |

### Usage

```bash
# Install LLM dependencies
pip install -e ".[llm]"

# Run with Anthropic (default)
python -m src.platform --rom roms/pokemon_red.gb --agent llm --spectate

# Run with Bedrock
python -m src.platform --rom roms/pokemon_red.gb --agent llm --provider bedrock --model anthropic.claude-3-haiku-20240307-v1:0

# With vision (include screen in prompts)
python -m src.platform --rom roms/pokemon_red.gb --agent llm --vision --spectate
```

### Strategic Actions

The LLM chooses from high-level actions:
- `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT` - Navigation
- `PRESS_A`, `PRESS_B` - Interaction
- `USE_MOVE_1` - `USE_MOVE_4`, `FLEE` - Battle
- `OPEN_MENU`, `WAIT` - Misc

### Efficiency

- `--action-skip N` - Only call LLM every N frames (default: 10)
- Cache invalidation on battle state changes
- Retry with exponential backoff on API errors

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

Each parallel environment runs in a separate process (`SubprocVecEnv`) with its own copy of the ROM and save state files. This prevents file contention issues when multiple PyBoy instances try to access the same files simultaneously.

### Training Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  24 Parallel     │────▶│   CNN Policy     │────▶│  PPO Update  │
│  Environments    │◀────│  (on MPS/GPU)    │◀────│  (Gradients) │
│  (SubprocVecEnv) │     │                  │     │              │
└──────────────────┘     └──────────────────┘     └──────────────┘
         │
         ▼
┌──────────────────┐
│  Each env:       │
│  - Temp ROM copy │
│  - 24 frames/act │
│  - Screen → CNN  │
│  - RAM → rewards │
└──────────────────┘
```

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
