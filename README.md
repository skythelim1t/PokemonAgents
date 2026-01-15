# Pokemon RL Agent

Train AI agents to play Pokemon Red using reinforcement learning.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

A local-first platform for training RL agents on Pokemon Red. Uses PPO with a CNN policy to learn directly from screen pixels, with rewards based on game progress (badges, exploration, battles).

**Features:**
- **Parallel training** - 24 environments running simultaneously
- **Apple Silicon optimized** - MPS acceleration for M1/M2/M3 Macs
- **Rich reward shaping** - Badges, exploration, battles, leveling, catching Pokemon
- **Live visualization** - Watch training in real-time with Pygame
- **TensorBoard integration** - Monitor training progress

## Quick Start

### Prerequisites

- Python 3.10+
- Pokemon Red ROM (`.gb` file) - not included, you must provide your own
- macOS, Linux, or Windows

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pokemon-rl-agent.git
cd pokemon-rl-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Setup

1. Place your Pokemon Red ROM in the `roms/` directory:
   ```
   roms/pokemon_red.gb
   ```

2. Create an initial save state (optional but recommended - skips intro):
   ```bash
   python -m src.utils.state_manager --rom roms/pokemon_red.gb
   ```
   Play until you're ready to start training, then save the state.

### Training

```bash
# Train with PPO (headless, fast)
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/ready_to_play.state

# Train with live visualization (slower)
python -m src.training.train_visual --rom roms/pokemon_red.gb --state saves/ready_to_play.state
```

### Watch a Trained Model

```bash
python -m src.training.run_model \
  --rom roms/pokemon_red.gb \
  --state saves/ready_to_play.state \
  --model models/ppo_*/final_model.zip
```

## How It Works

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PyBoy Emulator │────▶│  Gymnasium Env  │────▶│  PPO Agent      │
│  (Pokemon Red)  │◀────│  (Rewards/Obs)  │◀────│  (CNN Policy)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Emulator**: PyBoy runs Pokemon Red, exposing screen pixels and RAM
2. **Environment**: Gymnasium wrapper calculates rewards from game state
3. **Agent**: PPO with CNN learns to maximize rewards over time

### Reward System

| Event | Reward |
|-------|--------|
| Earn badge | +10.0 |
| Catch Pokemon | +5.0 |
| Discover new map | +2.0 |
| Defeat enemy | +1.0 |
| Level up | +0.5 |
| Explore new tile | +0.005 |
| Pokemon faints | -1.0 |

### Training Performance

On Apple M3 Max with 24 parallel environments:

| Device | Speed | 10M steps |
|--------|-------|-----------|
| CPU | ~130 FPS | ~18 hours |
| MPS (GPU) | ~900 FPS | ~3 hours |

## Monitoring

```bash
# View training metrics in TensorBoard
tensorboard --logdir models/ppo_*/logs
```

Then open http://localhost:6006

## Project Structure

```
pokemon-rl-agent/
├── src/
│   ├── agents/           # Agent implementations
│   ├── emulator/         # PyBoy wrapper, memory reading
│   ├── environment/      # Gymnasium environment
│   ├── training/         # Training scripts
│   └── ui/               # Spectator UI
├── models/               # Trained models (output)
├── roms/                 # ROM files (not included)
└── saves/                # Save states
```

## Configuration

Training parameters can be adjusted via command line:

```bash
python -m src.training.train_rl \
  --rom roms/pokemon_red.gb \
  --state saves/ready_to_play.state \
  --envs 24 \              # Parallel environments
  --timesteps 10000000 \   # Total training steps
  --lr 0.00025             # Learning rate
```

## Requirements

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - Environment interface
- [Pygame](https://www.pygame.org/) - Visualization
- [PyTorch](https://pytorch.org/) - Neural networks

## Legal Notice

This project does not include any Nintendo ROMs or copyrighted material. You must provide your own legally obtained Pokemon Red ROM file.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) - Inspiration and reference
- [PyBoy](https://github.com/Baekalfen/PyBoy) - Excellent Game Boy emulator with Python API
