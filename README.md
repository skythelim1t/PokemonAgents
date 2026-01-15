# Pokemon AI Agents

Train AI agents to play Pokemon Red using reinforcement learning or LLM-based decision making.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

A local-first platform for building and running AI agents on Pokemon Red. Supports two agent types:

- **RL Agents (PPO)** - Learn from pixels using reinforcement learning
- **LLM Agents** - Make strategic decisions using Claude, GPT, or AWS Bedrock models

**Features:**
- **Parallel RL training** - 24 environments running simultaneously
- **Apple Silicon optimized** - MPS acceleration for M1/M2/M3 Macs
- **LLM support** - Anthropic Claude, OpenAI GPT, AWS Bedrock
- **Rich reward shaping** - Badges, exploration, battles, leveling, catching Pokemon
- **Live visualization** - Watch agents play in real-time with Pygame
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

## LLM Agent

Run an LLM-powered agent that makes strategic decisions using natural language reasoning.

### Setup

```bash
# Install LLM dependencies
pip install -e ".[llm]"

# Set API key (for Anthropic or OpenAI)
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
# AWS Bedrock uses default credential chain (no key needed if authenticated)
```

### Running LLM Agent

```bash
# Run with Anthropic Claude (default)
python -m src.platform --rom roms/pokemon_red.gb --state saves/ready_to_play.state --agent llm --spectate

# Run with OpenAI GPT
python -m src.platform --rom roms/pokemon_red.gb --agent llm --provider openai --model gpt-4o --spectate

# Run with AWS Bedrock
python -m src.platform --rom roms/pokemon_red.gb --agent llm --provider bedrock --model anthropic.claude-3-haiku-20240307-v1:0 --spectate

# With vision (include screen in prompts)
python -m src.platform --rom roms/pokemon_red.gb --agent llm --vision --spectate
```

### LLM Agent Options

| Option | Description |
|--------|-------------|
| `--provider` | `anthropic`, `openai`, or `bedrock` |
| `--model` | Model name/ID |
| `--vision` | Include screen images in prompts |
| `--action-skip` | Only call LLM every N frames (default: 10) |

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
│   ├── agents/
│   │   ├── llm_agent.py      # LLM strategic agent
│   │   ├── llm/              # LLM providers, prompts, actions
│   │   ├── ppo_agent.py      # PPO RL agent
│   │   └── random_agent.py   # Random baseline
│   ├── emulator/             # PyBoy wrapper, memory reading
│   ├── environment/          # Gymnasium environment
│   ├── executor/             # Strategic action → button translation
│   ├── training/             # RL training scripts
│   └── ui/                   # Spectator UI
├── models/                   # Trained models (output)
├── roms/                     # ROM files (not included)
└── saves/                    # Save states
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

**Core:**
- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [Gymnasium](https://gymnasium.farama.org/) - Environment interface
- [Pygame](https://www.pygame.org/) - Visualization

**RL Training:**
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [PyTorch](https://pytorch.org/) - Neural networks

**LLM Agent:**
- [Anthropic](https://github.com/anthropics/anthropic-sdk-python) - Claude API
- [OpenAI](https://github.com/openai/openai-python) - GPT API
- [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) - AWS Bedrock

## Legal Notice

This project does not include any Nintendo ROMs or copyrighted material. You must provide your own legally obtained Pokemon Red ROM file.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) - Inspiration and reference
- [PyBoy](https://github.com/Baekalfen/PyBoy) - Excellent Game Boy emulator with Python API
