# Pokemon AI Agents

Train AI agents to play Pokemon Red using reinforcement learning or LLM-based decision making.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

A local-first platform for building and running AI agents on Pokemon Red. Supports multiple agent types:

- **RL Agents**
  - **PPO** - Standard PPO with CNN policy and frame stacking
  - **RecurrentPPO** - PPO with LSTM memory for temporal reasoning
- **LLM Agents** - Make strategic decisions using Claude, GPT, or AWS Bedrock models

**Features:**
- **Parallel RL training** - Up to 24 environments running simultaneously
- **Apple Silicon optimized** - MPS acceleration for M1/M2/M3 Macs
- **LLM support** - Anthropic Claude, OpenAI GPT, AWS Bedrock
- **Rich reward shaping** - Badges, exploration, battles, leveling, catching Pokemon
- **Persistent exploration** - Tiles tracked across episodes to encourage new discoveries
- **Live spectate mode** - Watch RL training in real-time with `--spectate`
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

# Train with PPO and live spectate window
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/ready_to_play.state --spectate

# Train with RecurrentPPO (LSTM memory)
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/ready_to_play.state --agent recurrent

# Train RecurrentPPO with spectate
python -m src.training.train_rl --rom roms/pokemon_red.gb --state saves/ready_to_play.state --agent recurrent --spectate
```

### Watch a Trained Model

```bash
# Watch PPO model
python -m src.training.watch_rl --rom roms/pokemon_red.gb --model models/ppo_*/final_model.zip

# Watch RecurrentPPO model
python -m src.training.watch_rl --rom roms/pokemon_red.gb --model models/recurrent_*/final_model.zip --agent recurrent
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
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  PyBoy Emulator │────▶│  Gymnasium Env  │────▶│  RL Agent           │
│  (Pokemon Red)  │◀────│  (Rewards/Obs)  │◀────│  PPO / RecurrentPPO │
└─────────────────┘     └─────────────────┘     └─────────────────────┘
```

1. **Emulator**: PyBoy runs Pokemon Red, exposing screen pixels and RAM
2. **Environment**: Gymnasium wrapper calculates rewards from game state
3. **Agent**: PPO (CNN + frame stacking) or RecurrentPPO (CNN + LSTM) learns to maximize rewards

### Reward System

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
| Healing | +0.2 | Scaled by HP healed |
| Explore new tile | +0.005 | Persists across episodes |
| Money gained | +0.01 | Per $100 |
| Pokemon faints | -1.0 | Per fainted party member |
| Whiteout | -5.0 | All Pokemon fainted |
| Run away | 0 | No reward for fleeing |

### Dynamic Episode Length

Episodes get longer as the agent progresses, giving more time to explore further:

```
max_steps = 10,240 + (completed_events × 2,048)
```

| Events Completed | Max Steps |
|------------------|-----------|
| 0 | 10,240 |
| 5 | 20,480 |
| 10 | 30,720 |

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
  --agent ppo \            # ppo or recurrent
  --envs 24 \              # Parallel environments (max 16 for recurrent)
  --timesteps 10000000 \   # Total training steps
  --lr 0.00025 \           # Learning rate
  --spectate               # Live visualization window
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

## Requirements

**Core:**
- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [Gymnasium](https://gymnasium.farama.org/) - Environment interface
- [Pygame](https://www.pygame.org/) - Visualization

**RL Training:**
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PPO algorithm
- [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) - RecurrentPPO
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
