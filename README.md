# Universal Game Nexus (v3.0)

An Agnostic Neuro-Symbolic RL Engine with Vision Encoding, SLAM Mapping, and Plug-and-Play Solver Pipelines.

## 🚀 Key Features

- **Agnostic Architecture**: Zero Python changes required to target new games. Environment and reward pipelines are driven entirely by YAML configurations.
- **Pluggable Solver Pipeline**: Hot-swap between **Agent57 (Meme Core)**, **LLM Reasoning**, and **Evolutionary** solvers. Includes a **BrainSolverAdapter** for backward compatibility with legacy agents.
- **DreamerV3 Training**: Recurrent world model training with **Latent Unrolling** for synthetic imagination rollouts and actor-critic optimization.
- **Hierarchical Reward System**: Centralized `RewardNormalizer` with support for per-game `reward_schedule` to balance intrinsic exploration vs extrinsic goals.
- **SLAM (Simultaneous Localization and Mapping)**: Dynamically constructs a 2D occupancy grid, tracking "Walkable" vs "Blocked" tiles with a composite SQL index for performance.
- **Robust Infrastructure**: MCP tool call idempotency via sequence numbers, SQLite WAL mode for concurrency, and priority-ordered replay buffer eviction.
- **Real-Time Interactive Dashboard**: 
    - **Spatial Belief**: Live SLAM heatmaps and collision traces.
    - **Cognitive Visualization**: 2D PCA projection of LSTM hidden states.
    - **Curiosity Monitoring**: Real-time RND error and novelty traces.
    - **Reward Attribution**: Pie charts showing which solvers are driving performance.

## 🧠 Brains & Solvers

- **Agentic Solver**: Flagship solver with long-term memory (RAG) and multi-model strategy planning.
- **MCTS Solver**: MuZero-inspired tree search using Representation, Dynamics, and Prediction networks.
- **Evolution Brain**: Tripartite Neural Network evolved via Genetic Algorithms.
- **LLM Meta-Reasoner**: symbolic executive override that synthesizes sub-solver proposals into high-level actions.

## 🛠️ Configuration (YAML)

Target any game by defining its RAM layout and UI signals in a profile:
```yaml
name: "pokemon_red"
emulator: "pyboy"
rewards:
  - type: "exploration"
    category: "intrinsic"
    params: { threshold: 0.1 }
  - type: "ram_tracker"
    category: "extrinsic"
    params: { variables: { "badges": { "multiplier": 100 } } }
reward_schedule:
  intrinsic: 1.0
  extrinsic: 2.5
```

## 🏃 Usage

### Standard Launch
```bash
uv run nexus --config configs/pokemon_red_agentic.yaml
```

### Dry Run (Validation)
Validate your YAML, ROM paths, and model weights without starting steps:
```bash
uv run nexus --config configs/pokemon_red.yaml --dry-run
```
