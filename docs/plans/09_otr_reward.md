# Phase 9: The OTR Reward Layer [CONSOLIDATED & COMPLETED]

## Objective
Implemented Optimal Transport Reward (OTR) to align agent behavior with expert demonstrations, solving the "Zero Knowledge" exploration problem via topological trajectory alignment.

## Achieved Architecture

### 1. Optimal Transport Math
- **Sinkhorn-Knopp Utility:** Implemented a manual Sinkhorn algorithm in `src/autogameplayer/utils/optimal_transport.py` to compute Wasserstein distance between agent and expert trajectories.
- **Cost Matrix:** Developed cosine-distance cost matrix computation using normalized vision latents.

### 2. Expert Demonstration Pipeline
- **TAS Parser:** Created `ExpertParser` in `src/autogameplayer/utils/expert_parser.py` which replays `.json` TAS recordings through the emulator to extract high-dimensional vision latents and SLAM coordinates.
- **Demonstration Storage:** Experts are saved as `.pt` files containing synchronized sequences of latents and spatial metadata.

### 3. OTR Reward Function
- **Trajectory Alignment:** Implemented `OTRReward` in `src/autogameplayer/rewards/otr.py`, rewarding the agent based on the negative exponential distance to the expert demonstration.
- **Windowed Comparison:** Uses a sliding window mechanism to compare local agent progress against a sub-segment of the expert demo, allowing for long-horizon alignment.
- **Hindsight OTR:** Integrated SLAM-based "Shortcut Detection". If the agent reaches a coordinate appearing later in the expert demo, the comparison window automatically "jumps" forward, preventing the agent from being penalized for finding faster paths.

### 4. Integration & Agnosticism
- **Registry Integration:** Registered the `otr` reward type, allowing it to be enabled via YAML config for any game with an available expert demonstration.
- **Universal Latents:** Uses the same `VisionEncoder` (DINOv2) for both expert parsing and live rewards, ensuring feature-space consistency.
