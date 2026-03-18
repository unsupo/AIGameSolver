# Phase 1: Core Foundation & Agnosticism [CONSOLIDATED & COMPLETED]

## Objective
Established a game-agnostic foundation where environment, reward, and perception pipelines are decoupled from specific game logic and driven entirely by YAML configurations.

## Achieved Architecture

### 1. Dynamic Reward Pipeline
- **Generalized Tracking:** Replaced game-specific classes with `RAMTrackerReward`, which parses addresses and multipliers from YAML.
- **Hierarchical Normalization:** Integrated `RewardNormalizer` as the single source of truth for reward capping, intrinsic/extrinsic weighting, and stagnation penalties.
- **Inter-Episode Isolation:** Implemented `RewardFunction.reset()` across all rewards (OCR, Dialogue, Exploration) to ensure a clean state at the start of every episode.
- **Reward Scheduling:** Added support for a `reward_schedule` in YAML to dynamically shift the balance between intrinsic curiosity and extrinsic goals.

### 2. Emulator & Perception Abstraction
- **Capability Discovery:** Workers now use the MCP `get_capabilities` tool to dynamically configure action spaces and button sets.
- **Generic UI Signals:** Hardcoded pixel coordinates for dialogue arrows and intro keywords moved to YAML `ui_signals` config block.
- **Async Vision Encoding:** Moved DINOv2 vector generation to a `ThreadPoolExecutor`, eliminating the bottleneck in the game loop.
- **Vision Resolution:** Implemented `llm_vision_resolution` scaling to reduce token costs for LLM-based solvers.
- **Terminal Logic:** Generalized game-over detection using `TerminalConditionConfig` (RAM thresholds, OCR regex, or step limits).

### 3. Agnostic Data Model & Storage
- **Flexible Metadata:** Refactored `StepRecord` to include `state_metadata` for non-grid games and separate fields for LSTM hidden states (`lstm_h`, `lstm_c`).
- **SQL Optimization:** Added a composite index on `(map_id, x, y)` in the `explored_locations` table to prevent full table scans during pathfinding.
- **Connection Stability:** Implemented thread-local SQLite connection pooling to ensure safe concurrent access from multiple workers and the dashboard.

### 4. Centralized Configuration
- **Magic Constants:** Moved all hardcoded thresholds (vision delta, cosine similarity, stagnation limits) into `settings.py`.
- **Generic Stages:** Renamed Pokemon-centric `GameStage` enums to universal equivalents (`COMBAT`, `EXPLORATION`, `UI`, `LOADING`).
