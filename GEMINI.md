# AutoGamePlayer Project Mandates

This document serves as the foundational memory for Gemini CLI regarding the AutoGamePlayer project's architecture, critical logic, and technical constraints.

Dry and solid focus on agnostic architecture with a huge focus on turn-key automation.

## 🚀 Core Architectural Mandates

### 1. Process Isolation & Graceful Shutdown
- **Constraint:** All child processes (Game Server, AI Player, Dashboard) MUST be started with `start_new_session=True` in `subprocess.Popen`.
- **Reason:** This isolates them from the terminal's `SIGINT` (Ctrl+C), allowing the parent process to orchestrate a surgical shutdown sequence: **AI Player (Save State) -> Dashboard -> Game Server**.

### 2. The "Bootstrap" Lifecycle (Slot 0)
- **Logic:** The first time an AI clears the intro (transitioning from `intro_map_ids` to the game world), it MUST save a checkpoint to **Slot 0**.
- **Mandate:** On every launch, the system MUST attempt to load Slot 0. If it exists, the AI starts directly in the overworld, bypassing the title screen and naming sequences.

### 3. Save State Strategy (MCTS-Lite)
- **Rolling Checkpoints:** The system MUST maintain a rolling ring of saves in **Slots 2-7**, saving every 50 steps.
- **Intelligent Rollback:** If the stagnation counter hits its limit (30 steps), the AI MUST load the **most recent rolling slot** saved before the stagnation started, rather than reverting to Slot 1.
- **Drift Mode:** Upon loading any state, the AI enters "Drift Mode" (increased randomness) for 30 steps to ensure it branches into a new timeline.

### 4. Stuck Detection & Stagnation Response
- **Spatial Stagnation:** A "stuck" state is defined by **20 consecutive steps** where the `map_id` and `(x, y)` coordinates remain identical, regardless of frame hash changes.
- **Reward Normalization:** The `RewardNormalizer` is the **single source of truth** for intrinsic vs extrinsic weighting, curiosity bursts, and stagnation penalties.
- **Mandatory Reset:** Every `RewardFunction` implementation MUST implement `reset()` to clear per-episode state (seen hashes, OCR keywords, stagnation counters). The environment MUST call `reset()` on all reward functions at the start of every episode to prevent inter-episode reward drift.
- **Timeline Branching:** If the stagnation counter reaches **30 steps**, the AI MUST force-load a recent rolling slot (or Slot 1 as fallback) and enter Drift Mode.
- **Visual Stagnation:** If visual/OCR progress is effectively zero for 5 steps, the AI is marked as stuck.
- **Loop Avoidance:** If the `CriticAgent` detects a state loop (same tile/OCR repeated > 3 times in 8 steps), it MUST trigger an immediate plan invalidation and record a `loop_warning` memory.

### 5. Strategic Planning & Execution
- **Plan Structure:** The `PlannerAgent` MUST output a structured JSON plan containing a `goal`, `steps`, and an `abort_condition`.
- **Cache Invalidation:** The planning cache MUST be invalidated whenever the AI is stuck (`is_stuck=True`) to force a fresh strategic goal from the LLM.
- **Episodic Warmup:** On startup, the `AgenticBrain` MUST "warm up" its episodic memory by loading the last 50 steps from the replay buffer.
- **Cost-Aware Pathfinding:** The `Pathfinder` MUST use Dijkstra-weighted A* where `impassable_score` from the world map acts as a movement cost (Standard=1, Collision Penalty=10x).
- **Macro Synthesis:** Any successful path found by the `Pathfinder` MUST be automatically promoted to a named skill (**`SKILL_PATH_TO_X_Y`**) in the `StrategyOptimizer` with high reliability.

### 6. Spatial & Procedural Memory
- **Global World Map:** Every unique `(map_id, x, y)` coordinate visited across all sessions MUST be persisted to the `explored_locations` table.
- **World Map Summary:** The planner MUST be provided with a summary of explored tiles for the current map.
- **Hyper-Aggressive Collision:** If a movement button is pressed but RAM coordinates remain static, the system MUST immediately set the `impassable_score` for that tile to **1.0** (Hard Wall) in the world map.

### 7. Macro Learning & Matching (Self-TAS)
- **Gold Run Logic:** The `ReflectionAgent` MUST prioritize "Gold Runs"—sequences with the highest **reward-to-step density**.
- **Reward Speed Scoring:** Sequence mining uses the formula `score = (freq * mean_reward) / total_frames`. Efficiency is prioritized over raw reward.
- **Stagnation Penalty:** Macros that finish with a `vision_delta < 0.05` OR result in zero state delta (Map, Coords, and Hash identical) MUST be immediately penalized in reliability.
- **Hash-Based Execution:** The `MacroAwareBrain` uses exact state hash matching for 0-latency TAS execution.
- **Jittering:** Genetic optimization MUST use "Jittering" (randomly +/- 2 frames) to find frame-perfect paths.

### 8. Input & Concurrency
- **Sequential Input Queue:** All inputs MUST be processed via a single-threaded queue in `GameSession`. NEVER spawn threads per button press.
- **Macro Batching:** Macros MUST be sent as a single batch from Client to Server to prevent network-induced drift between steps.
- **Database Concurrency:** All LTM interactions MUST use **WAL (Write-Ahead Logging)** mode and `get_db_connection` with a 30s timeout to prevent locking between play and dreaming processes.

## 🎮 Pokémon Red Logic & Heuristics

### 1. Map Definitions
- **Intro Maps:** `[0, 39]` (Map 0 = Title/Naming, Map 39 = Red's Bedroom).
- **Overworld Start:** Any map not in the intro list.

### 2. Critical Memory Addresses (RAM)
- `0xD35E`: Current Map ID
- `0xD362`: Player X tile
- `0xD361`: Player Y tile
- `0xD16B`: Badge count (0-8)
- `0xD163`: Party size
- `0xD31C`: Player money (3 bytes BCD)
- `0xFFD6`: `sprites_enabled` (HRAM). 1 in Overworld, 0 in Menus/Intro. **CRITICAL for UI_LOCKED logic.**
- `0xCFC4`: `menu_id_ram` (WRAM). 1 if menu open, 0 if closed.

### 3. UI State & Perception Logic
- **UI_LOCKED Prevention:** If the overworld is explorable but the status is `LOCKED`, verify `sprites_enabled (0xFFD6)` is `1`. 
- **Vision Warmup:** DINOv2 (`facebook/dinov2-small`) requires ~40-60s for initialization.

## 🛠️ Technical Implementation Guards
- **Syntax Verification:** Every time code is modified, the system MUST run `uv run ruff check . --fix` to ensure syntax integrity and style consistency. If errors remain, they MUST be resolved manually before completing the task.
- **Robust Schema Parsing:** All Pydantic models related to configuration (`GameConfig`, `GameProfile`) and core data (`GameState`, `Action`, `Observation`) MUST include `model_config = {"extra": "allow"}`. This prevents system crashes when schemas evolve or extra metadata is enriched by sub-agents.
- **Robust MCP Communication:** 
    - All MCP tools returning complex objects MUST return a `dict` (via `model_dump()`) and use `Any` as the return type hint. This bypasses fragile FastMCP internal auto-serialization which can return error strings on failure.
    - All MCP tools MUST accept an optional `_seq: int = None` parameter to support the client's idempotency tracking.
- **Telemetry & Attribution:**
    - Every `ActionProposal` MUST include a `source` string (or `solver_name` in metadata).
    - `SessionOrchestrator` MUST log reward attribution by solver to allow the dashboard to render attribution pie charts.
- **Scope Safety:** Always initialize method-level variables (like `macro_json`, `description`) at the top of async methods. NEVER use `if 'var' in locals()` as it is fragile in asynchronous contexts.
- **Multimodal Default:** Always include game screenshots in the LLM prompt for all models unless they are explicitly known to be text-only.
- **Unified Dialogue/OCR:** The system MUST always run OCR for a frame if memory-based flags (`is_dialogue`, `is_menu_open`, or `battle_state > 0`) are set, even if `include_ocr` is globally False. Text content is critical for decision-making in these states.
- **Robust Parsing:** Use `extract_json_from_llm_response` (regex-based) for all LLM outputs.
- **Output Priming:** All agent prompts MUST include: *"Respond ONLY with a valid JSON object. Do not include markdown formatting, preamble, or conversational text."*
- **Path Consistency:** Always use `settings.models_dir` for artifacts in `data/models/`.

## 🏃 Runner & CLI Commands
- **Dashboard Toggle:** Use `--no-dashboard` with `nexus` to skip Streamlit UI.
- **Skill Inspection:** Use `uv run nexus --list-skills` to dump the macro registry.

## 🤖 Autonomous Macro Synthesis
- **Gold Run Discovery:** `ReflectionAgent` MUST analyze high-reward sequences and distill them into `SKILL_` macros in the SQL database.
- **Macro Interception:** The `MacroAwareBrain` MUST check for exact `vision_hash` matches before invoking the LLM to provide 0-latency TAS execution.
- **Genetic Refinement:** The `StrategyOptimizer` MUST periodically evolve the macro population using mutations (jittering frame counts) and crossover to find the fastest possible paths.
- **Reward Speed Priority:** Mined sequences MUST be scored using `(frequency * reward) / total_frames` to prioritize efficiency.
