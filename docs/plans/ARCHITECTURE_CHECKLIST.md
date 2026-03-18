# Agnostic Neuro-Symbolic RL Engine
## Architectural Blueprint & Full Implementation Checklist
**autogameplayer — v3.0 Target Architecture**

---

## Legend

| Tag | Meaning |
|-----|---------|
| `[ ]` | Standard checklist item |
| `[NEW]` | Added in this revision |
| `[UPDATED]` | Expanded or clarified item |
| `[CRITICAL]` | Blocking or high-risk — must not be skipped |

---

## Phase 1: Core Abstraction & Emulator Agnosticism [COMPLETED]
*SOLID/DRY — Decouple environment and reward pipelines from all game-specific logic.*

### Reward Pipeline
- [x] **Abstract RAM Rewards:** Replaced game-specific classes with `RAMTrackerReward`.
- `[CRITICAL]` [x] **Dialogue/OCR Reward Reset:** Implemented `reset()` across all reward functions.
- `[CRITICAL]` [x] **Exploration Reward Stagnation Counter Reset:** Fixed via `RewardFunction.reset()`.
- `[UPDATED]` [x] **Centralize Reward Normalization:** Created `RewardNormalizer` class.
- [x] **Remove Duplicate Reward Logic:** Logic delegated to `RewardNormalizer`.
- `[NEW]` [x] **Per-Game Reward Weight Scheduling:** Added `reward_schedule` to YAML.

### Emulator & Action Space Abstraction
- [x] **Abstract Emulator Capabilities:** Uses MCP `get_capabilities`.
- `[UPDATED]` [x] **UI Signal Extraction:** Moved dialogue/intro detection to YAML `ui_signals`.
- `[CRITICAL]` `[NEW]` [x] **Abstract the Done/Terminal Signal:** Defined `TerminalConditionConfig` in YAML.
- `[NEW]` [x] **Emulator Health-Check Contract:** Implemented `IEmulatorCore.health_check()`.
- `[NEW]` [x] **Action Space Type Config:** Supported via YAML.
- `[UPDATED]` [x] **Remove GameStage Pokemon Bias:** Renamed to generic stages (`COMBAT`, `EXPLORATION`, etc).

### Observation Pipeline
- [x] **Vision Backbone Abstraction:** Uses `settings.vision_model`.
- `[UPDATED]` [x] **Observation Pipeline Agnosticism:** Abstracted OCR behind `IOCRBackend`.
- `[CRITICAL]` `[NEW]` [x] **Async Vision Encoding:** Implemented using `ThreadPoolExecutor`.
- `[NEW]` [x] **LLM Vision Resolution:** Implemented downscaling for LLM prompt images.

### Data Model Agnosticism
- [x] **Agnostic Step Metadata:** Refactor `StepRecord` with `state_metadata`.
- [x] **Configurable Map Dimensions:** Added `map_id_dim` to constructor and auxiliary losses.
- `[CRITICAL]` `[NEW]` [x] **R2D2 Hidden State Fields:** Added `lstm_h`, `lstm_c` to `StepRecord`.

### Code Quality / DRY
- [x] **Unify Vector Lookups:** Unified into `_query_action_by_vector` in `memory.py`.
- [x] **Centralize Magic Constants:** Moved to `settings.py`.
- `[CRITICAL]` `[NEW]` [x] **SQLite Connection Pool:** Implemented via `threading.local()`.
- `[CRITICAL]` `[NEW]` [x] **State Tracker Memory Leak:** Implemented max-size eviction and reset.
- `[NEW]` [x] **SLAM Index:** Added composite index on `(map_id, x, y)`.

---

## Phase 2: The Pluggable Solver Pipeline [COMPLETED]
*Composite Pattern — Infrastructure for mixing solvers and routing to LLMs.*

### Contracts & Interfaces
- [x] **Define `ActionProposal`:** Standardized dataclass implemented.
- `[UPDATED]` [x] **Define `BaseSolver`:** ABC with step/episode hooks.
- `[CRITICAL]` `[UPDATED]` [x] **Define `AgentContext`:** Typed dataclass implemented.
- [x] **Solver Registry:** Created in `solvers/registry.py`.

### Solver Implementations
- [x] **Implement `RandomSolver`:** Completed.
- `[UPDATED]` [x] **Implement `EnsembleSolver`:** Completed.
- `[UPDATED]` [x] **Implement `RouterSolver`:** Completed.
- `[NEW]` [x] **Implement `GreedyEpsilonSolver`:** Completed.

### Refactoring Existing Systems
- `[UPDATED]` [x] **Solver Decorator for `MacroAwareBrain`:** Refactored as `MacroSolverDecorator`.
- `[UPDATED]` [x] **Encapsulate Tree Search:** Refactored into `TreeSearchSolver`.
- [x] **Retire the Brain ABC:** Main CLI migrated to `SessionOrchestrator` and `BaseSolver`.
- `[NEW]` [x] **`SessionOrchestrator`:** Extracted core loop from runner.
- `[NEW]` [x] **Solver Composition Logging:** Logged via telemetry.

---

## Phase 3: The Strict Agent57 Implementation [COMPLETED]
*Upgrade from MuZero-style to a recurrent, multi-armed Agent57 core.*

### Meta-Controller (Bandit)
- `[UPDATED]` [x] **Implement Sliding-Window UCB:** Completed in `src/autogameplayer/core/bandit.py`.
- [x] **Bandit Persistence:** Stats persisted to SQLite.
- `[UPDATED]` [x] **Async Bandit Safety:** Uses `asyncio.Lock`.
- [x] **Arm Warm-Start:** Implemented.
- `[CRITICAL]` `[NEW]` [x] **Value Replay Invalidation on Arm Change:** Implemented `invalidate_stale_arms`.

### Neural Network Architecture
- [x] **UVFA:** Conditioned on $\beta$ and $\gamma$.
- [x] **Split Value Heads:** Extrinsic and Intrinsic value heads.
- [x] **Target Networks:** Soft/hard updates implemented in trainer.
- [x] **Recurrent Core (R2D2):** Upgraded to `LSTMCell` with sequence burn-in.
- `[NEW]` [x] **Network Input Normalization:** Added `LayerNorm` to torso.

### Curiosity & Novelty
- [x] **Episodic Memory (k-NN):** Implemented in `curiosity.py`.
- `[CRITICAL]` `[UPDATED]` [x] **Novelty Combination Formula:** Multiplicative Agent57 formula with clipping.

### Training Stability
- [x] **Transformed Bellman Operator:** implemented $h(x)$ and $h^{-1}(x)$.
- `[UPDATED]` [x] **Sequence Burn-in & Hidden State Seeding:** Supports 40-step burn-in.
- `[UPDATED]` [x] **Prioritized Experience Replay:** Added `priority` column and sampling logic.
- `[CRITICAL]` `[UPDATED]` [x] **Episode Boundary Signaling:** Proper handling of `done` states.

---

## Phase 4: The LLM Meta-Reasoner [COMPLETED]
*Wrap AI models in a symbolic reasoning layer as a configurable executive override.*

- [x] **Implement `LLMMetaSolver`:** High-level executive solver.
- `[CRITICAL]` `[UPDATED]` [x] **Confidence Gating:** Threshold-based triggers implemented.
- [x] **KnowledgeBase RAG Integration:** RAGsnippets injected into prompts.
- `[CRITICAL]` `[UPDATED]` [x] **Unified Brain Merger:** Reflector logic integrated.
- [x] **Session Memory Deduplication:** Unified write paths.
- `[NEW]` [x] **LLM Response Caching:** Hash-based response caching implemented.
- `[CRITICAL]` `[NEW]` [x] **Async LLM Timeout:** 15s timeout with advisor fallback.
- `[NEW]` [x] **Structured Output Enforcement:** JSON schema parsing.
- `[NEW]` [x] **Token Budget Tracking:** Logged to `llm_costs.json`.
- `[UPDATED]` [x] **LLM Call Budget:** `max_calls_per_episode` implemented.

---

## Phase 5: GitOps Configuration & Factories [COMPLETED]
*Enable architectural changes entirely via YAML.*

- [x] **Pydantic Solver Schemas:** Recursive config schemas.
- [x] **YAML Integration:** `agent_pipeline` standard.
- `[CRITICAL]` `[UPDATED]` [x] **The Factory Maker:** `SolverFactory` with dependency support.
- [x] **Runtime Hot-Swap:** Config watcher triggers live rebuild.
- [x] **Solver State Migration:** `transfer_state()` preserves memory.
- [x] **Config Schema Validation on Startup:** Validates Pydantic schemas.
- `[CRITICAL]` `[NEW]` [x] **Robust YAML Parsing:** Enabled `extra="allow"` across all config models (`GameConfig`, `GameProfile`, `Checkpoint`, etc.) AND core models (`GameState`, `Observation`, `Action`) to prevent crashes on evolving schemas or data enrichment.
- `[CRITICAL]` `[NEW]` [x] **Robust MCP Serialization:** Return `dict` (via `model_dump()`) from tools returning complex Pydantic models to prevent server-side validation strings from crashing the client.
- [x] **Config Robustness Tests:** Added `tests/test_config_robustness.py` to verify schema flexibility across config and core models.
- `[UPDATED]` [x] **Solver Schema Validation CLI:** `--validate-config` flag.
- `[NEW]` [x] **Dry-Run Mode:** `--dry-run` flag.
- `[NEW]` [x] **Environment Variable Override:** `AGP_` prefix support.

---

## Phase 6: Observability & Testing [COMPLETED]
*Ensure the system is debuggable, measurable, and regression-safe.*

### Dashboard & Metrics
- [x] **Dashboard Solver Panels:** Bandit stats and attribution charts.
- `[NEW]` [x] **LSTM Hidden State Visualization:** 2D PCA projection in dashboard.
- `[NEW]` [x] **RND Error History:** Curiosity trace in dashboard.
- `[UPDATED]` [x] **Structured Logging:** Standardized logging levels.

### Testing
- `[UPDATED]` [x] **Unit Tests for Core Abstractions:** Added `test_core_logic.py`.
- `[CRITICAL]` `[NEW]` [x] **Headless Episode Smoke Test:** Added `test_smoke.py`.
- `[NEW]` [x] **Bandit Convergence Test:** Implemented.
- `[NEW]` [x] **Dead-Letter Queue:** failing batches saved to disk.

---

## Phase 7: Infrastructure & Deployment [COMPLETED]
*Make the system operable long-term.*

- [x] **SQLite WAL Mode:** Enabled for concurrency.
- `[CRITICAL]` `[UPDATED]` [x] **Replay Buffer Eviction Policy:** Priority-ordered pruning.
- `[CRITICAL]` `[NEW]` [x] **MCP Tool Call Idempotency:** Sequence numbers implemented.
- `[NEW]` [x] **MPS-Optimized Kernels:** Full `device="mps"` support for Apple Silicon.
- `[CRITICAL]` `[NEW]` [x] **Graceful Shutdown Handler:** model saving and DB flushing on exit.

---

## Phase 8: Curriculum & Skill Transfer [COMPLETED]
*Adaptive curricula and vision-based skill matching.*

- `[NEW]` [x] **Abstract Curriculum Condition:** `ConditionEvaluator` implemented.
- `[CRITICAL]` `[NEW]` [x] **Curriculum Regression Detector:** Rollback support.
- `[NEW]` [x] **Curriculum Dashboard Panel:** Stage tracking tab.
- `[NEW]` [x] **`SkillSolver`:** First-class skill execution.
- `[NEW]` [x] **Game-Agnostic Skill Matching:** Vision embedding similarity matching.

---

## Phase 9: The OTR Reward Layer [COMPLETED]
*Trajectory alignment with expert demonstrations.*

- `[NEW]` [x] **OTR Demonstration Parser:** TAS to latent parser.
- `[CRITICAL]` `[NEW]` [x] **Sinkhorn Reward Generator:** Sinkhorn-Knopp math utility.
- [x] **Dense Reward Mapping:** Exponential distance reward.
- `[NEW]` [x] **Windowed Alignment:** Sliding comparison window.
- `[NEW]` [x] **Hindsight OTR:** SLAM-based window retargeting.
