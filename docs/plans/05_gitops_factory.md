# Plan: Phase 5 — GitOps Configuration & Factories [COMPLETED]

## Objective
Enabled full architectural changes via YAML without Python edits. Established a central factory to build complex solver graphs and supported runtime hot-swapping.

## Achieved Architecture

### 1. Configuration & Schemas
- [x] **Pydantic Solver Schemas:** Implemented recursive `SolverConfig` in `src/autogameplayer/core/config_loader.py` supporting nested ensembles, routers, and decorators.
- [x] **YAML Integration:** Standardized `agent_pipeline` block in `GameConfig` for defining the solver graph.
- [x] **Schema Validation [CRITICAL]:** Integrated Pydantic validation into the startup sequence; actionable errors are printed before the emulator launches.
- [x] **Dry-Run Mode:** Added `--dry-run` and `--validate-config` flags to the CLI to validate YAML, ROM paths, and model weights without executing game steps.

### 2. Solver Factory
- [x] **Recursive Instantiation [CRITICAL]:** Implemented `SolverFactory` in `src/autogameplayer/solvers/factory.py` to build complex solver trees with proper dependency management.
- [x] **Runtime Hot-Swap:** Implemented a file watchdog in the CLI that monitors `config.yaml` and triggers a live solver rebuild via `SessionOrchestrator.hot_swap_solver()`.
- [x] **State Migration:** Added `BaseSolver.transfer_state()` to allow LSTM hidden states and bandit windows to persist across live configuration reloads.

### 3. Reliability & Versioning
- [x] **Config Migration:** Supported backward compatibility by providing default `SolverConfig` fallbacks when legacy `brain` fields are used.
- [x] **Env Var Overrides:** Configured Pydantic `Settings` to allow any field to be overridden via `AGP_<FIELD>` for Docker and CI/CD environments.
- [x] **Per-Source Metrics:** `SessionOrchestrator` and telemetry now track reward and decision attribution by specific solver sources within an ensemble.
