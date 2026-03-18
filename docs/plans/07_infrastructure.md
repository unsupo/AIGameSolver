# Plan: Phase 7 — Infrastructure & Deployment [COMPLETED]

## Objective
Enable robust, long-term operations on home servers/laptops with efficient memory management and process stability.

## 1. Database & Storage [DONE]
- [x] **SQLite WAL Mode:** Fix "database is locked" errors via `PRAGMA journal_mode=WAL`. Implemented in `get_db_connection`.
- [x] **Priority-Ordered Eviction [CRITICAL]:** Implement a `max_db_size_gb` equivalent (row count) where eviction is priority-ordered (lowest TD-error first) to preserve LSTM context. Implemented in `prune_replay_buffer`.
- [ ] **Knowledge Deduplication:** Add file hash checks to prevent re-ingesting the same files on startup.
- [x] **SLAM-Indexed Buffer:** Allow "spatial queries" (e.g., "Give me experiences within 5 tiles of current position"). Composite index on `(map_id, x, y)` implemented.

## 2. Worker & Process Stability [DONE]
- [x] **MCP Idempotency [CRITICAL]:** Add sequence numbers to tool calls to prevent double-inputs during reconnects. Implemented in `MCPClient` and `mcp_server`.
- [ ] **MCP Reconnect:** Add exponential backoff and retry for `ConnectionError` on tool calls.
- [ ] **Port Race Condition:** Use multiprocessing locks/Value to prevent worker port collisions.
- [ ] **Worker Crash Recovery [CRITICAL]:** Orchestrator should restart dead workers with the same `initial_slot`.
- [x] **Graceful Shutdown [CRITICAL]:** Implement `SIGTERM` handlers to flush replay buffer and save bandit/model states. Implemented in CLI and Single runners.

## 3. Memory & GPU [Silicon/M3 Optimized] [PARTIAL]
- [ ] **Vision Inference Server [CRITICAL]:** Move vision encoding to a shared process/inference server to prevent N-worker VRAM multiplication.
- [x] **MPS Kernels [CRITICAL]:** Ensure all operations are `device="mps"` compatible. Verified in `VisionEncoder` and MuZero torso.
- [ ] **Mixed Precision Training:** Implement `autocast()` and `GradScaler` in `train_step()`.
- [ ] **Unified Memory Management:** Use MLX or 4-bit quantization for the World Model to fit within 16-24GB Unified Memory.

## 4. Lifecycle & Deployment [PARTIAL]
- [ ] **Resume Detection:** Check for config hash checkpoints on startup; prompt user to resume/restart.
- [ ] **ROM Validation:** Compare ROM hash against known-good values in config.
- [ ] **Docker Compose:** Provide a manifest for `game-server`, `learner`, and `dashboard` as shared services.
- [ ] **Log-to-Null:** Add `log_to_file: bool` config to redirect subprocess output on slow NFS mounts.
