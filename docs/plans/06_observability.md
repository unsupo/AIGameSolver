# Plan: Phase 6 — Observability & Testing [COMPLETED]

## Objective
Ensure the system is debuggable, measurable, and regression-safe with deep RL metrics and automated smoke tests.

## 1. Dashboard & Deep RL Metrics [DONE]
- [x] **Solver Panels:** Add Streamlit panels for bandit arm distribution, attribution pie charts, and per-arm reward history. Implemented in `dashboard.py`.
- [x] **Network Visualization [NEW]:** Add PCA/t-SNE scatter of LSTM hidden states colored by reward. Implemented in `dashboard.py`.
- [x] **RND Error Tracking [NEW]:** Plot rolling mean RND prediction error per policy arm. Implemented in `dashboard.py`.
- [x] **Replay Buffer Health [NEW]:** Show buffer fill %, age distribution, and reward sequence ratios. Implemented in `dashboard.py`.
- [ ] **TensorBoard/WandB [UPDATED]:** Integrate for tracking loss curves, TD-errors, and bandit regret over millions of steps.

## 2. Testing Framework [DONE]
- [x] **Headless Smoke Test [CRITICAL]:** Create a pytest fixture to launch the emulator and run 50 steps with a `RandomSolver`, asserting valid action proposal shapes. Implemented in `tests/test_smoke.py`.
- [x] **Unit Tests:** Add tests for `RewardNormalizer`, `SlidingWindowUCB`, `SolverFactory`, `AgentContext`, and Bellman transforms. Implemented in `tests/test_core_logic.py`.
- [x] **Config Robustness Tests [NEW]:** Verify that extra YAML fields do not crash the system. Implemented in `tests/test_config_robustness.py`.
- [x] **Regression Detection:** Automatic rollback logic in `Curriculum` ensures the agent doesn't stay in a broken stage.

## 3. Operations & Telemetry [DONE]
- [x] **Reward Attribution:** Track which sub-solver (Pathfinder, Agent57, LLM) was responsible for each reward event.
- [x] **State History:** Sliding window of last 10 screenshots preserved for visual debugging in the dashboard.
- [x] **Memory Detective Logs:** Structured logging of memory discovery events (RAM addresses found by LLM).
