# Phase 11 — Research-Backed Improvements [IN PROGRESS]

## Overview
Grounding the architecture in 2022–2025 research (DreamerV3, HER, GCIC).

---

## 1. DreamerV3-style RSSM (symlog+two-hot rewards) [COMPLETED]
- [x] **Symlog Math:** Implemented in `src/autogameplayer/utils/math.py`.
- [x] **Two-Hot Encoding:** Categorical reward/value prediction for robust scaling.
- [x] **Softmax Decoding:** Expectation-based scalar readout for inference.

## 2. Symlog Observations [COMPLETED]
- [x] **Input Normalization:** Applied `symlog` to encoder inputs in `RepresentationNetwork`.

## 3. Block GRU + RMSNorm [COMPLETED]
- [x] **Block GRU:** Replaced LSTM with 4-parallel-block GRU for MPS acceleration.
- [x] **RMSNorm:** Integrated into torso and recurrent cells for training stability.

## 4. Imagination-Based Policy Training [COMPLETED]
- [x] **Trainer Infrastructure:** `MuZeroOfflineTrainer` supports sequence training.
- [x] **Latent Unrolling:** Full synthetic imagination rollouts implemented in `train_step`.

## 5. Hindsight Experience Replay (HER) [COMPLETED]
- [x] **Replay Processor:** Created `src/autogameplayer/core/replay_processor.py`.
- [x] **Relabeling Logic:** Automatically boosts priority for transitions leading to sparse curriculum goals.

## 6. LLM Scaffolding for Vision [COMPLETED]
- [x] **Structured Perception:** `LLMMetaSolver` now passes OCR, SLAM, and sub-solver proposals as text.
- [x] **Decision Context:** High-level executive synthesis instead of raw pixel reasoning.

## 9. Goal-Conditioned Intrinsic Curiosity (GCIC) [COMPLETED]
- [x] **Curriculum Relevance:** Intrinsic rewards are scaled by proximity to current curriculum goal.

---

## Summary Table [UPDATED]

| # | Improvement | Phase | Priority | Status |
|---|-------------|-------|----------|--------|
| 1 | DreamerV3 RSSM | 3 | High | ✅ COMPLETED |
| 2 | Symlog Inputs | 1, 3 | Medium | ✅ COMPLETED |
| 3 | Block GRU + RMSNorm | 3, 7 | Medium | ✅ COMPLETED |
| 4 | Imagination Training | 3 | High | ✅ COMPLETED |
| 5 | HER for Curriculum | 8 | High | ✅ COMPLETED |
| 6 | LLM Scaffolding | 4 | Medium | ✅ COMPLETED |
| 9 | GCIC | 3, 8 | Medium | ✅ COMPLETED |
| 11| Diffusion Policy Head| 3 | High | 📅 DEFERRED |
| 12| Active Foveated Vision| 1, 7 | High | 📅 DEFERRED |
