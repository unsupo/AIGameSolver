# Plan: Phase 8 — Curriculum & Skill Transfer

## Objective
Implement adaptive curricula and a first-class skill system for high-reward sequence re-use and transfer.

## 1. Curriculum Learning
- [x] **Agnostic Condition Evaluator [NEW]:** Replace checkpoint RAM reads with a generic evaluator (RAM, Reward, Vision Similarity).
- [x] **Curriculum Regression Detector [CRITICAL]:** Automatically roll back the bootstrap slot if performance on checkpoint N+1 falls below checkpoint N.
- [ ] **Curriculum Dashboard:** Add a panel showing stage success rates and rollback history.

## 2. Skill / Macro System
- [x] **`SkillSolver` Implementation [NEW]:** Create a `BaseSolver` to execute high-reward sequences discovered via `GoldRunDiscovery` and genetic refinement.
- [ ] **Agnostic Skill Matching:** Use vision-embedding similarity instead of exact screen-hashes to activate skills.
- [ ] **Population Diversity [NEW]:** Implement niching (fitness sharing) in `PopulationManager` to maintain behavioral diversity.
- [ ] **TAS Recording:** Capture nested actions executed during branching tree searches.

## 3. Multi-Game Training
- [ ] **Multi-Game Orchestrator [NEW]:** Implement round-robin training across a list of `[rom, config]` pairs.
- [ ] **Game ID Context:** Use `game_id` in `AgentContext` to maintain separate bandit statistics per game.
- [ ] **Cross-Game Transfer:** Implement a `SkillLibrary` for cross-game skill reuse using embedding similarity.
# Phase 8 — Curriculum & Skill Transfer

## Objective
Adaptive curricula that advance automatically and roll back on regression. A first-class `SkillLibrary` that captures, generalises, and reuses high-reward sequences across games.

---

## 1. Curriculum Learning

### Agnostic condition evaluator
- [x] **Replace all checkpoint RAM reads with `ConditionEvaluator`.**
  ```python
  class ConditionEvaluator:
      def evaluate(self, ctx: AgentContext, cond: ConditionConfig) -> bool: ...
  ```
  Condition types (YAML):
    ```yaml
    condition:
      type: "ram"
      params:
        address: 0xD35E
        target_value: 1
    ```
    - `ram`: RAM probe comparison (supports hex strings or ints for address)
    - `reward`: threshold check on last reward
    - `ocr`: regex match on current frame OCR text
    - `vision_similarity`: cosine similarity of current vision embedding to a target vector

  **Backward Compatibility:** The `Checkpoint` model supports a legacy format where `address` and `target_value` are defined at the top level of the checkpoint object (mapping automatically to a `ram` type condition).

  ### Regression detector
  - [x] **[CRITICAL] Automatic bootstrap rollback.** After advancing to stage N+1, measure the moving average reward over `curriculum_eval_window` episodes. If it falls below the stage N moving average by more than `curriculum_regression_threshold`, automatically load the stage N bootstrap slot and demote. Log the rollback as a structured event.

- [ ] **Rollback history in dashboard:** Panel showing stage advancement timeline with rollback events marked. Makes curriculum health visible without digging through logs.

### Population diversity
- [ ] **Fitness sharing / niching in `PopulationManager`:** Workers that are behaviourally similar (cosine similarity of their recent trajectory latents > `niching_threshold`) share fitness scores, reducing the effective reward of redundant strategies. Encourages the population to maintain diverse exploration styles.

---

## 2. Skill / Macro System

### `SkillLibrary`
- [x] **[CRITICAL] `SkillSolver` as a `BaseSolver`.** Executes sequences from the `SkillLibrary`. Integrates as a standard arm in the `EnsembleSolver` or `RouterSolver`. `MacroAwareDecorator` becomes a thin wrapper that checks the library before delegating.
- [ ] **Agnostic skill matching via embedding similarity.** Skills are activated by cosine similarity between the current `vision_latent` and the skill's stored `trigger_latent` — not by exact screen hash. Threshold from YAML `skill_match_threshold: 0.92`. This allows skills learned on one visual variant of a state to activate on slightly different visual variants.
- [ ] **Cross-game skill transfer via `SkillLibrary`.** Skills are stored with their `trigger_latent` and `game_id`. The `SkillSolver` searches across all games (with a cross-game discount on the similarity threshold). A walking skill from Pokémon Red that produced high reward may activate in Link's Awakening if the visual context is similar enough.
- [ ] **`GoldRunDiscovery`:** After each episode, scan the replay buffer for the highest-reward contiguous subsequence of length ≥ `min_skill_length`. If the reward exceeds `skill_promotion_threshold`, add it to the `SkillLibrary` with the entry-frame latent as the trigger.
- [ ] **Genetic refinement of skills:** `PopulationManager` evolves candidate skill sequences via mutation (random action substitution) and crossover (splice two sequences at a random point). Fitness = total reward when replaying the sequence from a saved state.
- [ ] **Reactive macro execution:** During skill execution, if SLAM detects a collision (odometer displacement < expected), abort the skill immediately. Do not execute the remaining actions blindly.

### TAS recording
- [ ] **Automatic `.json` TAS capture** on three events:
    1. `GoldRunDiscovery` promotes a new skill.
    2. A map warp is detected (milestone).
    3. A curriculum stage advances.
    - TAS file format: `{game_id, stage, actions: [{button, duration_frames}], entry_latent, total_reward}`.

---

## 3. Multi-Game Training

### `MultiGameOrchestrator`
- [ ] **Round-robin across `[rom, config]` pairs.** Each worker is assigned a game from the list in YAML `multi_game.roms`. After each episode, the next game in the rotation is assigned. Workers share the `ReplayBuffer` and `MEMETrainer` — the World Model learns a shared latent space across games.
- [ ] **`game_id` in `AgentContext`:** Bandit UCB statistics are stored per `(arm, game_id)`. The bandit for Pokémon Red and the bandit for Link's Awakening are separate, even though the network is shared.
- [ ] **`SkillLibrary` with cross-game index:** Skills indexed by `trigger_latent` across all `game_id` values. When matching, return the best match from the current game first; fall back to cross-game matches with a reduced similarity threshold.

---

## Acceptance Criteria

- [ ] Curriculum advances from stage 1 → 2 without Python changes when `badges >= 1` in YAML config.
- [ ] A curriculum regression (simulated by reducing reward) triggers a rollback within `curriculum_eval_window` episodes — verified in `test_curriculum.py`.
- [ ] A skill promoted in game A activates in game B when the visual similarity exceeds the cross-game threshold — verified in `test_cross_game_skill.py`.
- [ ] `SkillSolver` aborts execution within 1 frame of detecting a SLAM collision.