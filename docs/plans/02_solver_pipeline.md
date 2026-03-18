# Phase 2: The Pluggable Solver Pipeline [CONSOLIDATED & COMPLETED]

## Objective
Established a composite architecture that replaces monolithic decision-making with a modular, config-driven solver graph.

## Achieved Architecture

### 1. Standardized Decision Contracts
- **Unified Interface:** All decision engines now implement `BaseSolver`, returning a standardized `ActionProposal` (logits, action, confidence, reasoning).
- **Typed Context:** Replaced raw dictionary probes with a typed `AgentContext` dataclass, providing solvers with structured spatial, UI, and exploration state.
- **Feedback Loop:** Integrated `on_step_end` and `on_episode_end` hooks for online learning and bandit updates.

### 2. Composite Solver Patterns
- **Solver Registry:** Created a central registry for dynamic instantiation of decision engines by string name.
- **Ensemble Solver:** Implemented weighted sum aggregation ($P(a|s) = \Sigma w_i P_i(a|s)$) to combine multiple intelligence sources.
- **Router Solver:** Enabled conditional logic to swap between solvers based on context (e.g., using LLM for menu breakouts and Agent57 for overworld).
- **Decorators:** Implemented `GreedyEpsilonSolver` and `MacroSolverDecorator` to inject exploration and TAS-playback logic around any base solver.

### 3. Lifecycle & Orchestration
- **Session Orchestrator:** Extracted the core step-loop from the runner into a standalone class that manages map transitions, stagnation tracking, and solver feedback.
- **Solver Factory:** Built a recursive factory that instantiates complex solver trees from YAML `agent_pipeline` definitions, supporting topological dependency sorting.
- **Legacy Migration:** Successfully deprecated the `Brain` interface, migrating the main CLI and evolution systems to the new `SessionOrchestrator` model.

### 4. Advanced Logic Encapsulation
- **Tree Search Solver:** Refactored state-branching and timeline exploration into a first-class solver component.
- **Macro Integration:** Integrated the genetic macro system as a solver decorator, allowing sub-second TAS shortcuts to override standard model predictions when screen hashes match known skills.
