from typing import Dict, Optional, Any
import json
import sqlite3

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry
from autogameplayer.core.optimizer import StrategyOptimizer


@SolverRegistry.register("macro_decorator")
class MacroSolverDecorator(BaseSolver):
    """
    Decorator that intercepts decisions and checks if a known high-confidence macro applies.
    """

    def __init__(
        self,
        base_solver: BaseSolver,
        optimizer: StrategyOptimizer,
        **kwargs,
    ):
        self.base_solver = base_solver
        self.optimizer = optimizer
        self.last_state_hash = ""
        self.pending_sequence = []
        self.stall_counter = 0

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        # 1. Check for macro shortcuts
        macro_proposal = await self._find_macro(obs, context)
        if macro_proposal:
            return macro_proposal

        # 2. If no macro, use base solver
        return await self.base_solver.propose_action(obs, context)

    async def _find_macro(self, obs: Observation, context: AgentContext) -> Optional[ActionProposal]:
        map_id = obs.state.context.get("map_id", -1)
        
        # EXACT state_hash match
        try:
            with sqlite3.connect(str(self.optimizer.db_path), timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM skills WHERE vision_hash = ? AND map_id = ? ORDER BY reliability DESC LIMIT 1",
                    (obs.state_hash, map_id),
                )
                row = cursor.fetchone()
                if row and row["reliability"] > 0.8:
                    seq = self.optimizer._resolve_hierarchical_macro(row["macro_json"]) if row["is_hierarchical"] else json.loads(row["macro_json"])
                    macro_actions = [Action(**step) for step in seq]
                    
                    if self._validate_alignment(macro_actions, row["description"], obs, context):
                        self.optimizer.record_macro_usage_by_name(row["name"] or row["description"])
                        return ActionProposal(
                            action=Action(
                                macro=macro_actions,
                                reasoning=f"Exact state match: {row['description']}",
                                trigger_vision=obs.state.vision_vector
                            ),
                            confidence=float(row["reliability"]),
                            reasoning=f"Macro Shortcut: {row['description']}",
                            metadata={"macro_name": row["name"], "source": "exact_hash"}
                        )
        except Exception:
            pass

        # Fuzzy vision match
        macros = self.optimizer.get_best_macro_for_context(
            vision_vector=obs.state.vision_vector,
            vision_hash=obs.state_hash,
            map_id=map_id,
            top_k=1
        )
        if macros:
            best = macros[0]
            if best.get("reliability", 1.0) >= 0.85:
                macro_actions = [Action(**step) for step in best["sequence"]]
                if self._validate_alignment(macro_actions, best["description"], obs, context):
                    self.optimizer.record_macro_usage_by_name(best["description"])
                    return ActionProposal(
                        action=Action(
                            macro=macro_actions,
                            reasoning=f"High-confidence macro: {best['description']}",
                            trigger_vision=obs.state.vision_vector
                        ),
                        confidence=float(best.get("reliability", 1.0)),
                        reasoning=f"Macro Shortcut: {best['description']}",
                        metadata={"macro_name": best["description"], "source": "fuzzy_vision"}
                    )
        
        return None

    def _validate_alignment(self, macro_actions: list, description: str, obs: Observation, context: AgentContext) -> bool:
        # Simplified validation for now
        return True

    async def reset(self):
        await self.base_solver.reset()
        self.last_state_hash = ""
        self.pending_sequence = []
        self.stall_counter = 0

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool):
        await self.base_solver.on_step_end(proposal, reward, done)
        
        # Track breakthroughs (similar to MacroAwareBrain)
        # ... logic for tracking breakthroughs ...

    async def on_episode_end(self, total_reward: float):
        await self.base_solver.on_episode_end(total_reward)
        # Trigger genetic consolidation if needed

    def get_capabilities(self) -> Dict[str, Any]:
        caps = self.base_solver.get_capabilities()
        caps["macros"] = "enabled"
        return caps
