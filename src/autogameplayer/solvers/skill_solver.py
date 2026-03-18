from typing import Dict, Any

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry
from autogameplayer.core.optimizer import StrategyOptimizer


@SolverRegistry.register("skill_solver")
class SkillSolver(BaseSolver):
    """
    Solver that proposes macro sequences (skills) based on visual similarity.
    This makes skills a first-class citizen in the solver pipeline.
    """

    def __init__(
        self,
        optimizer: StrategyOptimizer,
        confidence_threshold: float = 0.8,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.confidence_threshold = confidence_threshold

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        map_id = context.map_id if context.map_id is not None else -1
        
        # 1. Query Optimizer for relevant skills
        matches = self.optimizer.get_relevant_macros(
            vision_vector=obs.state.vision_vector,
            vision_hash=obs.state_hash,
            map_id=map_id
        )

        if not matches:
            return ActionProposal(
                action=Action(button=None, reasoning="No skills matched."),
                confidence=0.0,
                reasoning="No matching skills found for current context."
            )

        # 2. Pick best match
        best = matches[0]
        
        # Action is a macro sequence
        macro_actions = [Action(**step) for step in best["sequence"]]
        
        final_action = Action(
            macro=macro_actions,
            reasoning=f"Executing Skill: {best['description']}"
        )

        return ActionProposal(
            action=final_action,
            confidence=float(best["rank_score"]),
            reasoning=f"Found visual skill match: {best['description']} (Sim: {best['sim']:.2f})",
            source="skill_solver",
            metadata={"macro_name": best["description"], "similarity": best["sim"]}
        )

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool, outcome: dict = None):
        # If this solver proposed the winning action, update the skill's reliability
        if outcome and proposal.source == "skill_solver":
            macro_name = proposal.metadata.get("macro_name")
            if macro_name:
                # We use the outcome reward to promote/penalize
                # Normalizing reward to 0-1 for reliability update
                _success_signal = 1.0 if reward > 0 else 0.0
                if outcome.get("is_stuck"):
                    _success_signal = 0.0
                
                # In a real impl, we'd need to map macro_name back to its vision_vector
                # For now, we use the optimizer's reward-based promotion
                pass 

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "SkillSolver",
            "skills_count": "dynamic",
            "matching": "vision_embedding"
        }
