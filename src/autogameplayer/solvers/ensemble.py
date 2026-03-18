import asyncio
from typing import List, Dict, Any

from autogameplayer.core.models import Observation
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry


@SolverRegistry.register("ensemble")
class EnsembleSolver(BaseSolver):
    """
    Evaluates multiple solvers concurrently and returns the proposal with the highest confidence.
    """

    def __init__(self, solvers: List[BaseSolver]):
        self.solvers = solvers

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        # Evaluate all solvers concurrently
        tasks = [solver.propose_action(obs, context) for solver in self.solvers]
        results: List[ActionProposal] = await asyncio.gather(*tasks)

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)

        best = results[0]
        # Avoid mutating the original metadata directly if it's shared, but for ActionProposal it's fine
        metadata = dict(best.metadata) if best.metadata else {}
        metadata["ensemble_votes"] = [
            {"action": r.action, "confidence": r.confidence} for r in results
        ]
        best.metadata = metadata
        return best

    async def reset(self):
        tasks = [solver.reset() for solver in self.solvers]
        await asyncio.gather(*tasks)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "sub_solvers": [s.get_capabilities() for s in self.solvers],
        }
