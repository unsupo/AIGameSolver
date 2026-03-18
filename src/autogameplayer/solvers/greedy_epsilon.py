import random
from typing import Dict, Any

from autogameplayer.core.models import Observation
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry


@SolverRegistry.register("greedy_epsilon")
class GreedyEpsilonSolver(BaseSolver):
    """
    Wraps a primary solver. With probability epsilon, acts randomly.
    Otherwise, defers to the primary solver.
    """

    def __init__(
        self,
        primary_solver: BaseSolver,
        random_solver: BaseSolver,
        epsilon: float = 0.1,
    ):
        self.primary_solver = primary_solver
        self.random_solver = random_solver
        self.epsilon = epsilon

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        if random.random() < self.epsilon:
            proposal = await self.random_solver.propose_action(obs, context)
            proposal.metadata["greedy_epsilon_routed"] = "random"
            return proposal

        proposal = await self.primary_solver.propose_action(obs, context)
        proposal.metadata["greedy_epsilon_routed"] = "primary"
        return proposal

    async def reset(self):
        await self.primary_solver.reset()
        await self.random_solver.reset()

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "primary": self.primary_solver.get_capabilities(),
            "epsilon": self.epsilon,
        }
