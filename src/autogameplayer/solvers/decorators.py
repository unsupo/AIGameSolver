from typing import List, Dict, Any
import random

from autogameplayer.core.models import Observation
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry


@SolverRegistry.register("epsilon_greedy")
class GreedyEpsilonSolver(BaseSolver):
    """
    Decorator that injects epsilon-greedy exploration around any BaseSolver.
    """

    def __init__(
        self,
        base_solver: BaseSolver,
        epsilon: float = 0.1,
        supported_buttons: List[str] = None,
        **kwargs,
    ):
        self.base_solver = base_solver
        self.epsilon = epsilon
        # Attempt to get buttons from base solver if not provided
        self.buttons = supported_buttons or getattr(base_solver, "buttons", [])

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        if random.random() < self.epsilon and self.buttons:
            action = random.choice(self.buttons)
            return ActionProposal(
                action=action,
                confidence=self.epsilon,
                reasoning=f"Epsilon-greedy exploration ({self.epsilon:.2f}).",
                metadata={"is_exploration": True},
            )

        return await self.base_solver.propose_action(obs, context)

    async def reset(self):
        await self.base_solver.reset()

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool):
        await self.base_solver.on_step_end(proposal, reward, done)

    async def on_episode_end(self, total_reward: float):
        await self.base_solver.on_episode_end(total_reward)

    def get_capabilities(self) -> Dict[str, Any]:
        caps = self.base_solver.get_capabilities()
        caps["exploration"] = f"epsilon-greedy ({self.epsilon})"
        return caps
