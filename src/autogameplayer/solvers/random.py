import random
from typing import List

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry


@SolverRegistry.register("random")
class RandomSolver(BaseSolver):
    """
    Executes random actions uniformly from the supported input list.
    """

    def __init__(self, supported_buttons: List[str]):
        self.supported_buttons = supported_buttons

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        button = random.choice(self.supported_buttons)
        return ActionProposal(
            action=Action(button=button, reasoning="Randomly sampled button press."),
            confidence=0.5,
            reasoning="Randomly sampled button press.",
            metadata={"solver_name": "RandomSolver"},
        )
