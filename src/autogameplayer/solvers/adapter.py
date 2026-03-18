from typing import Dict, Any
from autogameplayer.core.interfaces import Brain
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.models import Observation
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry

@SolverRegistry.register("brain_adapter")
class BrainSolverAdapter(BaseSolver):
    """
    Adapts a legacy Brain implementation to the new BaseSolver interface.
    """

    def __init__(self, brain: Brain, **kwargs):
        self.brain = brain

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        # Brain.act returns an Action object
        action = await self.brain.act(obs)
        
        # Determine confidence - Brain doesn't have a standardized way to return it
        # but we can try to guess or use a default.
        confidence = 0.5
        if hasattr(action, "confidence"):
            confidence = action.confidence
        
        return ActionProposal(
            action=action,
            confidence=confidence,
            reasoning=getattr(action, "reasoning", "Action from adapted brain."),
            metadata={"solver_name": self.brain.__class__.__name__}
        )

    async def reset(self):
        # Optional: check if brain has reset
        if hasattr(self.brain, "reset"):
            await self.brain.reset()

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool, outcome: dict = None):
        if hasattr(self.brain, "on_step_end"):
            await self.brain.on_step_end(proposal, reward, done, outcome)

    async def on_episode_end(self, total_reward: float):
        if hasattr(self.brain, "on_episode_end"):
            await self.brain.on_episode_end(total_reward)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": f"AdaptedBrain({self.brain.__class__.__name__})",
            "brain_type": self.brain.__class__.__name__
        }
