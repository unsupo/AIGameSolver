from typing import List, Dict, Optional
import random
import torch

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry
...
@SolverRegistry.register("random")
class RandomSolver(BaseSolver):
    """
    Baseline solver returning uniform logits and random actions.
    """

    def __init__(self, supported_buttons: List[str], **kwargs):
        self.buttons = supported_buttons

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        button = random.choice(self.buttons)
        # Uniform logits
        logits = torch.ones(len(self.buttons)) / len(self.buttons)
        return ActionProposal(
            action=Action(button=button, reasoning="Randomly selected action."),
            confidence=1.0 / len(self.buttons),
            reasoning="Randomly selected action.",
            metadata={"logits": logits.tolist(), "solver_name": "RandomSolver"},
        )


@SolverRegistry.register("ensemble")
class EnsembleSolver(BaseSolver):
    """
    Computes final distribution as weighted sum of sub-solvers.
    P(a|s) = Σ wᵢ Pᵢ(a|s)
    """

    def __init__(
        self,
        solvers: Dict[str, BaseSolver],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        self.solvers = solvers
        self.weights = weights or {name: 1.0 for name in solvers}
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        combined_logits = None
        explanations = []

        for name, solver in self.solvers.items():
            proposal = await solver.propose_action(obs, context)
            weight = self.weights.get(name, 0.0)

            # Expecting logits in metadata for ensemble math
            logits = proposal.metadata.get("logits")
            if logits is not None:
                logits_tensor = torch.tensor(logits) * weight
                if combined_logits is None:
                    combined_logits = logits_tensor
                else:
                    combined_logits += logits_tensor

            explanations.append(f"{name} ({weight:.2f}): {proposal.reasoning}")

        if combined_logits is None:
            # Fallback if no sub-solvers provided logits
            return ActionProposal(
                action=Action(button=None, reasoning="No valid sub-proposals."),
                confidence=0.0,
                reasoning="Ensemble had no valid sub-proposals."
            )

        # Get best action from combined distribution
        best_idx = torch.argmax(combined_logits).item()
        # This assumes all sub-solvers share the same action space/ordering.
        # In a real implementation, we'd need an action mapping layer.
        
        # For now, we'll try to get buttons from the first solver that has them
        buttons = []
        for s in self.solvers.values():
            if hasattr(s, "buttons"):
                buttons = s.buttons
                break
        
        action_btn = buttons[best_idx] if buttons else f"idx_{best_idx}"
        confidence = float(torch.max(combined_logits))

        return ActionProposal(
            action=Action(button=action_btn, reasoning=f"Ensemble selection: {best_idx}"),
            confidence=confidence,
            reasoning=" | ".join(explanations),
            metadata={"logits": combined_logits.tolist()},
        )

    async def reset(self):
        for solver in self.solvers.values():
            await solver.reset()

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool):
        for solver in self.solvers.values():
            await solver.on_step_end(proposal, reward, done)

    async def on_episode_end(self, total_reward: float):
        for solver in self.solvers.values():
            await solver.on_episode_end(total_reward)


@SolverRegistry.register("router")
class RouterSolver(BaseSolver):
    """
    Swaps between solvers based on AgentContext (e.g. stuck steps or dialogue).
    """

    def __init__(
        self,
        default_solver: BaseSolver,
        fallback_solver: BaseSolver,
        stuck_threshold: int = 10,
        **kwargs,
    ):
        self.default_solver = default_solver
        self.fallback_solver = fallback_solver
        self.stuck_threshold = stuck_threshold
        self.active_solver_name = "default"

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        use_fallback = (
            context.consecutive_stuck_steps >= self.stuck_threshold
            or context.is_dialogue
            or context.is_stuck
        )

        if use_fallback:
            self.active_solver_name = "fallback"
            return await self.fallback_solver.propose_action(obs, context)
        else:
            self.active_solver_name = "default"
            return await self.default_solver.propose_action(obs, context)

    async def reset(self):
        await self.default_solver.reset()
        await self.fallback_solver.reset()

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool):
        await self.default_solver.on_step_end(proposal, reward, done)
        await self.fallback_solver.on_step_end(proposal, reward, done)

    async def on_episode_end(self, total_reward: float):
        await self.default_solver.on_episode_end(total_reward)
        await self.fallback_solver.on_episode_end(total_reward)
