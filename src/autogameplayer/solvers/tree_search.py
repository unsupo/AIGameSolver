
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry


@SolverRegistry.register("tree_search")
class TreeSearchSolver(BaseSolver):
    """
    Solver that explores multiple timeline branches using save-states.
    """

    def __init__(
        self,
        base_solver: BaseSolver,
        branching_depth: int = 3,
        num_branches: int = 3,
        branch_slot: int = 99,
        **kwargs,
    ):
        self.base_solver = base_solver
        self.branching_depth = branching_depth
        self.num_branches = num_branches
        self.branch_slot = branch_slot

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        # If we're already in a branch, or context doesn't support it, use base solver
        # For now, we'll return an Action that contains 'explore_branches'
        # which the Environment/Orchestrator will handle.
        
        # In a more advanced implementation, the solver would orchestrate the branching 
        # calls to the environment itself.
        
        branches = []
        for _ in range(self.num_branches):
            branch_actions = []
            # This is a bit recursive/tricky since we need to simulate.
            # For now, we'll just propose a set of random or base-solver sequences.
            for _ in range(self.branching_depth):
                # Placeholder: get action from base solver without simulated obs
                prop = await self.base_solver.propose_action(obs, context)
                action_obj = prop.action if isinstance(prop.action, Action) else Action(button=str(prop.action))
                branch_actions.append(action_obj)
            branches.append(branch_actions)

        return ActionProposal(
            action=Action(
                explore_branches=branches,
                reasoning=f"Exploring {self.num_branches} branches of depth {self.branching_depth}."
            ),
            confidence=1.0,
            reasoning="Timeline branching search.",
            metadata={"source": "tree_search", "num_branches": self.num_branches}
        )

    async def reset(self):
        await self.base_solver.reset()

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool):
        await self.base_solver.on_step_end(proposal, reward, done)

    async def on_episode_end(self, total_reward: float):
        await self.base_solver.on_episode_end(total_reward)
