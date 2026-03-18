from typing import Dict, Any

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry


@SolverRegistry.register("router")
class RouterSolver(BaseSolver):
    """
    Routes to a specific solver based on context rules or an LLM routing call.
    """

    def __init__(self, solvers: Dict[str, BaseSolver], default_solver: str):
        self.solvers = solvers
        self.default_solver = default_solver
        if self.default_solver not in self.solvers:
            raise ValueError(f"Default solver {default_solver} not in solver map.")

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        # 1. Macro Lock Check (Highest Priority - mid-sequence execution)
        if context.active_macro and len(context.active_macro) > 0:
            action = context.active_macro.pop(0)
            return ActionProposal(
                action=action,
                confidence=1.0,
                reasoning="Executing LLM-requested bottleneck macro step.",
                metadata={"routed_by": "macro_lock"},
            )

        # 2. PRIORITY 1: Physical Stuck State or Loop Detection (Route to LLM)
        # Threshold lowered to 10 as per instruction
        if context.consecutive_stuck_steps >= 10 or context.is_loop_detected:
            if "llm" in self.solvers:
                print("🧠 Bottleneck/Collision detected. Routing to LLM Meta-Reasoner.")
                solver = self.solvers["llm"]
                proposal = await solver.propose_action(obs, context)

                # If the LLM returned a macro sequence, lock to it
                macro_seq = None
                if (
                    isinstance(proposal.action, Action)
                    and proposal.action.macro
                    and len(proposal.action.macro) > 0
                ):
                    macro_seq = proposal.action.macro
                elif proposal.metadata and "macro_sequence" in proposal.metadata:
                    macro_seq = proposal.metadata["macro_sequence"]

                if macro_seq:
                    context.lock_to_macro(macro_seq)
                    action = context.active_macro.pop(0)
                    proposal.action = action

                metadata = dict(proposal.metadata) if proposal.metadata else {}
                metadata["routed_by"] = "llm_bottleneck"
                proposal.metadata = metadata
                return proposal

        # 3. PRIORITY 2: UI Lock (Menus/Dialogues)
        is_dialogue = context.metadata.get("is_dialogue", False) or obs.state.has_dialogue_box
        if is_dialogue and "menu" in self.solvers:
            return await self.solvers["menu"].propose_action(obs, context)

        # 4. PRIORITY 3: Primary Neural Solver (MuZero/Agent57)
        primary_solver_name = self.default_solver
        if primary_solver_name in self.solvers:
            try:
                proposal = await self.solvers[primary_solver_name].propose_action(obs, context)
                
                # Check if the primary solver is 'clueless' (NoDataError pattern)
                # If confidence is extremely low and we have a random fallback, use it.
                if proposal.confidence < 0.1 and "random" in self.solvers:
                    raise ValueError("Primary solver has no confident data for this state.")
                    
                metadata = dict(proposal.metadata) if proposal.metadata else {}
                metadata["routed_by"] = primary_solver_name
                proposal.metadata = metadata
                return proposal
            except (ValueError, Exception) as e:
                # 5. PRIORITY 4: Fallback to FAST Random
                if "random" in self.solvers:
                    print(f"💨 MuZero/Primary is clueless ({e}). Defaulting to FAST Random.")
                    proposal = await self.solvers["random"].propose_action(obs, context)
                    metadata = dict(proposal.metadata) if proposal.metadata else {}
                    metadata["routed_by"] = "random_fallback"
                    proposal.metadata = metadata
                    return proposal

        # Ultimate Fallback
        solver = self.solvers[self.default_solver]
        return await solver.propose_action(obs, context)

    async def reset(self):
        for solver in self.solvers.values():
            await solver.reset()

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "solvers": {k: s.get_capabilities() for k, s in self.solvers.items()},
        }
