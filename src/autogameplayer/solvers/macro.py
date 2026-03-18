import sqlite3
import json
from typing import Dict, Any, List

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.solvers.registry import SolverRegistry


@SolverRegistry.register("macro")
class MacroSolver(BaseSolver):
    """
    A solver wrapper that intercepts decisions and checks if a known high-confidence macro applies.
    """

    def __init__(self, inner_solver: BaseSolver, optimizer: StrategyOptimizer):
        self.inner = inner_solver
        self._optimizer = optimizer
        self.last_state_hash = ""
        self.pending_sequence = []
        self.stall_counter = 0

    def _validate_macro_alignment(
        self,
        macro_actions: List[Action],
        reasoning: str,
        observation: Observation,
        context: AgentContext,
    ) -> bool:
        goal = context.current_goal or ""
        goal = goal.lower()
        if not goal:
            return True

        macro_desc = reasoning.lower()
        first_buttons = [
            getattr(a, "button", "") for a in macro_actions[:3] if hasattr(a, "button")
        ]
        is_overworld = context.metadata.get("is_overworld_active", False)

        if any(k in goal for k in ["dialogue", "talk", "text", "advance"]):
            if (
                any(b in ["up", "down", "left", "right"] for b in first_buttons)
                and "skip" not in macro_desc
            ):
                if (
                    context.metadata.get("is_dialogue", False)
                    or observation.state.has_dialogue_box
                ):
                    return False
                if is_overworld:
                    return True

        if any(k in goal for k in ["walk", "move", "go to", "exit", "enter"]):
            if all(b in ["a", "b"] for b in first_buttons) and not any(
                k in macro_desc for k in ["skip", "confirm", "open"]
            ):
                if is_overworld:
                    return False
                return True

        return True

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        map_id = obs.state.context.get("map_id", -1)

        # 1. Check for an EXACT state_hash match first (0 latency exact execution)
        try:
            with sqlite3.connect(str(self._optimizer.db_path), timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM skills WHERE vision_hash = ? AND map_id = ? ORDER BY reliability DESC LIMIT 1",
                    (obs.state_hash, map_id),
                )
                row = cursor.fetchone()
                if row and row["reliability"] > 0.8:
                    seq = (
                        self._optimizer._resolve_hierarchical_macro(row["macro_json"])
                        if row["is_hierarchical"]
                        else json.loads(row["macro_json"])
                    )
                    macro_actions = [Action(**step) for step in seq]

                    if self._validate_macro_alignment(
                        macro_actions, row["description"], obs, context
                    ):
                        print(
                            f"⚡ Hash Shortcut: {row['description']} (Reliability: {row['reliability']:.2f})"
                        )
                        self._optimizer.record_macro_usage_by_name(
                            row["name"] or row["description"]
                        )

                        macro_obj = Action(
                            macro=macro_actions,
                            reasoning=f"Exact state match: {row['description']}",
                            trigger_vision=obs.state.vision_vector,
                        )
                        return ActionProposal(
                            action=macro_obj,
                            confidence=1.0,
                            reasoning="Exact macro hash match.",
                            metadata={
                                "macro_name": row["name"],
                                "reliability": row["reliability"],
                            },
                        )
        except Exception:
            pass

        # 2. Check for Fuzzy Macro match
        if obs.state.vision_vector:
            if obs.state.is_intro_screen:
                try:
                    with sqlite3.connect(
                        str(self._optimizer.db_path), timeout=10
                    ) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.execute(
                            "SELECT * FROM skills WHERE name LIKE '%SKIP_INTRO%' OR description LIKE '%skip intro%' ORDER BY reliability DESC LIMIT 1"
                        )
                        row = cursor.fetchone()
                        if row:
                            seq = (
                                self._optimizer._resolve_hierarchical_macro(
                                    row["macro_json"]
                                )
                                if row["is_hierarchical"]
                                else json.loads(row["macro_json"])
                            )
                            macro_actions = [Action(**step) for step in seq]
                            print(
                                f"🎬 Title Screen Triggering macro: {row['name'] or row['description']}"
                            )
                            self._optimizer.record_macro_usage_by_name(
                                row["name"] or row["description"]
                            )
                            macro_obj = Action(
                                macro=macro_actions,
                                reasoning=f"Title Screen Shortcut: {row['description']}",
                                trigger_vision=obs.state.vision_vector,
                            )
                            return ActionProposal(
                                action=macro_obj,
                                confidence=1.0,
                                reasoning="Intro screen match.",
                            )
                except Exception:
                    pass

            macros = self._optimizer.get_best_macro_for_context(
                vision_vector=obs.state.vision_vector,
                vision_hash=obs.state_hash,
                map_id=map_id,
                top_k=1,
            )

            if macros:
                best = macros[0]
                is_intro_phase = obs.state.is_intro_screen
                threshold = 0.6 if is_intro_phase else 0.85

                if best.get("reliability", 1.0) >= threshold:
                    macro_actions = [Action(**step) for step in best["sequence"]]
                    if self._validate_macro_alignment(
                        macro_actions, best["description"], obs, context
                    ):
                        print(
                            f"⚡ Fuzzy Shortcut: {best['description']} (Reliability: {best.get('reliability', 1.0):.2f})"
                        )
                        self._optimizer.record_macro_usage_by_name(best["description"])
                        macro_obj = Action(
                            macro=macro_actions,
                            reasoning=f"High-confidence fuzzy macro: {best['description']}",
                            trigger_vision=obs.state.vision_vector,
                        )
                        return ActionProposal(
                            action=macro_obj,
                            confidence=best.get("reliability", 1.0),
                            reasoning="Fuzzy macro match.",
                            metadata={
                                "macro_name": best.get("name"),
                                "reliability": best.get("reliability"),
                            },
                        )

        # 3. Yield to inner solver
        proposal = await self.inner.propose_action(obs, context)
        return proposal

    async def reset(self):
        self.stall_counter = 0
        self.pending_sequence = []
        await self.inner.reset()

    def get_capabilities(self) -> Dict[str, Any]:
        return {"name": self.__class__.__name__, "inner": self.inner.get_capabilities()}
