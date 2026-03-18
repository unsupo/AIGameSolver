from autogameplayer.core.interfaces import Brain
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.optimizer import StrategyOptimizer


class MacroAwareBrain(Brain):
    """A brain wrapper that intercepts decisions and checks if a known high-confidence macro applies."""

    def __init__(self, inner_brain: Brain, optimizer: StrategyOptimizer):
        self.inner = inner_brain
        self._optimizer = optimizer
        # State tracking for visual breakthroughs
        self.last_state_hash = ""
        self.pending_sequence = []
        self.stall_counter = 0

    @property
    def has_reflection(self) -> bool:
        return getattr(self.inner, "has_reflection", False)

    @property
    def reflector(self):
        return getattr(self.inner, "reflector", None)

    @property
    def optimizer(self):
        return self._optimizer or getattr(self.inner, "optimizer", None)

    @property
    def session_id(self):
        return getattr(self.inner, "session_id", "unknown")

    def _validate_macro_alignment(
        self, macro_actions: list, reasoning: str, observation: Observation
    ) -> bool:
        """
        Brief validation: Does the shortcut macro align with the current Strategic Goal?
        Includes Goal-Dependency Mapping: Accepts movement in overworld even if goal is dialogue.
        """
        # If inner brain doesn't have a plan, we can't validate
        if not hasattr(self.inner, "current_plan") or not self.inner.current_plan:
            return True

        ctx = observation.state.context
        is_overworld = ctx.get("is_overworld_active", False)

        # PlannerAgent stores current_plan as a dict with 'goal'
        plan = self.inner.current_plan
        goal = ""
        if isinstance(plan, dict):
            goal = str(plan.get("goal", "")).lower()
        elif isinstance(plan, str):
            goal = plan.lower()

        if not goal:
            return True

        # Extract primary intent of the macro
        macro_desc = reasoning.lower()
        first_buttons = [getattr(a, "button", "") for a in macro_actions[:3]]

        # --- FEATURE: Goal-Dependency Mapping ---
        # 1. Dialogue Intent
        if any(k in goal for k in ["dialogue", "talk", "text", "advance"]):
            # If goal is dialogue but macro is movement...
            if (
                any(b in ["up", "down", "left", "right"] for b in first_buttons)
                and "skip" not in macro_desc
            ):
                # REJECT if we are already in dialogue (stuck in menu)
                if ctx.get("is_dialogue", False) or observation.state.has_dialogue_box:
                    print(
                        f"⚖️ Voter: REJECTED movement shortcut '{reasoning}' - Goal is DIALOGUE and we are in a textbox."
                    )
                    return False
                # ACCEPT if we are in the overworld (movement is the path to the next dialogue)
                if is_overworld:
                    # print(f"⚖️ Voter: ACCEPTED movement shortcut '{reasoning}' while goal is DIALOGUE (Overworld navigation).")
                    return True

        # 2. Movement Intent
        if any(k in goal for k in ["walk", "move", "go to", "exit", "enter"]):
            # If goal is movement but macro is mashing A/B
            if all(b in ["a", "b"] for b in first_buttons) and not any(
                k in macro_desc for k in ["skip", "confirm", "open"]
            ):
                # REJECT if in overworld (mashing A/B while walking is usually noise)
                if is_overworld:
                    print(
                        f"⚖️ Voter: REJECTED button mash shortcut '{reasoning}' - Goal is MOVEMENT and we are in Overworld."
                    )
                    return False
                # ACCEPT if we are in dialogue (must clear text to continue moving)
                return True

        return True

    async def act(self, observation: Observation, mcp_client=None) -> Action:
        # 1. Check for an EXACT state_hash match first (0 cost, ultra-fast)
        action = None
        map_id = observation.state.context.get("map_id", -1)

        try:
            import sqlite3
            import json

            with sqlite3.connect(str(self.optimizer.db_path), timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM skills WHERE vision_hash = ? AND map_id = ? ORDER BY reliability DESC LIMIT 1",
                    (observation.state_hash, map_id),
                )
                row = cursor.fetchone()
                if row and row["reliability"] > 0.8:
                    seq = (
                        self.optimizer._resolve_hierarchical_macro(row["macro_json"])
                        if row["is_hierarchical"]
                        else json.loads(row["macro_json"])
                    )
                    macro_actions = [Action(**step) for step in seq]

                    # --- FEATURE: Voter System ---
                    if self._validate_macro_alignment(
                        macro_actions, row["description"], observation
                    ):
                        print(
                            f"⚡ Hash Shortcut: {row['description']} (Reliability: {row['reliability']:.2f})"
                        )
                        self.optimizer.record_macro_usage_by_name(
                            row["name"] or row["description"]
                        )
                        action = Action(
                            macro=macro_actions,
                            reasoning=f"Exact state match: {row['description']}",
                            trigger_vision=observation.state.vision_vector,
                        )
                    # -----------------------------
        except Exception:
            pass

        # 2. Check for a high-confidence vision vector macro (Fuzzy match)
        if not action and observation.state.vision_vector:
            map_id = observation.state.context.get("map_id", -1)

            # --- FEATURE: Title Screen Detection ---
            if observation.state.is_intro_screen:
                try:
                    import sqlite3
                    import json

                    with sqlite3.connect(
                        str(self.optimizer.db_path), timeout=10
                    ) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.execute(
                            "SELECT * FROM skills WHERE name LIKE '%SKIP_INTRO%' OR description LIKE '%skip intro%' ORDER BY reliability DESC LIMIT 1"
                        )
                        row = cursor.fetchone()
                        if row:
                            seq = (
                                self.optimizer._resolve_hierarchical_macro(
                                    row["macro_json"]
                                )
                                if row["is_hierarchical"]
                                else json.loads(row["macro_json"])
                            )
                            macro_actions = [Action(**step) for step in seq]

                            print(
                                f"🎬 Title Screen Detected! Triggering macro: {row['name'] or row['description']}"
                            )
                            self.optimizer.record_macro_usage_by_name(
                                row["name"] or row["description"]
                            )
                            action = Action(
                                macro=macro_actions,
                                reasoning=f"Title Screen Shortcut: {row['description']}",
                                trigger_vision=observation.state.vision_vector,
                            )
                except Exception:
                    pass

            if not action:
                macros = self.optimizer.get_best_macro_for_context(
                    vision_vector=observation.state.vision_vector,
                    vision_hash=observation.state_hash,
                    map_id=map_id,
                    top_k=1,
                )

                if macros:
                    best = macros[0]
                    intro_maps = [0]
                    if (
                        hasattr(self.inner, "config")
                        and self.inner.config
                        and hasattr(self.inner.config, "heuristics")
                    ):
                        intro_maps = self.inner.config.heuristics.intro_map_ids

                    is_intro_phase = (
                        map_id in intro_maps
                    ) or observation.state.is_intro_screen
                    threshold = 0.6 if is_intro_phase else 0.85

                    if best.get("reliability", 1.0) >= threshold:
                        macro_actions = [Action(**step) for step in best["sequence"]]

                        # --- FEATURE: Voter System ---
                        if self._validate_macro_alignment(
                            macro_actions, best["description"], observation
                        ):
                            print(
                                f"⚡ Macro Shortcut: {best['description']} (Reliability: {best.get('reliability', 1.0):.2f})"
                            )
                            self.optimizer.record_macro_usage_by_name(
                                best["description"]
                            )
                            action = Action(
                                macro=macro_actions,
                                reasoning=f"High-confidence macro: {best['description']}",
                                trigger_vision=observation.state.vision_vector,
                            )
                        # -----------------------------

            if action:
                # Interrogate inner brain
                if hasattr(self.inner, "_process_step_outcome"):
                    await self.inner._process_step_outcome(
                        observation, mcp_client=mcp_client
                    )

                if hasattr(self.inner, "memory"):
                    self.inner.memory.record_step(observation, action)
                if hasattr(self.inner, "step_count"):
                    self.inner.step_count += 1

                return action

        # 2. If no macro, let the inner brain decide
        if not action:
            action = await self.inner.act(observation, mcp_client=mcp_client)

        # 4. POST-STEP TRACKING: Universal Breakthrough Discovery
        current_hash = observation.state_hash
        if self.last_state_hash == current_hash:
            self.stall_counter += 1
        else:
            if self.stall_counter == 0 and len(self.pending_sequence) >= 2:
                msg = f"BREAKTHROUGH CANDIDATE: Sequence {self.pending_sequence} broke visual loop."
                print(f"✨ {msg}")
                ltm = getattr(self.inner, "long_term_memory", None)
                if ltm:
                    import asyncio

                    asyncio.create_task(
                        ltm.add_memory(
                            msg, {"type": "discovery", "session_id": self.session_id}
                        )
                    )

            self.stall_counter = 0
            self.pending_sequence = []

        self.last_state_hash = current_hash
        self.pending_sequence.append(action.button.lower() if action.button else "none")

        return action

    async def close(self):
        if hasattr(self.inner, "close"):
            await self.inner.close()
