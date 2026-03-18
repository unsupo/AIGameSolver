import asyncio
from autogameplayer.core.models import Observation
from autogameplayer.core.config_loader import GameConfig
from .memory import EpisodicMemory, LongTermMemory


class CriticAgent:
    """Agent responsible for evaluating performance and detecting stagnation (stuck states)."""

    def __init__(
        self,
        ltm: LongTermMemory = None,
        session_id: str = "default",
        config: GameConfig = None,
    ):
        self.milestones = set()
        self.last_states = {}  # Tracks previous values for transition detection
        self.hash_counts = {}  # Tracks vision_hash encounters for satiation
        self.interaction_counts = {}
        self.map_step_counts = {}
        self.ltm = ltm
        self.session_id = session_id
        self.config = config
        self.eval_count = 0

        self._load_global_milestones()
        if self.ltm:
            hc, ic = self.ltm.load_decay_counts()
            self.hash_counts = hc
            self.interaction_counts = ic

    def _load_global_milestones(self):
        """Hydrate milestones from LTM so the AI knows what it already achieved across runs."""
        if not self.ltm:
            return

        # Use a new method on LTM to avoid raw DB calls here
        loaded = self.ltm.get_global_milestones()
        for k in loaded:
            self.milestones.add(k)

        if self.milestones:
            print(
                f"🧠 Critic: Loaded {len(self.milestones)} global milestones from LTM."
            )

    def _register_milestone(self, key: str, reward: float, guidance: str):
        """Helper to record permanent milestones to LTM."""
        if key in self.milestones:
            return 0.0, ""

        self.milestones.add(key)
        if self.ltm:
            asyncio.create_task(
                self.ltm.add_memory(
                    guidance, {"type": "milestone", "milestone_key": key}
                )
            )
        return reward, guidance

    def evaluate(
        self, memory: EpisodicMemory, current_obs: Observation
    ) -> tuple[float, bool, str, bool]:
        """Returns (reward_delta, is_stuck, guidance, is_loop)."""
        # BULLETPROOF COOLDOWN: Total immunity during escape maneuvers
        if memory.steps:
            last_reason = list(memory.steps)[-1].reasoning or ""
            if any(
                r in last_reason
                for r in ["Breaking stuck state", "Mash Breakout", "Interface Breakout"]
            ):
                return (
                    0.0,
                    False,
                    "Cooldown: Waiting for escape maneuver to finish.",
                    False,
                )

        ctx = current_obs.state.context
        last_reward = ctx.get("last_reward", 0.0)
        map_id = ctx.get("map_id", -1)
        curr_coords = (ctx.get("x", 0), ctx.get("y", 0))

        # Load configurable intro maps, default to [0] if missing
        heuristics_obj = (
            getattr(self.config, "heuristics", None) if self.config else None
        )

        # Safely convert the Pydantic model to a dictionary so .get() and .items() work
        if hasattr(heuristics_obj, "model_dump"):
            heuristics = heuristics_obj.model_dump()  # Pydantic v2
        elif hasattr(heuristics_obj, "dict"):
            heuristics = heuristics_obj.dict()  # Pydantic v1
        elif isinstance(heuristics_obj, dict):
            heuristics = heuristics_obj
        else:
            heuristics = {}

        intro_maps = heuristics.get("intro_map_ids", [0])

        is_dialogue = (
            ctx.get("is_dialogue", False)
            or current_obs.state.has_dialogue_box
            or map_id == -1
        )
        is_overworld = ctx.get("is_overworld_active", False)

        reward_delta = 0.0
        is_stuck = False
        is_loop = False
        guidance = ""

        # --- 6. CONFIG-DRIVEN MILESTONES (Discovery Rewards) ---
        from autogameplayer.core.models import GameStage

        # A. Stage Discovery (Agnostic)
        stage = current_obs.state.stage
        if stage != self.last_states.get("stage", GameStage.UNKNOWN):
            s_key = f"stage_{stage.value}"
            if s_key not in self.milestones:
                r, g = self._register_milestone(
                    s_key,
                    2.0,
                    f"STAGE ACHIEVED: You entered the {stage.value} phase of the game!",
                )
                reward_delta += r
                guidance = g
        self.last_states["stage"] = stage

        # B. Map Discovery (Agnostic)
        if (
            map_id != self.last_states.get("map_id", -1)
            and self.last_states.get("map_id", -1) != -1
        ):
            m_key = f"reach_map_{map_id}"
            if m_key not in self.milestones:
                r, g = self._register_milestone(
                    m_key,
                    5.0,
                    f"OBJECTIVE ACHIEVED: You discovered a new area (Map #{map_id})!",
                )
                reward_delta += r
                guidance = g
        self.last_states["map_id"] = map_id

        # C. Config-driven Context Monitors
        # Evaluates game-specific context variables (e.g. party_count, money, badges_count) defined in YAML
        if heuristics and "milestones" in heuristics:
            for ctx_key, milestone_cfg in heuristics["milestones"].items():
                current_val = ctx.get(ctx_key, 0)
                last_val = self.last_states.get(ctx_key, 0)

                if isinstance(current_val, (int, float)) and current_val > last_val:
                    # Value increased! E.g. obtained a badge or money
                    base_reward = milestone_cfg.get("reward", 1.0)
                    msg = milestone_cfg.get(
                        "guidance", f"OBJECTIVE ACHIEVED: {ctx_key} increased!"
                    )

                    is_critical = milestone_cfg.get("critical", False)
                    if is_critical:
                        # Only reward critical milestones once globally
                        m_key = f"{ctx_key}_{current_val}"
                        if m_key not in self.milestones:
                            r, g = self._register_milestone(m_key, base_reward, msg)
                            reward_delta += r
                            guidance = g
                    else:
                        # Reward every time (e.g. money)
                        reward_delta += base_reward
                        guidance = msg

                self.last_states[ctx_key] = current_val

        # 0. MAP-SPECIFIC STAGNATION PENALTY (Session-based)
        if not hasattr(self, "map_step_counts"):
            self.map_step_counts = {}
        if not hasattr(self, "interaction_counts"):
            self.interaction_counts = {}

        self.map_step_counts[map_id] = self.map_step_counts.get(map_id, 0) + 1

        # --- FEATURE: Dopamine Decay (Reward Satiation) ---
        # If the same state hash (exact screen) is seen repeatedly,
        # rewards turn into heavy penalties.
        h_count = self.hash_counts.get(current_obs.state_hash, 0) + 1
        self.hash_counts[current_obs.state_hash] = h_count

        if h_count > 3:
            # We've seen this exact screen too much. It's boring/a trap.
            reward_delta -= 10.0
            if h_count % 5 == 0:
                guidance = f"STATE SATIATION: You have returned to this exact state {h_count} times. It is a dead end. Choose a completely different path."
        elif h_count == 1:
            # New visual state!
            reward_delta += 1.0
        # --------------------------------------------------

        # --- FEATURE: Interaction Boredom (Dopamine Decay) ---
        # If the AI keeps pressing A at the exact same coordinate (SNES fix)
        # the reward should decay rapidly to encourage moving away.
        last_button = memory.steps[-1].button if memory.steps else "none"
        if last_button == "a":
            coord_key = f"{map_id}_{curr_coords[0]}_{curr_coords[1]}"
            count = self.interaction_counts.get(coord_key, 0) + 1
            self.interaction_counts[coord_key] = count

            # Decay Logic:
            if count == 1:
                interaction_reward = 5.0
            elif count < 5:
                interaction_reward = 2.0
            elif count < 10:
                interaction_reward = 0.5
            else:
                interaction_reward = -10.0

            reward_delta += interaction_reward
            if count >= 10:
                guidance = f"CRITICAL BOREDOM: You have interacted with this object {count} times. It is now causing negative progress. You MUST walk away and find fresh dopamine elsewhere."
        # ----------------------------------------------------

        # --- FEATURE: Map-Specific Multiplier ---
        # Ignore stagnation on intro maps
        stagnation_multiplier = 0.0 if map_id in intro_maps else 1.0
        # ----------------------------------------

        if self.map_step_counts.get(map_id, 0) >= 100:
            map_stagnation = (
                (self.map_step_counts[map_id] // 100) * 0.1 * stagnation_multiplier
            )
            reward_delta -= map_stagnation
            if self.map_step_counts[map_id] % 100 == 0 and stagnation_multiplier > 0:
                guidance = f"MAP STAGNATION: You have spent {self.map_step_counts[map_id]} steps on Map #{map_id}. Move to a new area."

        if last_reward < -0.5:
            is_stuck = True

        if len(memory.steps) < 5:
            return reward_delta, is_stuck, guidance, is_loop

        recent = list(memory.steps)[-8:]

        # --- 7. PROGRESS DETECTION ---
        # If progress is zero for multiple steps, we are stuck
        if len(memory.steps) >= 5:
            recent_progress = []
            for i in range(1, 6):
                s = memory.steps[-i]
                p = s.vision_delta + (
                    1.0
                    if i < len(memory.steps) and s.ocr != memory.steps[-i - 1].ocr
                    else 0.0
                )
                recent_progress.append(p)

            if sum(recent_progress) < 0.05:  # Threshold for "Zero Progress"
                is_stuck = True
                guidance = "STAGNATION: Zero progress detected (Vision, OCR, and Hash are static). Switch strategy."

        # 1. Static Screen Detection (Frustration)
        if current_obs.state.vision_delta < 0.01:
            is_intro_map = map_id in intro_maps

            if (is_dialogue or is_intro_map) and any(
                s.button in ["a", "b", "start"] for s in recent[-3:]
            ):
                pass
            else:
                is_stuck = True
                guidance = "STAGNATION: The screen is not changing. You are hitting an obstacle or running a useless macro."

        # 2. STATE ENTROPY CHECK (The Loop Detector)
        last_action_reason = memory.steps[-1].reasoning if memory.steps else ""
        if len(memory.steps) >= 6 and "Breaking stuck state" not in last_action_reason:
            curr_ocr_strip = (current_obs.state.ocr_text or "").strip()
            state_fingerprint = f"{curr_coords[0]}_{curr_coords[1]}_{curr_ocr_strip}"

            recent_fingerprints = []
            for step in list(memory.steps)[-8:]:
                step_ocr = (step.ocr or "").strip()
                recent_fingerprints.append(
                    f"{step.coords[0]}_{step.coords[1]}_{step_ocr}"
                )

            loop_count = recent_fingerprints.count(state_fingerprint)

            if loop_count > 3:
                reward_delta -= 2.0
                is_stuck = True
                is_loop = True
                guidance = (
                    f"STATE LOOP DETECTED: You keep returning to the exact same state {state_fingerprint}. "
                    "This visual pattern is a dead end. Use a different escape vector (A sequence of 4+ buttons) to break the cycle."
                )

        # 3. Cutscene Awareness
        if ctx.get("is_cutscene_locked"):
            is_stuck = False
            reward_delta = 0.0
            guidance = (
                "Game is executing a scripted event. Wait for control to be returned."
            )

        # 4. Semantic Reward Filtering
        if current_obs.state.vision_delta > 0.05:
            if is_overworld:
                reward_delta += 1.0
            else:
                reward_delta = 0.0

        # 5. Dialogue Feedback Loop
        if is_dialogue:
            last_ocr = memory.steps[-1].ocr if memory.steps else ""
            curr_ocr = current_obs.state.ocr_text or ""
            has_arrow = current_obs.state.has_dialogue_arrow
            last_button = memory.steps[-1].button if memory.steps else "none"

            if (
                curr_ocr == last_ocr
                and not has_arrow
                and last_button in ["a", "b", "start"]
            ):
                reward_delta -= 0.5
                guidance = "DIALOGUE MASHING: You are pressing buttons while text is still printing. Wait for the arrow."

            if curr_ocr != last_ocr and last_button in ["a", "b", "start"]:
                reward_delta += 1.0

        # 7. SPATIAL STAGNATION (Long-term)
        if len(memory.steps) >= 20:
            long_recent = list(memory.steps)[-20:]
            spatial_static = all(
                s.coords == curr_coords and s.map_id == map_id for s in long_recent
            )

            if (
                spatial_static
                and not is_dialogue
                and not ctx.get("is_cutscene_locked")
                and stagnation_multiplier > 0
            ):
                is_stuck = True
                guidance = "LONG-TERM SPATIAL STAGNATION: Position and Map have not changed for 20 steps. Strategy is failing to move the player."

        # Short-term coordinate stagnation
        if (
            not is_stuck
            and curr_coords != (0, 0)
            and not is_dialogue
            and stagnation_multiplier > 0
        ):
            static = all(s.coords == curr_coords for s in recent[-5:])
            directional = all(
                s.button in ["up", "down", "left", "right"] for s in recent[-5:])
            if directional and static:
                is_stuck = True
                guidance = f"MOVEMENT BLOCKED: You are pressing {recent[-1].button.upper()} but coordinates remain {curr_coords}. You are hitting a wall or NPC. Try moving orthogonal to your current path."


        # --- FEATURE: Dead End Persistence ---
        if is_stuck and self.ltm:
            # Record this visual state as a 'Dead End' in persistent memory
            asyncio.create_task(
                self.ltm.record_dead_end(current_obs.state_hash, self.session_id)
            )
        # ------------------------------------

        self.eval_count += 1
        if self.eval_count % 50 == 0 and self.ltm:
            asyncio.create_task(
                self.ltm.persist_decay_counts(self.hash_counts, self.interaction_counts)
            )

        return reward_delta, is_stuck, guidance, is_loop
