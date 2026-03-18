import asyncio
import concurrent.futures
import random
import time
from collections import deque
import sqlite3
import torch

from autogameplayer.core.config import settings
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.utils.llm import LLMClientProtocol, OllamaBootstrap
from autogameplayer.utils.database import get_db_connection

from .memory import EpisodicMemory, LongTermMemory
from .planner import PlannerAgent
from .actor import ActorAgent
from .critic import CriticAgent
from .reflector import ReflectionAgent
from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.core.registry import Registry
from autogameplayer.core.knowledge import KnowledgeBase
from autogameplayer.core.curiosity import CuriosityEngine
from autogameplayer.core.meta_controller import MetaController
from autogameplayer.core.rnd import RandomNetworkDistillation


@Registry.register_brain("agentic")
class AgenticBrain(Brain):
    """
    Advanced agentic architecture using a Planner-Actor-Critic pattern.
    Features: Long-term memory (RAG), Automated Reflection, and Discovery Rewards.
    """

    def __init__(
        self,
        controller: Controller,
        config: GameConfig = None,
        llm_client: LLMClientProtocol = None,
    ):
        self.controller = controller
        self.config = config
        self.model = (config.llm_model if config else None) or settings.llm_model
        self.planner_model = (config.planner_model if config else None) or self.model
        self.reflector_model = (
            config.reflector_model if config else None
        ) or self.model
        self.session_id = f"session_{int(time.time())}"  # Unique ID for this run

        OllamaBootstrap.bootstrap(
            [self.model, self.planner_model, self.reflector_model, "nomic-embed-text"]
        )

        if llm_client is None:
            from autogameplayer.utils.llm import get_llm_client

            self.client = get_llm_client()
        else:
            self.client = llm_client

        self.memory = EpisodicMemory()
        self.long_term_memory = LongTermMemory(self.client)
        self.critic = CriticAgent(
            ltm=self.long_term_memory, session_id=self.session_id, config=self.config
        )
        self.optimizer = StrategyOptimizer(self.client, self.reflector_model)
        self.knowledge = KnowledgeBase(self.client)
        self.curiosity_engine = CuriosityEngine()
        self.meta_controller = MetaController()

        # Determine device for PyTorch modules
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnd = RandomNetworkDistillation(device=self.device)
        self.current_personality = 0

        from autogameplayer.utils.pathfinding import Pathfinder

        self.pathfinder = Pathfinder(self.long_term_memory.storage_path)

        self.planner = PlannerAgent(
            self.client,
            self.planner_model,
            self.config,
            optimizer=self.optimizer,
            knowledge=self.knowledge,
            critic=self.critic,
        )
        self.actor = ActorAgent(
            self.client,
            self.model,
            self.controller.buttons,
            self.config,
            ltm=self.long_term_memory,
            optimizer=self.optimizer,
        )
        self.reflector = ReflectionAgent(
            self.client, self.reflector_model, self.optimizer, knowledge=self.knowledge
        )

        # --- NEW: Predictive Architecture (MuZero Sub-Brain) ---
        self.muzero = None
        if getattr(settings, "muzero_model", None):
            from autogameplayer.muzero.brain import MuZeroBrain

            self.muzero = MuZeroBrain(self.controller, model_name=settings.muzero_model)
        # -------------------------------------------------------

        self.current_plan = {
            "goal": "Explore the world and advance the story.",
            "steps": [],
            "abort_condition": "Stagnation",
            "expected_map_after": None,
        }
        self.step_count = 0
        self.last_map_id = -1
        self.last_pos = (0, 0)
        self.last_button = ""
        self.last_action_obj = None
        self.last_state_hash = ""
        self.last_recalled_memories = []

        # Stagnation management
        self.stagnation_counter = 0
        self.collision_counter = 0
        self.collision_points = []  # List of (x, y) for prompt context
        self.transition_lock = 0  # Cooldown after map change
        self.map_history = deque(maxlen=10) # Track recent maps
        self.last_transition_time = 0
        self.is_oscillating = False
        self.anchor_recovery_count = 0  # Track recovery frequency
        self.macro_history = {}  # state_hash -> (last_macro_name, last_step)
        self.last_success_step = 0
        self.last_save_step = 0
        self.last_rolling_save_step = 0
        self.drift_steps = 0
        self.maps_discovered = 0
        self.discovered_maps = set()
        self.dialogue_exit_cooldown = 0
        self.last_is_dialogue = False
        self.menu_step_count = 0  # 2. FEATURE: Menu Exhaustion tracking

        # Async task management
        self._tasks = set()
        self._is_reflecting = False
        self._is_planning = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @property
    def has_reflection(self) -> bool:
        return True

    async def plan(self, observation: Observation, mcp_client=None):
        """
        Predictive Planning Phase:
        Generates a denoised 'Hidden State' latent representation of the world.
        Also runs mental simulations to find optimal sub-sequences.
        """
        if self.muzero:
            # 1. Generate Hidden State
            await self.muzero.plan(observation, mcp_client)

            # 2. FEATURE: Mental Imagination (Internal Dreaming)
            # We use the dynamics model to find a sequence that maximizes reward
            if observation.state.hidden_state:
                h_tensor = torch.tensor(
                    observation.state.hidden_state, dtype=torch.float32
                ).to(self.muzero.device)
                best_indices, best_score = (
                    self.muzero.dreamer.find_best_mental_sequence(
                        h_tensor, depth=10, num_sequences=50
                    )
                )

                if (
                    best_score > 0.5
                ):  # Only use if the model is confident in a high reward path
                    dream_buttons = [self.muzero.buttons[idx] for idx in best_indices]
                    print(
                        f"💭 Mental Dream: Found sequence {dream_buttons} with score {best_score:.2f}"
                    )
                    # We can store this in the plan or observation context for the Actor
                    observation.state.context["mental_dream"] = dream_buttons

    async def act(self, observation: Observation, mcp_client=None) -> Action:
        # --- FEATURE: Meta-Controller (Personality Selection) ---
        new_personality = await self.meta_controller.select_personality()

        if (
            self.current_personality is not None
            and new_personality != self.current_personality
        ):
            print(
                f"🎭 Personality Shift: Arm changed from {self.current_personality} to {new_personality}."
            )

        self.current_personality = new_personality

        if self.muzero:
            self.muzero.current_personality = self.current_personality
        # --------------------------------------------------------

        ctx = observation.state.context
        map_id = ctx.get("map_id", -1)
        (ctx.get("x", 0), ctx.get("y", 0))
        is_dialogue = ctx.get("is_dialogue", False)

        # --- FEATURE: Bridge Burning (Curriculum Advancement) ---
        # If we successfully transitioned from Map 38/39 (Intro Bedroom) to Map 37 (1F),
        # promote this state to Slot 0 so rollbacks are more efficient.
        if self.last_map_id in [38, 39, 0] and map_id == 37:
            print(
                f"🔥 Bridge Burned: Reached Map #{map_id}. Saving NEW Global Bootstrap to Slot {settings.bootstrap_slot}..."
            )
            if mcp_client:
                asyncio.create_task(
                    mcp_client.call_tool(
                        "manage_checkpoint",
                        {"action": "save", "slot": settings.bootstrap_slot},
                    )
                )
        # --------------------------------------------------------

        self.last_is_dialogue = is_dialogue

        # Capture current state for outcome processing (Odometer logic)
        self.last_pos = (ctx.get("x", 0), ctx.get("y", 0))
        self.last_map_id = map_id
        self.last_state_hash = observation.state_hash

        # 1. Background Reflection & Evolution (Every 100 steps)
        if (
            self.step_count > 0
            and self.step_count % 100 == 0
            and not self._is_reflecting
        ):
            self._is_reflecting = True

            async def run_reflection():
                try:
                    limit = 500 if self.step_count % 500 == 0 else 100
                    await self.reflector.analyze_session(
                        self.session_id, self.long_term_memory, limit=limit
                    )

                    if self.optimizer:
                        self.optimizer.mine_sequences(top_k=5)
                        self.optimizer.evolve_population(top_k=5)
                        self.optimizer.cleanup_database()
                        await self.optimizer.async_compress_skills()
                finally:
                    self._is_reflecting = False

            task = asyncio.create_task(run_reflection())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # 2. Outcome Processing from last step
        rollback_occurred = await self._process_step_outcome(
            observation, mcp_client=mcp_client
        )
        if rollback_occurred:
            return Action(
                button="none",
                duration=10,
                reasoning="Dead End Rollback: Resetting to Bootstrap state.",
            )

        # 3. Handle Stagnation & Timeline Branching
        is_dialogue = ctx.get("is_dialogue", False)
        pos = (ctx.get("x", 0), ctx.get("y", 0))

        if is_dialogue:
            # RESET stagnation during dialogue - we are progressing text!
            self.stagnation_counter = 0
            if observation.guidance:
                observation.guidance = observation.guidance.replace(
                    "STAGNATION", "DIALOGUE_ACTIVE"
                )
            
            # --- FEATURE: High-Frequency Intro A-Spam ---
            # If in the intro phase, spam A more aggressively to clear text blocks.
            intro_maps = self.config.heuristics.intro_map_ids if self.config else [0]
            if map_id in intro_maps or map_id == 38:
                # Force A every 2 steps instead of waiting for menu exhaustion
                if self.step_count % 2 == 0:
                    print("📖 Intro Dialogue: High-frequency A-press to clear text.")
                    action = Action(button="a", duration=5, reasoning="Intro Dialogue: Aggressive A-spam to clear text blocks.")
                    self.memory.record_step(observation, action, solver_name="IntroAProcessor")
                    self.step_count += 1
                    return action

        # 2. Update Stagnation Counter
        is_menu_open = ctx.get("is_menu_open", False)
        if is_menu_open:
            self.menu_step_count += 1
            if self.stagnation_counter > 0:
                print("⏸️ Stagnation Paused: Menu is active.")
            self.stagnation_counter = 0  # Lock to 0 during menu interactions
            
            # --- FEATURE: Robust Intro Protection ---
            # Disable B-Spam breakout if we are on the title screen, 'New Game' menu, or Naming Screen
            # to prevent backing out of the start sequence or resetting names.
            ocr_text = (observation.state.ocr_text or "").upper()
            is_start_menu = "POKEMON" in ocr_text or "NEW GAME" in ocr_text
            is_naming_screen = "NAME" in ocr_text or "ED" in ocr_text or "A B C" in ocr_text or map_id in [0, 38]

            # Use higher threshold for critical intro/UI areas
            exhaustion_threshold = 50 if (is_start_menu or is_naming_screen) else 10

            # --- FEATURE: Menu Exhaustion (B button weight) ---
            if self.menu_step_count > exhaustion_threshold:
                if is_start_menu or is_naming_screen:
                    btn = "start" if is_start_menu else "a"
                    print(f"💤 Menu Exhaustion (Intro): Forcing {btn.upper()} press...")
                    action = Action(
                        button=btn,
                        duration=15,
                        reasoning="Intro Menu Exhaustion: Forcing progress to avoid reset loop.",
                    )
                else:
                    print("💤 Menu Exhaustion: Forcing B press to exit interface.")

                    action = Action(
                        button="b",
                        duration=5,
                        reasoning="Menu Exhaustion: Interface active for too many steps without progress. Returning to world.",
                    )
                # Determine solver name for telemetry
                solver_name = "AgenticBrain"
                if action.reasoning:
                    if "Menu Exhaustion" in action.reasoning:
                        solver_name = "MenuExhaustion"
                    elif "Interface active" in action.reasoning:
                        solver_name = "MenuExhaustion"
                    elif "Interface Breakout" in action.reasoning:
                        solver_name = "InterfaceBreakout"
                    elif "Pathfinding" in action.reasoning:
                        solver_name = "Pathfinder"
                    elif "Stagnation" in action.reasoning or "Stagnant" in action.reasoning:
                        solver_name = "StagnationRecovery"

                self.memory.record_step(observation, action, solver_name=solver_name)
                self.step_count += 1
                return action
        else:
            self.menu_step_count = 0 # Reset on world interaction

        if self.transition_lock > 0:
            self.transition_lock -= 1
            print(f"🔒 Transition Lock Active ({self.transition_lock} steps). Favoring forward movement.")

        if (
            observation.guidance
            and ("STAGNATION" in observation.guidance or "STAGNATION_COLLISION_BURST" in observation.guidance)
            and not is_dialogue
            and not is_menu_open
        ):
            self.stagnation_counter += 1

            if "STAGNATION_COLLISION_BURST" in observation.guidance:
                # II. Anchor-Based Recovery (Elastic Reset)
                # If oscillating, snap to the OLDEST anchor to force a major timeline branch
                if self.is_oscillating:
                    print("🔄 OSCILLATION RECOVERY: Snapping to oldest frontier anchor.")
                    # Get oldest anchor by timestamp
                    try:
                        with get_db_connection(self.long_term_memory.storage_path) as conn:
                            conn.row_factory = sqlite3.Row
                            cursor = conn.execute("SELECT * FROM spatial_anchors WHERE map_id = ? ORDER BY timestamp ASC LIMIT 1", (map_id,))
                            anchor = dict(cursor.fetchone())
                    except Exception:
                        anchor = self.long_term_memory.get_nearest_spatial_anchor(map_id, pos[0], pos[1])
                else:
                    anchor = self.long_term_memory.get_nearest_spatial_anchor(map_id, pos[0], pos[1])

                if anchor and mcp_client:
                    print(f"⚓ ANCHOR RECOVERY: Snapping back to waypoint at ({anchor['x']}, {anchor['y']})")
                    self.anchor_recovery_count += 1
                    
                    # If we've recovered too many times, force a major long-horizon plan
                    if self.anchor_recovery_count > 3:
                        print("🚨 RECOVERY LOOP: Furthest Anchor Snap triggered.")
                        observation.guidance = (observation.guidance or "") + "\nCRITICAL_BOTTLENECK_PLAN_REQUIRED"
                        # Try to find the OLDEST anchor (furthest in time/branch)
                        try:
                            with get_db_connection(self.long_term_memory.storage_path) as conn:
                                conn.row_factory = sqlite3.Row
                                cursor = conn.execute("SELECT * FROM spatial_anchors WHERE map_id = ? ORDER BY timestamp ASC LIMIT 1", (map_id,))
                                oldest_anchor = dict(cursor.fetchone())
                                if oldest_anchor:
                                    anchor = oldest_anchor
                        except Exception:
                            pass
                        self.anchor_recovery_count = 0 # Reset after hard snap

                    asyncio.create_task(mcp_client.call_tool("manage_checkpoint", {"action": "load", "slot": anchor["slot_id"]}))
                    self.stagnation_counter = 0
                    self.collision_counter = 0
                    self.is_oscillating = False
                    self.drift_steps = 50 # Heavier drift for major reset
                    return Action(reasoning=f"Anchor Recovery: Resetting to {'OLDEST' if self.is_oscillating else 'NEAREST'} discovery at ({anchor['x']}, {anchor['y']})")
                
                escape_btn = random.choice(["up", "down", "left", "right"])
                print(f"🚨 COLLISION RECOVERY: Forcing {escape_btn.upper()} for 45 frames.")
                action = Action(
                    button=escape_btn,
                    duration=45,
                    reasoning="Collision Burst Recovery: Physically clearing obstacle after multiple hits.",
                )
                # Determine solver name for telemetry
                solver_name = "AgenticBrain"
                if action.reasoning:
                    if "Menu Exhaustion" in action.reasoning:
                        solver_name = "MenuExhaustion"
                    elif "Interface active" in action.reasoning:
                        solver_name = "MenuExhaustion"
                    elif "Interface Breakout" in action.reasoning:
                        solver_name = "InterfaceBreakout"
                    elif "Pathfinding" in action.reasoning:
                        solver_name = "Pathfinder"
                    elif "Stagnation" in action.reasoning or "Stagnant" in action.reasoning:
                        solver_name = "StagnationRecovery"

                self.memory.record_step(observation, action, solver_name=solver_name)
                self.step_count += 1
                return action

            if self.stagnation_counter == 10:
                print("🧠 Triggering Instant Debug Reflection...")
                task = asyncio.create_task(
                    self.reflector.analyze_failure(
                        self.session_id,
                        self.long_term_memory,
                        map_id,
                        self.last_pos,
                        observation,
                    )
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

            if 5 < self.stagnation_counter < 30 and self.stagnation_counter % 5 == 0:
                print(
                    f"⚠️ Stagnation at {self.stagnation_counter}. Injecting entropy to force exploration."
                )
                self.drift_steps = 3

            if self.stagnation_counter >= 30:
                rollback_slot = settings.tas_trigger_slot
                if self.last_rolling_save_step > 0:
                    print(
                        f"🚨 CRITICAL STAGNATION ({self.stagnation_counter}): Branching timeline. Rolling back to Slot {rollback_slot}..."
                    )

                if mcp_client:
                    asyncio.create_task(
                        mcp_client.call_tool(
                            "stop_recording",
                            {
                                "filename": f"stagnation_rollback_{int(time.time())}.json",
                                "metadata": {
                                    "reason": "Rollback due to stagnation",
                                    "stagnation_count": self.stagnation_counter,
                                },
                            },
                        )
                    )

                self.stagnation_counter = 0
                self.drift_steps = 30  # Enter high-entropy mode
                self.curiosity_engine.end_episode()

                action = Action(
                    load_state=rollback_slot,
                    reasoning=f"Automated timeline branch: Rolling back to slot {rollback_slot} due to prolonged stagnation.",
                )
                # Determine solver name for telemetry
                solver_name = "AgenticBrain"
                if action.reasoning:
                    if "Menu Exhaustion" in action.reasoning:
                        solver_name = "MenuExhaustion"
                    elif "Interface active" in action.reasoning:
                        solver_name = "MenuExhaustion"
                    elif "Interface Breakout" in action.reasoning:
                        solver_name = "InterfaceBreakout"
                    elif "Pathfinding" in action.reasoning:
                        solver_name = "Pathfinder"
                    elif "Stagnation" in action.reasoning or "Stagnant" in action.reasoning:
                        solver_name = "StagnationRecovery"

                self.memory.record_step(observation, action, solver_name=solver_name)
                self.step_count += 1
                return action

            recent_escape_buttons = [
                s.button
                for s in list(self.memory.steps)[-4:]
                if s.reasoning and "Breaking stuck state" in s.reasoning
            ]
            dpad_exhausted = len(recent_escape_buttons) >= 2

            is_strict_overworld = (
                ctx.get("is_overworld_active", False)
                and not ctx.get("is_dialogue", False)
                and not ctx.get("is_engine_locked", False)
                and not ctx.get("is_menu_open", False)
                and ctx.get("is_player_named", True)
            )

            if is_strict_overworld and not dpad_exhausted:
                escape_btn = random.choice(["up", "down", "left", "right"])
                print(
                    f"⚠️  STAGNATION: Physically breaking stuck state at {ctx.get('coordinates', 'unknown')} with {escape_btn.upper()}."
                )
                action = Action(
                    button=escape_btn,
                    duration=30,
                    reasoning="Breaking stuck state by physically moving away.",
                )
            else:
                recent_steps = list(self.memory.steps)[-6:]
                recent_ui_escapes = [
                    s
                    for s in recent_steps
                    if s.reasoning and "Interface Breakout" in s.reasoning
                ]

                if len(recent_ui_escapes) >= 3:
                    all_same_hash = all(
                        s.vision_hash == observation.state_hash
                        for s in recent_ui_escapes
                    )
                    if all_same_hash:
                        escape_btn = random.choice(["down", "left", "up", "right"])
                        print(
                            f"🔥 Entropy Hold: Interface breakout failing. Forcing {escape_btn.upper()} for 60 frames."
                        )
                        action = Action(
                            button=escape_btn,
                            duration=60,
                            reasoning=f"Entropy Hold: Physically clearing zone with {escape_btn.upper()} after spam failed.",
                        )
                        # Determine solver name for telemetry
                        solver_name = "AgenticBrain"
                        if action.reasoning:
                            if "Menu Exhaustion" in action.reasoning:
                                solver_name = "MenuExhaustion"
                            elif "Interface active" in action.reasoning:
                                solver_name = "MenuExhaustion"
                            elif "Interface Breakout" in action.reasoning:
                                solver_name = "InterfaceBreakout"
                            elif "Pathfinding" in action.reasoning:
                                solver_name = "Pathfinder"
                            elif "Stagnation" in action.reasoning or "Stagnant" in action.reasoning:
                                solver_name = "StagnationRecovery"

                        self.memory.record_step(observation, action, solver_name=solver_name)
                        self.step_count += 1
                        return action

                b_count = len([s for s in recent_ui_escapes if s.button == "b"])
                if b_count >= 2:
                    escape_btn = random.choice(["a", "start"])
                else:
                    escape_btn = "b"

                print(
                    f"🔄 Escalating Escape: Using {escape_btn.upper()} spam to break UI lock."
                )
                action = Action(
                    button=escape_btn,
                    repeat=15,
                    until_visual_change=True,
                    reasoning=f"Interface Breakout: Closing menu/modal using {escape_btn.upper()}.",
                )
        else:
            self.stagnation_counter = 0  # Reset on progress

            # --- Macro Cooldown Logic ---
            state_hash = observation.state_hash
            if state_hash in self.macro_history:
                last_macro, last_step = self.macro_history[state_hash]
                # If we've used a macro in this exact visual state within the last 5 steps, block it
                if self.step_count - last_step < 5:
                    block_msg = f"\n⚠️ FORBIDDEN MACRO: You just used {last_macro} in this state. It failed to significantly change the game state. DO NOT repeat it. Choose a different sequence."
                    observation.guidance = (
                        (observation.guidance + block_msg)
                        if observation.guidance
                        else block_msg
                    )
            # ----------------------------

            try:
                session_metrics = {
                    "total_steps": self.step_count,
                    "maps_discovered": self.maps_discovered,
                    "badges": ctx.get("badges_count", 0),
                    "steps_since_save": self.step_count - self.last_save_step,
                    "collision_points": list(self.collision_points),
                    "transition_lock": self.transition_lock,
                    "is_oscillating": self.is_oscillating,
                    "map_history": list(self.map_history),
                }

                # Inject internal state for sub-actor logic
                ctx["last_button"] = self.last_button

                action = await self.actor.get_next_action(
                    observation,
                    self.current_plan,
                    self.memory,
                    mcp_client=mcp_client,
                    drift=(self.drift_steps > 0),
                    session_metrics=session_metrics,
                )

                # Record macro usage for cooldown
                if action.macro or (
                    action.button and action.button.upper().startswith("SKILL_")
                ):
                    macro_name = (
                        action.button
                        if (action.button and action.button.upper().startswith("SKILL_"))
                        else f"Macro({action.reasoning[:15]})"
                    )
                    self.macro_history[observation.state_hash] = (
                        macro_name,
                        self.step_count,
                    )

                # --- FEATURE: Pathfinder Integration ---
                # Check for target_coords from either Actor OR Global Server Goal
                target_coords = action.target_coords
                server_goal = ctx.get("target_coords")

                if server_goal and not target_coords:
                    target_coords = server_goal
                    print(
                        f"🎯 Pathfinder: Using Global Navigation Goal: {target_coords}"
                    )

                if target_coords:
                    goal_str = self.current_plan.get("goal", "Unknown")
                    print(f"⏳ Actor: Deciding path for goal: '{goal_str}'")
                    start_pos = (ctx.get("x", 0), ctx.get("y", 0))
                    path = self.pathfinder.find_path(map_id, start_pos, target_coords)
                    if path:
                        print(
                            f"📍 Pathfinder: Found path to {target_coords} ({len(path)} steps)."
                        )
                        macro_actions = [
                            Action(button=btn, duration=10, until_visual_change=True)
                            for btn in path
                        ]

                        # FEATURE: Macro Synthesis (Pathfinding to Skill)
                        if self.optimizer:
                            skill_name = (
                                f"SKILL_PATH_TO_{target_coords[0]}_{target_coords[1]}"
                            )
                            skill_desc = f"Pathfinder sequence from {start_pos} to {target_coords} on Map {map_id}."

                            # Convert Action list to simple dict sequence for optimizer
                            macro_json = [
                                {"button": m.button, "frames": m.duration}
                                for m in macro_actions
                            ]

                            # Save as a highly reliable skill since it's calculated from real SLAM data
                            self.optimizer.save_macro(
                                vision_vector=observation.state.vision_vector,
                                sequence=macro_json,
                                map_id=map_id,
                                coords=start_pos,
                                description=skill_desc,
                                score=5.0,  # High initial score
                                vision_hash=observation.state_hash,
                            )
                            # Explicitly name it for re-use
                            try:
                                with sqlite3.connect(
                                    str(self.optimizer.db_path), timeout=10
                                ) as conn:
                                    conn.execute(
                                        "UPDATE skills SET name = ?, reliability = 0.9 WHERE vision_hash = ? AND map_id = ?",
                                        (skill_name, observation.state_hash, map_id),
                                    )
                                    conn.commit()
                            except Exception:
                                pass

                        action = Action(
                            macro=macro_actions,
                            reasoning=f"Pathfinding to {target_coords} | {action.reasoning}",
                        )

                        # Clear the global goal if it was used successfully
                        if server_goal and mcp_client:
                            asyncio.create_task(
                                mcp_client.call_tool("clear_navigation_goal", {})
                            )
                    else:
                        print(
                            f"⚠️ Pathfinder: No path found from {start_pos} to {target_coords}. Falling back to LLM action."
                        )
                # ---------------------------------------

                if self.drift_steps > 0:
                    self.drift_steps -= 1
            except Exception:
                action = self.fallback_action(self.controller)

        needs_new_plan = (
            self.step_count % 30 == 0
            or map_id != self.last_map_id
            or observation.guidance
            and "STAGNATION" in observation.guidance
        )

        if needs_new_plan and not self._is_planning:
            self._is_planning = True

            async def run_planner():
                try:
                    plan, recalled = await self.planner.generate_plan(
                        observation, self.long_term_memory
                    )
                    self.current_plan = plan
                    self.last_recalled_memories = recalled

                    if mcp_client:
                        # Update the server so the dashboard sees the recalled memories
                        asyncio.create_task(
                            mcp_client.call_tool(
                                "set_recalled_memories", {"memories": recalled}
                            )
                        )
                        asyncio.create_task(
                            mcp_client.call_tool(
                                "set_plan", {"plan": plan.get("goal", "Unknown")}
                            )
                        )
                finally:
                    self._is_planning = False

            task = asyncio.create_task(run_planner())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # Determine solver name for telemetry
        solver_name = "AgenticBrain"
        if action.reasoning:
            if "Menu Exhaustion" in action.reasoning:
                solver_name = "MenuExhaustion"
            elif "Interface active" in action.reasoning:
                solver_name = "MenuExhaustion"
            elif "Interface Breakout" in action.reasoning:
                solver_name = "InterfaceBreakout"
            elif "Pathfinding" in action.reasoning:
                solver_name = "Pathfinder"
            elif "Stagnation" in action.reasoning or "Stagnant" in action.reasoning:
                solver_name = "StagnationRecovery"

        self.memory.record_step(observation, action, solver_name=solver_name)
        self.step_count += 1
        self.last_button = action.button.lower() if action.button else "none"
        self.last_action_obj = action
        self.last_state_hash = observation.state_hash

        prefix = "(Plan Pending) " if self._is_planning else ""
        repeat_str = f" (x{action.repeat})" if action.repeat > 1 else ""
        print(
            f"🎮 {prefix}Action: {action.button.upper() if action.button else 'NONE'}{repeat_str} | {action.reasoning[:50]}"
        )

        observation.state.recalled_memories = self.last_recalled_memories

        return action

    def fallback_action(self, controller: Controller) -> Action:
        btns = controller.buttons if controller.buttons else ["a", "up"]
        
        # --- FEATURE: Context-Aware Fallback ---
        # Prioritize START or A when in the intro phase/title screen
        map_id = self.last_map_id
        if map_id == 0 or map_id is None:
            priority_btns = [b for b in btns if b.lower() in ["start", "a"]]
            if priority_btns:
                return Action(
                    button=random.choice(priority_btns), 
                    duration=10, 
                    reasoning="Internal brain fallback (Intro priority)"
                )

        return Action(
            button=random.choice(btns), duration=5, reasoning="Internal brain fallback"
        )

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    async def _process_step_outcome(self, observation: Observation, mcp_client=None):
        """Refactored outcome processor to handle rewards, memory, and discoveries."""
        ctx = observation.state.context
        map_id = ctx.get("map_id", -1)
        pos = (ctx.get("x", 0), ctx.get("y", 0))
        ocr = (observation.state.ocr_text or "").upper()
        is_menu_open = ctx.get("is_menu_open", False)

        reward_delta, is_stuck, critic_guidance, is_loop = self.critic.evaluate(
            self.memory, observation
        )

        # --- FEATURE: Never Give Up (NGU) Reward Loop ---
        if observation.state.vision_vector and not is_menu_open:
            # Convert vision_vector (List[float]) to tensor
            v_tensor = torch.tensor(
                observation.state.vision_vector, dtype=torch.float32
            )

            # 1. Episodic Novelty (CuriosityEngine with state-based decay)
            r_episodic = self.curiosity_engine.compute_intrinsic_reward(
                v_tensor,
                r_lifelong=self.rnd.compute_intrinsic_reward(v_tensor, train=False),
            )

            # 2. Lifelong Novelty (Random Network Distillation)
            # Only train RND if not in menu to avoid UI obsession
            r_lifelong = self.rnd.compute_intrinsic_reward(v_tensor, train=True)

            # 3. Add current step to episodic memory buffer to degrade future immediate rewards
            self.curiosity_engine.add_to_episode(v_tensor)

            # 4. Agent57 Combined Reward Formula
            # r_total = r_extrinsic + L * (min(max(r_episodic, 1.0), 5.0) * r_lifelong)
            L = self.meta_controller.get_beta(self.current_personality)
            combined_intrinsic = min(max(r_episodic, 1.0), 5.0) * r_lifelong

            # --- FEATURE: Menu Reward Scaling ---
            if is_menu_open:
                combined_intrinsic *= 0.1
            # ------------------------------------

            # --- FEATURE: Spatial Anchor Bonus (Frontier exploration) ---
            anchor = self.long_term_memory.get_nearest_spatial_anchor(map_id, pos[0], pos[1])
            if anchor:
                dist = ((anchor['x'] - pos[0])**2 + (anchor['y'] - pos[1])**2)**0.5
                # Bonus for exploration frontier (dist > 10 tiles)
                if dist > 10:
                    combined_intrinsic += 0.5

            reward_delta += L * combined_intrinsic
        # ------------------------------------------------

        # --- FEATURE: Discovery Milestone -> Save Spatial Anchor ---
        if reward_delta > 1.5 and not is_menu_open:
            # Discovery Thresholding: Only save if far from existing anchors
            anchor = self.long_term_memory.get_nearest_spatial_anchor(map_id, pos[0], pos[1])
            should_save = True
            if anchor:
                dist = ((anchor['x'] - pos[0])**2 + (anchor['y'] - pos[1])**2)**0.5
                if dist < 3.0:
                    should_save = False
            
            if should_save:
                # Slot 1 is usually the 'Latest Good' rolling save
                slot_id = 1
                print(
                    f"📍 New Discovery! Registering Spatial Anchor at {pos} (Slot {slot_id})"
                )
                task = asyncio.create_task(
                    self.long_term_memory.record_spatial_anchor(
                        map_id=map_id,
                        x=pos[0],
                        y=pos[1],
                        state_hash=observation.state_hash,
                        slot_id=slot_id,
                        description=f"High Reward Discovery (Reward: {reward_delta:.2f})",
                    )
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        # ------------------------------------------------

        # Update Meta-Controller with the new reward signal
        await self.meta_controller.update(self.current_personality, reward_delta)

        # --- FEATURE: Lifelong Map Discovery Reward ---
        if map_id != -1 and map_id not in self.discovered_maps:
            self.discovered_maps.add(map_id)
            # Massive dopamine hit for finding a new room/area
            map_reward = 10.0
            reward_delta += map_reward
            print(
                f"🗺️ Lifelong Milestone: Discovered Map #{map_id}! (+{map_reward} Reward)"
            )
        # ----------------------------------------------

        self.memory.update_last_step(observation, is_stuck, reward=reward_delta)

        # --- FEATURE: Recursive Dead End Rollback ---
        # If this EXACT visual state has caused stagnation in 3+ separate sessions,
        # assume the current 'Frontier' save is a Dead End and rollback to Bootstrap.
        # if self.long_term_memory and not is_dialogue:
        #     dead_end_count = self.long_term_memory.get_dead_end_count(observation.state_hash)
        #     if dead_end_count >= 3:
        #         print(f"💀 DEAD END DETECTED: This state ({observation.state_hash}) has failed in {dead_end_count} sessions.")
        #         print(f"🔄 Recursive Rollback: Returning to Slot {settings.bootstrap_slot} (Bootstrap).")
        #
        #         if mcp_client:
        #             # 1. Force load the absolute starting point
        #             asyncio.create_task(mcp_client.call_tool("manage_checkpoint", {"action": "load", "slot": settings.bootstrap_slot}))
        #
        #             # 2. Add a strong negative memory to the LTM to penalize the path that led here
        #             asyncio.create_task(self.long_term_memory.add_memory(
        #                 f"DEAD END PATH: The route leading to state {observation.state_hash} is a failure. Try a totally different strategy.",
        #                 {"type": "rule", "penalty": 10.0, "state_hash": observation.state_hash}
        #             ))
        #
        #         # Reset session progress to avoid immediate re-trigger
        #         self.stagnation_counter = 0
        #         self.drift_steps = 50 # Heavier drift to escape the old timeline
        #         # We return True to indicate a rollback occurred
        #         return True
        # --------------------------------------------

        if critic_guidance:
            observation.guidance = (
                (observation.guidance + "\n" + critic_guidance)
                if observation.guidance
                else critic_guidance
            )

        if is_loop:
            msg = f"CRITICAL WARNING (Loop Detected): State hash {observation.state_hash} is part of a visual loop. AVOID OLD STRATEGY HERE."
            task = asyncio.create_task(
                self.long_term_memory.add_memory(
                    msg,
                    {
                        "type": "loop_warning",
                        "vision_hash": observation.state_hash,
                        "map_id": map_id,
                    },
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # --- FEATURE: Odometer (Move Validation) ---
        is_explorable = "EXPLORABLE" in ctx.get("interface_mode", "")

        if (
            is_explorable
            and self.last_action_obj
            and self.last_button in ["up", "down", "left", "right"]
        ):
            steps = self.last_action_obj.repeat
            ex, ey = self.last_pos
            if self.last_button == "up":
                ey -= steps
            elif self.last_button == "down":
                ey += steps
            elif self.last_button == "left":
                ex -= steps
            elif self.last_button == "right":
                ex += steps

            actual_pos = pos
            if actual_pos != (ex, ey) and map_id == self.last_map_id:
                bx, by = actual_pos
                if self.last_button == "up":
                    by -= 1
                elif self.last_button == "down":
                    by += 1
                elif self.last_button == "left":
                    bx -= 1
                elif self.last_button == "right":
                    bx += 1

                collision_msg = (
                    f"🚧 Odometer: Collision at ({bx}, {by}). This tile is BLOCKED."
                )
                print(collision_msg)
                self.collision_counter += 1
                
                # Update collision points for LLM context
                if (bx, by) not in self.collision_points:
                    self.collision_points.append((bx, by))
                    if len(self.collision_points) > 5:
                        self.collision_points.pop(0)

                if self.collision_counter >= 3:
                    print("🚨 CRITICAL COLLISION BURST: Forcing immediate escape.")
                    observation.guidance = (
                        (observation.guidance + "\n" + "STAGNATION_COLLISION_BURST")
                        if observation.guidance
                        else "STAGNATION_COLLISION_BURST"
                    )
                    self.collision_counter = 0

                observation.guidance = (
                    (observation.guidance + "\n" + collision_msg)
                    if observation.guidance
                    else collision_msg
                )

                task = asyncio.create_task(
                    self.long_term_memory.record_collision(map_id, bx, by)
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            elif actual_pos == (ex, ey) and map_id == self.last_map_id:
                self.collision_counter = 0
                task = asyncio.create_task(
                    self.long_term_memory.record_walkable(map_id, pos[0], pos[1])
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        if map_id != self.last_map_id and self.last_map_id != -1:
            self.transition_lock = 30  # I. Map Transition Cooldown
            self.collision_points = [] # Clear old blocked tiles for new map
            self.anchor_recovery_count = 0 # Reset recovery counter on progress
            
            # --- FEATURE: Map Transition Inertia ---
            # Detect if we just bounced back and forth
            now = time.time()
            if (now - self.last_transition_time) < 60:
                # If we've seen this map very recently, we might be oscillating
                if map_id in self.map_history:
                    print(f"🚨 OSCILLATION DETECTED: Bouncing back to Map #{map_id} too quickly.")
                    self.is_oscillating = True
            else:
                self.is_oscillating = False
            
            self.last_transition_time = now
            self.map_history.append(map_id)

            # Apply Directional Penalty (Entrance Vector Penalty)
            if self.last_button in ["up", "down", "left", "right"]:
                inverse_map = {"up": "down", "down": "up", "left": "right", "right": "left"}
                inverse_btn = inverse_map[self.last_button]
                self.actor.apply_action_penalty(inverse_btn, duration=500)
            # ---------------------------------------

            task = asyncio.create_task(
                self.long_term_memory.record_warp(
                    self.last_map_id, self.last_pos[0], self.last_pos[1]
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            last_map_name = self.config.profile.maps.get(
                self.last_map_id, f"Map #{self.last_map_id}"
            )
            curr_map_name = self.config.profile.maps.get(map_id, f"Map #{map_id}")
            warp_msg = f"WARP DISCOVERY: Transition from {last_map_name} {self.last_pos} to {curr_map_name} {pos}."
            print(f"🌀 {warp_msg}")

            # 1. Record to Session LTM
            task = asyncio.create_task(
                self.long_term_memory.add_memory(
                    warp_msg,
                    {
                        "type": "warp",
                        "from_map": self.last_map_id,
                        "to_map": map_id,
                        "from_pos": self.last_pos,
                        "to_pos": pos,
                    },
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            # 2. Record to Permanent Knowledge Base (RAG)
            if self.knowledge:
                asyncio.create_task(
                    self.knowledge.ingest_text(
                        f"{warp_msg} This coordinate is a door, stairs, or warp point.",
                        source="warp_discoveries",
                    )
                )

        if self.last_state_hash and self.last_action_obj:
            task = asyncio.create_task(
                self.long_term_memory.add_event_pattern(
                    state_hash=self.last_state_hash,
                    action=self.last_action_obj,
                    result_hash=observation.state_hash,
                    reward=reward_delta,
                    map_id=map_id,
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if self.memory.steps:
            last_record = self.memory.steps[-1]
            # Persist with hidden_state if available for MuZero/Predictive analysis
            task = asyncio.create_task(
                self.long_term_memory.add_step_to_replay(
                    self.session_id, self.step_count - 1, last_record
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if isinstance(map_id, int) and map_id != -1 and pos[0] is not None:
            if pos != (0, 0) or map_id == 0:
                task = asyncio.create_task(
                    self.long_term_memory.record_location(
                        map_id, pos[0], pos[1], state=1
                    )
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        # --- FEATURE: Proactive Dialogue Recording ---
        if ocr and len(ocr) > 3:
            # Check if this is a NEW dialogue we haven't noted in episodic memory yet
            if ocr not in self.memory.seen_dialogues:
                self.memory.seen_dialogues.add(ocr)

                btn_str = (
                    self.last_action_obj.button.upper()
                    if self.last_action_obj and self.last_action_obj.button
                    else "NONE"
                )
                discovery_text = f'DIALOGUE EVENT at {pos} on Map #{map_id}: I pressed {btn_str} and saw text: "{observation.state.ocr_text}"'

                # 1. Record to Session LTM (Vector + Metadata)
                task = asyncio.create_task(
                    self.long_term_memory.add_memory(
                        discovery_text,
                        {
                            "type": "dialogue_discovery",
                            "map_id": map_id,
                            "x": pos[0],
                            "y": pos[1],
                            "ocr": observation.state.ocr_text,
                            "button": btn_str,
                        },
                    )
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

                # 2. Record to Permanent Knowledge Base (RAG)
                if self.knowledge:
                    # We wrap it in a task to avoid blocking the main act loop
                    asyncio.create_task(
                        self.knowledge.ingest_text(
                            discovery_text, source=f"discovery_map_{map_id}"
                        )
                    )

                print(f"🗨️ Proactive Memory: Noted dialogue at {pos}")
        # ---------------------------------------------

        if self.last_action_obj and self.last_action_obj.reasoning:
            reason_text = f"Action: {self.last_action_obj.button.upper() if self.last_action_obj.button else 'NONE'} | Why: {self.last_action_obj.reasoning}"
            task = asyncio.create_task(
                self.long_term_memory.add_memory(
                    reason_text,
                    {
                        "map_id": map_id,
                        "type": "reasoning",
                        "x": pos[0],
                        "y": pos[1],
                        "vision_vector": observation.state.vision_vector,
                    },
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if len(self.memory.steps) > 0:
            last_step = list(self.memory.steps)[-1]
            if (
                last_step.button == "none"
                and self.last_action_obj
                and self.last_action_obj.macro
            ):
                trigger_vision = self.last_action_obj.trigger_vision

                is_trap = (
                    map_id == last_step.map_id
                    and pos == last_step.coords
                    and observation.state_hash == last_step.vision_hash
                )

                if is_trap or is_stuck or observation.state.vision_delta < 0.05:
                    reason = (
                        "Exact State Loop"
                        if is_trap
                        else ("Stagnation" if is_stuck else "Low Delta")
                    )
                    print(
                        f"🧹 AI realized a visual macro failed ({reason}: {observation.state.vision_delta:.2f}). Demoting it..."
                    )
                    if trigger_vision:
                        self.optimizer.promote_macro(trigger_vision, outcome=0.0)
                    elif last_step.vision_vector:
                        self.optimizer.promote_macro(
                            last_step.vision_vector, outcome=0.0
                        )
                elif reward_delta > 0.5 or observation.state.vision_delta > 0.1:
                    print(
                        f"📈 Visual macro was successful! (Delta: {observation.state.vision_delta:.2f}) Promoting reliability."
                    )
                    if trigger_vision:
                        self.optimizer.promote_macro(trigger_vision, outcome=1.0)
                    elif last_step.vision_vector:
                        self.optimizer.promote_macro(
                            last_step.vision_vector, outcome=1.0
                        )

        if reward_delta > 1.0 or map_id != self.last_map_id:
            intro_maps = self.config.heuristics.intro_map_ids if self.config else [0]
            is_intro_sequence = (
                (map_id in intro_maps) or ("NEW GAME" in ocr) or ("OPTION" in ocr)
            )

            if not is_intro_sequence:
                if (
                    self.last_map_id in intro_maps
                    and map_id not in intro_maps
                    and self.last_map_id != -1
                ):
                    if mcp_client:
                        print(
                            f"🎉 INTRO CLEARED: Saving Bootstrap Checkpoint to Slot {settings.bootstrap_slot}..."
                        )
                        asyncio.create_task(
                            mcp_client.call_tool(
                                "manage_checkpoint",
                                {"action": "save", "slot": settings.bootstrap_slot},
                            )
                        )

                # --- Macro Map Verification ---
                if map_id == self.last_map_id:
                    # Map DID NOT change. Check if we just ran a transition macro
                    if self.last_action_obj and self.last_action_obj.macro:
                        reasoning = self.last_action_obj.reasoning or ""
                        if "SKIP_INTRO" in reasoning.upper() or (
                            self.last_action_obj.button
                            and "SKIP_INTRO" in self.last_action_obj.button.upper()
                        ):
                            # The macro failed to change the map
                            print(
                                "⚠️ SKILL_SKIP_INTRO failed to transition map. Adding cooldown."
                            )
                            # Add a temporary 'Forbidden' entry for the intro state hash
                            # We store the step count far in the future to act as a 60s cooldown (approx 60 steps)
                            self.macro_history[observation.state_hash] = (
                                "SKILL_SKIP_INTRO",
                                self.step_count + 60,
                            )

                if map_id != self.last_map_id:
                    self.maps_discovered += 1
                    print(
                        f"✨ New Area Discovered: Map #{map_id}. Saving 'Gold' state to Slot {settings.bootstrap_slot}."
                    )
                    if mcp_client:
                        asyncio.create_task(
                            mcp_client.call_tool(
                                "manage_checkpoint",
                                {"action": "save", "slot": settings.bootstrap_slot},
                            )
                        )

                    asyncio.create_task(
                        self.reflector.analyze_session(
                            self.session_id, self.long_term_memory, limit=50
                        )
                    )

                    desc = f"ACHIEVEMENT: Entered Map #{map_id} at {pos}."
                    task = asyncio.create_task(
                        self.long_term_memory.add_memory(
                            desc, {"map_id": map_id, "type": "location"}
                        )
                    )
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

                if (
                    self.step_count - self.last_save_step
                    > settings.rolling_save_interval
                    or reward_delta > 5.0
                ):
                    self.stagnation_counter = 0
                    self.last_success_step = self.step_count
                    self.last_save_step = self.step_count
                    if mcp_client:
                        print(
                            f"💾 Milestone Achieved: Auto-Saving state to Slot {settings.tas_trigger_slot}..."
                        )
                        asyncio.create_task(
                            mcp_client.call_tool(
                                "manage_checkpoint",
                                {"action": "save", "slot": settings.tas_trigger_slot},
                            )
                        )

    def _get_session_metrics(self) -> dict:
        return {
            "total_steps": self.step_count,
            "maps_discovered": self.maps_discovered,
            "last_save_step": self.last_save_step,
        }
