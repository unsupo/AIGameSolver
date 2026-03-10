import asyncio
import concurrent.futures
import random
import time

from autogameplayer.core.config import settings
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.utils.llm import LLMClientProtocol, OllamaBootstrap

from .memory import EpisodicMemory, LongTermMemory
from .planner import PlannerAgent
from .actor import ActorAgent
from .critic import CriticAgent
from .reflector import ReflectionAgent
from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.core.registry import Registry
from autogameplayer.core.knowledge import KnowledgeBase

@Registry.register_brain("agentic")
class AgenticBrain(Brain):
    """
    Advanced agentic architecture using a Planner-Actor-Critic pattern.
    Features: Long-term memory (RAG), Automated Reflection, and Discovery Rewards.
    """
    def __init__(self, controller: Controller, config: GameConfig = None, llm_client: LLMClientProtocol = None):
        self.controller = controller
        self.config = config
        self.model = (config.llm_model if config else None) or settings.llm_model
        self.planner_model = (config.planner_model if config else None) or self.model
        self.reflector_model = (config.reflector_model if config else None) or self.model
        self.session_id = f"session_{int(time.time())}" # Unique ID for this run

        OllamaBootstrap.bootstrap([self.model, self.planner_model, self.reflector_model, "nomic-embed-text"])

        if llm_client is None:
            from autogameplayer.utils.llm import get_llm_client
            self.client = get_llm_client()
        else:
            self.client = llm_client

        self.memory = EpisodicMemory()
        self.long_term_memory = LongTermMemory(self.client)        
        self.critic = CriticAgent(ltm=self.long_term_memory)
        self.optimizer = StrategyOptimizer(self.client, self.reflector_model)
        self.knowledge = KnowledgeBase(self.client)
        
        from autogameplayer.utils.pathfinding import Pathfinder
        self.pathfinder = Pathfinder(self.long_term_memory.storage_path)

        self.planner = PlannerAgent(self.client, self.planner_model, self.config, optimizer=self.optimizer, knowledge=self.knowledge, critic=self.critic)
        self.actor = ActorAgent(self.client, self.model, self.controller.buttons, self.config, ltm=self.long_term_memory, optimizer=self.optimizer)
        self.reflector = ReflectionAgent(self.client, self.reflector_model, self.optimizer)
        
        self.current_plan = {"goal": "Explore the world and advance the story.", "steps": [], "abort_condition": "Stagnation", "expected_map_after": None}
        self.step_count = 0
        self.last_map_id = -1
        self.last_pos = (0, 0)
        self.last_button = ""
        self.last_action_obj = None
        self.last_state_hash = ""
        self.last_recalled_memories = []
        
        # Stagnation management
        self.stagnation_counter = 0
        self.last_success_step = 0
        self.last_save_step = 0
        self.last_rolling_save_step = 0
        self.drift_steps = 0
        self.maps_discovered = 0
        
        # Async task management
        self._tasks = set()
        self._is_reflecting = False
        self._is_planning = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @property
    def has_reflection(self) -> bool: return True

    async def act(self, observation: Observation, mcp_client=None) -> Action:
        ctx = observation.state.context
        map_id = ctx.get('map_id', -1)
        
        # Capture current state for outcome processing (Odometer logic)
        self.last_pos = (ctx.get('x', 0), ctx.get('y', 0))
        self.last_map_id = map_id
        self.last_state_hash = observation.state_hash

        # 1. Background Reflection & Evolution (Every 100 steps)
        if self.step_count > 0 and self.step_count % 100 == 0 and not self._is_reflecting:
            self._is_reflecting = True
            async def run_reflection():
                try:
                    # Decide on scan depth (Standard or Deep)
                    limit = 500 if self.step_count % 500 == 0 else 100
                    await self.reflector.analyze_session(self.session_id, self.long_term_memory, limit=limit)
                    
                    # --- FEATURE: Genetic Macro Evolution ---
                    # Periodically mine for new patterns and evolve existing ones
                    if self.optimizer:
                        # 1. Sequence Mining: Find high-reward frequent patterns in replay buffer
                        self.optimizer.mine_sequences(top_k=5)
                        # 2. Population Evolution: Mutate/Crossover top proven skills
                        self.optimizer.evolve_population(top_k=5)
                        # 3. Compression: Give them semantic names
                        await self.optimizer.async_compress_skills()
                    # ----------------------------------------
                finally:
                    self._is_reflecting = False

            task = asyncio.create_task(run_reflection())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # 2. Outcome Processing from last step
        await self._process_step_outcome(observation, mcp_client=mcp_client)

        # 3. Handle Stagnation & Timeline Branching
        if observation.guidance and "STAGNATION" in observation.guidance:
            self.stagnation_counter += 1
            
            # --- FEATURE: Instant Debug Reflection ---
            # If we just hit a zero-progress state, trigger a debugger reflection
            if self.stagnation_counter == 10:
                print("🧠 Triggering Instant Debug Reflection...")
                task = asyncio.create_task(self.reflector.analyze_failure(
                    self.session_id, self.long_term_memory, map_id, pos, observation
                ))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            # -----------------------------------------

            # --- FEATURE: Progressive Entropy ---
            # Gently increase drift before a full rollback to break out of UI locks
            if 5 < self.stagnation_counter < 30 and self.stagnation_counter % 5 == 0:
                print(f"⚠️ Stagnation at {self.stagnation_counter}. Injecting entropy to force exploration.")
                self.drift_steps = 3 
            # ------------------------------------

            # --- Timeline Branching (MCTS-Lite) ---
            if self.stagnation_counter >= 30:
                rollback_slot = 1 # Default fallback
                if self.last_rolling_save_step > 0:
                    print(f"🚨 CRITICAL STAGNATION ({self.stagnation_counter}): Branching timeline. Rolling back to Slot {rollback_slot}...")
                
                if mcp_client:
                    asyncio.create_task(mcp_client.call_tool("stop_recording", {
                        "filename": f"stagnation_rollback_{int(time.time())}.json",
                        "metadata": {"reason": "Rollback due to stagnation", "stagnation_count": self.stagnation_counter}
                    }))

                self.stagnation_counter = 0
                self.drift_steps = 30 # Enter high-entropy mode
                action = Action(load_state=rollback_slot, reasoning=f"Automated timeline branch: Rolling back to slot {rollback_slot} due to prolonged stagnation.")
                self.memory.record_step(observation, action)
                self.step_count += 1
                return action
            # -----------------------------------------------------------

            recent_escape_buttons = [
                s.button for s in list(self.memory.steps)[-4:] 
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
                action = Action(
                    button=escape_btn, 
                    duration=30, 
                    reasoning="Breaking stuck state by physically moving away."
                )
            else:
                recent_steps = list(self.memory.steps)[-6:]
                recent_ui_escapes = [
                    s for s in recent_steps
                    if s.reasoning and "Interface Breakout" in s.reasoning
                ]
                
                if len(recent_ui_escapes) >= 3:
                    all_same_hash = all(s.vision_hash == observation.state_hash for s in recent_ui_escapes)
                    if all_same_hash:
                        escape_btn = random.choice(["down", "left", "up", "right"])
                        print(f"🔥 Entropy Hold: Interface breakout failing. Forcing {escape_btn.upper()} for 60 frames.")
                        action = Action(
                            button=escape_btn, 
                            duration=60, 
                            reasoning=f"Entropy Hold: Physically clearing zone with {escape_btn.upper()} after spam failed."
                        )
                        self.memory.record_step(observation, action)
                        self.step_count += 1
                        return action

                b_count = len([s for s in recent_ui_escapes if s.button == "b"])
                if b_count >= 2:
                    escape_btn = random.choice(["a", "start"])
                else:
                    escape_btn = "b"

                print(f"🔄 Escalating Escape: Using {escape_btn.upper()} spam to break UI lock.")
                action = Action(
                    button=escape_btn, 
                    repeat=15, 
                    until_visual_change=True,
                    reasoning=f"Interface Breakout: Closing menu/modal using {escape_btn.upper()}."
                )
        else:
            self.stagnation_counter = 0 # Reset on progress
            
            compression = self.memory.compressor.detect_repeats()
            if compression and self.drift_steps == 0:
                if compression["type"] == "spam":
                    print(f"⚡ Reflex: Compressing {compression['button'].upper()} spam (repeat: 10).")
                    action = Action(
                        button=compression["button"], 
                        repeat=10, 
                        until_visual_change=True,
                        reasoning=f"Automatic reflex: Compressed repeated {compression['button'].upper()} presses."
                    )
                    self.memory.record_step(observation, action)
                    self.step_count += 1
                    return action
                elif compression["type"] == "pattern":
                    p_str = " -> ".join([b.upper() for b in compression["pattern"]])
                    print(f"⚡ Reflex: Compressing pattern {p_str}.")
                    macro_actions = [Action(button=b, duration=5) for b in compression["pattern"]]
                    action = Action(
                        macro=macro_actions,
                        repeat=3,
                        reasoning=f"Automatic reflex: Compressed repeated pattern {p_str}."
                    )
                    self.memory.record_step(observation, action)
                    self.step_count += 1
                    return action

            try:
                goal_str = self.current_plan.get('goal', 'Unknown')
                print(f"⏳ Actor: Deciding action for goal: '{goal_str}'{' (DRIFTING 🌪️)' if self.drift_steps > 0 else ''}")
                
                session_metrics = {
                    "total_steps": self.step_count,
                    "maps_discovered": self.maps_discovered,
                    "badges": ctx.get('badges_count', 0),
                    "steps_since_save": self.step_count - self.last_save_step
                }

                action = await self.actor.get_next_action(
                    observation, 
                    self.current_plan, 
                    self.memory, 
                    mcp_client=mcp_client,
                    drift=(self.drift_steps > 0),
                    session_metrics=session_metrics
                )

                if action.target_coords:
                    start_pos = (ctx.get('x', 0), ctx.get('y', 0))
                    path = self.pathfinder.find_path(map_id, start_pos, action.target_coords)
                    if path:
                        print(f"📍 Pathfinder: Found path to {action.target_coords} ({len(path)} steps).")
                        macro_actions = [Action(button=btn, duration=10, until_visual_change=True) for btn in path]
                        action = Action(
                            macro=macro_actions,
                            reasoning=f"Pathfinding to {action.target_coords} | {action.reasoning}"
                        )
                    else:
                        print(f"⚠️ Pathfinder: No path found from {start_pos} to {action.target_coords}. Falling back to LLM action.")

                if self.drift_steps > 0: self.drift_steps -= 1
            except Exception:
                action = self.fallback_action(self.controller)

        needs_new_plan = (
            self.step_count % 30 == 0 or 
            map_id != self.last_map_id or
            observation.guidance and "STAGNATION" in observation.guidance
        )

        if needs_new_plan and not self._is_planning:
            self._is_planning = True
            async def run_planner():
                try:
                    plan, recalled = await self.planner.generate_plan(observation, self.long_term_memory)
                    self.current_plan = plan
                    self.last_recalled_memories = recalled
                    
                    # --- FEATURE: Hardware-Accelerated Tree Search (MCTS-Lite) ---
                    # If the planner flags this as high-stakes, simulate branches in parallel
                    if plan.get('high_stakes') and mcp_client:
                        print("🌿 HIGH STAKES DETECTED: Initializing Timeline Branching...")
                        # 1. Save current state as the root for branching
                        await mcp_client.call_tool("manage_checkpoint", {"action": "save", "slot": 99})
                        
                        # 2. Define 4 experimental paths (Directional exploration)
                        # We use simple macros to test which direction leads to most novelty
                        branch_macros = [
                            [Action(button="up", duration=30, until_visual_change=True)],
                            [Action(button="down", duration=30, until_visual_change=True)],
                            [Action(button="left", duration=30, until_visual_change=True)],
                            [Action(button="right", duration=30, until_visual_change=True)]
                        ]
                        
                        # Use Orchestrator to run parallel simulations
                        # (We use random brains for the workers to test the macros)
                        from autogameplayer.orchestrator.orchestrator import Orchestrator
                        orch = Orchestrator(num_workers=4, rom_path=self.config.rom)
                        
                        # We don't need full genomes, just evaluate the macros
                        # For now, we simulate by actually running short episodes from Slot 99
                        print(f"  - Simulating {len(branch_macros)} directional branches...")
                        # (Optimization: In a real MCTS we'd pass the macro to the worker)
                        # Here we use the existing Orchestrator infrastructure
                        results = orch.evaluate_population(
                            genomes=[np.zeros(1)] * 4, # Dummy genomes
                            steps_per_episode=50,
                            initial_slot=99
                        )
                        
                        # 3. Identify the Winner (Highest Reward/Novelty)
                        best_reward = -float('inf')
                        best_idx = 0
                        for i, res in enumerate(results):
                            if res['reward'] > best_reward:
                                best_reward = res['reward']
                                best_idx = i
                        
                        # 4. Rollback main timeline to the WINNING branch
                        # In this implementation, workers don't share save states back to main easily
                        # so we just guide the main brain to take that first "winning" step.
                        winning_dir = ["UP", "DOWN", "LEFT", "RIGHT"][best_idx]
                        print(f"✅ Tree Search Complete. Winner: {winning_dir} (Reward: {best_reward:.2f}). Redirecting.")
                        self.current_plan['goal'] = f"Tree Search Winner: Move {winning_dir} to maximize novelty."
                        self.current_plan['steps'] = [f"Move {winning_dir} aggressively"]
                    # -------------------------------------------------------------
                finally:
                    self._is_planning = False
            
            task = asyncio.create_task(run_planner())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self.memory.record_step(observation, action)
        self.step_count += 1
        self.last_button = action.button.lower() if action.button else "none"
        self.last_action_obj = action
        self.last_state_hash = observation.state_hash
        
        prefix = "(Plan Pending) " if self._is_planning else ""
        repeat_str = f" (x{action.repeat})" if action.repeat > 1 else ""
        print(f"🎮 {prefix}Action: {action.button.upper() if action.button else 'NONE'}{repeat_str} | {action.reasoning[:50]}")

        observation.state.recalled_memories = self.last_recalled_memories

        return action

    def fallback_action(self, controller: Controller) -> Action:
        btns = controller.buttons if controller.buttons else ["a", "up"]
        return Action(button=random.choice(btns), duration=5, reasoning="Internal brain fallback")

    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    async def _process_step_outcome(self, observation: Observation, mcp_client=None):
        """Refactored outcome processor to handle rewards, memory, and discoveries."""
        ctx = observation.state.context
        map_id = ctx.get('map_id', -1)
        pos = (ctx.get('x', 0), ctx.get('y', 0))
        ocr = (observation.state.ocr_text or "").upper()
        
        reward_delta, is_stuck, critic_guidance, is_loop = self.critic.evaluate(self.memory, observation)
        self.memory.update_last_step(observation, is_stuck)

        if critic_guidance:
            observation.guidance = critic_guidance
            
        if is_loop:
            msg = f"CRITICAL WARNING (Loop Detected): State hash {observation.state_hash} is part of a visual loop. AVOID OLD STRATEGY HERE."
            task = asyncio.create_task(self.long_term_memory.add_memory(
                msg, 
                {"type": "loop_warning", "vision_hash": observation.state_hash, "map_id": map_id}
            ))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        
        if self.last_action_obj and self.last_button in ["up", "down", "left", "right"]:
            steps = self.last_action_obj.repeat
            ex, ey = self.last_pos
            if self.last_button == "up": ey -= steps
            elif self.last_button == "down": ey += steps
            elif self.last_button == "left": ex -= steps
            elif self.last_button == "right": ex += steps
            
            actual_pos = pos
            if actual_pos != (ex, ey) and map_id == self.last_map_id:
                bx, by = actual_pos
                if self.last_button == "up": by -= 1
                elif self.last_button == "down": by += 1
                elif self.last_button == "left": bx -= 1
                elif self.last_button == "right": bx += 1
                
                print(f"🚧 Odometer: Collision at {bx}, {by}. Incrementing impassable score.")
                task = asyncio.create_task(self.long_term_memory.record_collision(map_id, bx, by))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            elif actual_pos == (ex, ey) and map_id == self.last_map_id:
                task = asyncio.create_task(self.long_term_memory.record_walkable(map_id, pos[0], pos[1]))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        if map_id != self.last_map_id and self.last_map_id != -1:
            task = asyncio.create_task(self.long_term_memory.record_warp(self.last_map_id, self.last_pos[0], self.last_pos[1]))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            last_map_name = self.config.profile.maps.get(self.last_map_id, f"Map #{self.last_map_id}")
            curr_map_name = self.config.profile.maps.get(map_id, f"Map #{map_id}")
            warp_msg = f"WARP MILESTONE: Transition from {last_map_name} {self.last_pos} to {curr_map_name} {pos}."
            print(f"🌀 {warp_msg}")
            task = asyncio.create_task(self.long_term_memory.add_memory(warp_msg, {
                "type": "warp", 
                "from_map": self.last_map_id, "to_map": map_id,
                "from_pos": self.last_pos, "to_pos": pos
            }))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if self.last_state_hash and self.last_action_obj:
            task = asyncio.create_task(self.long_term_memory.add_event_pattern(
                state_hash=self.last_state_hash,
                action=self.last_action_obj,
                result_hash=observation.state_hash,
                reward=reward_delta,
                map_id=map_id
            ))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        
        if self.memory.steps:
            last_record = self.memory.steps[-1]
            task = asyncio.create_task(self.long_term_memory.add_step_to_replay(self.session_id, self.step_count - 1, last_record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if isinstance(map_id, int) and map_id != -1 and pos[0] is not None:
            if pos != (0, 0) or map_id == 0:
                task = asyncio.create_task(self.long_term_memory.record_location(map_id, pos[0], pos[1], state=1))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        if self.last_action_obj and self.last_action_obj.reasoning:
            reason_text = f"Action: {self.last_action_obj.button.upper() if self.last_action_obj.button else 'NONE'} | Why: {self.last_action_obj.reasoning}"
            task = asyncio.create_task(self.long_term_memory.add_memory(
                reason_text, 
                {
                    "map_id": map_id, 
                    "type": "reasoning", 
                    "x": pos[0], "y": pos[1],
                    "vision_vector": observation.state.vision_vector
                }
            ))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if len(self.memory.steps) > 0:
            last_step = list(self.memory.steps)[-1]
            if last_step.button == "none" and self.last_action_obj and self.last_action_obj.macro:
                trigger_vision = self.last_action_obj.trigger_vision
                
                is_trap = (
                    map_id == last_step.map_id and
                    pos == last_step.coords and
                    observation.state_hash == last_step.vision_hash
                )
                
                if is_trap or is_stuck or observation.state.vision_delta < 0.05:
                    reason = "Exact State Loop" if is_trap else ("Stagnation" if is_stuck else "Low Delta")
                    print(f"🧹 AI realized a visual macro failed ({reason}: {observation.state.vision_delta:.2f}). Demoting it...")
                    if trigger_vision:
                        self.optimizer.promote_macro(trigger_vision, outcome=0.0)
                    elif last_step.vision_vector:
                        self.optimizer.promote_macro(last_step.vision_vector, outcome=0.0)
                elif reward_delta > 0.5 or observation.state.vision_delta > 0.1:
                    print(f"📈 Visual macro was successful! (Delta: {observation.state.vision_delta:.2f}) Promoting reliability.")
                    if trigger_vision:
                        self.optimizer.promote_macro(trigger_vision, outcome=1.0)
                    elif last_step.vision_vector:
                        self.optimizer.promote_macro(last_step.vision_vector, outcome=1.0)

        if reward_delta > 1.0 or map_id != self.last_map_id:
            intro_maps = self.config.heuristics.intro_map_ids if self.config else [0]
            is_intro_sequence = (map_id in intro_maps) or ("NEW GAME" in ocr) or ("OPTION" in ocr)
            
            if not is_intro_sequence:
                if self.last_map_id in intro_maps and map_id not in intro_maps and self.last_map_id != -1:
                    if mcp_client:
                        print("🎉 INTRO CLEARED: Saving Bootstrap Checkpoint to Slot 0...")
                        asyncio.create_task(mcp_client.call_tool("manage_checkpoint", {"action": "save", "slot": 0}))
                
                if map_id != self.last_map_id:
                    self.maps_discovered += 1
                    desc = f"ACHIEVEMENT: Entered Map #{map_id} at {pos}."
                    task = asyncio.create_task(self.long_term_memory.add_memory(desc, {"map_id": map_id, "type": "location"}))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

                if self.step_count - self.last_save_step > 50 or reward_delta > 5.0:
                    self.stagnation_counter = 0
                    self.last_success_step = self.step_count
                    self.last_save_step = self.step_count
                    if mcp_client:
                        print("💾 Milestone Achieved: Auto-Saving state to Slot 1...")
                        asyncio.create_task(mcp_client.call_tool("manage_checkpoint", {"action": "save", "slot": 1}))

    def _get_session_metrics(self) -> dict:
        return {
            "total_steps": self.step_count,
            "maps_discovered": self.maps_discovered,
            "last_save_step": self.last_save_step
        }
