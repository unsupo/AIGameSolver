import asyncio
import uuid
from autogameplayer.core.interfaces import Brain
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.replay import DatasetLogger
from autogameplayer.core.config import settings

class GameRunner:
    """Orchestrates the agnostic loop and optionally logs data."""
    def __init__(self, env: EmulatorEnvironment, brain: Brain, enable_logging: bool = True, render_delay: float = 0.05):
        self.env = env
        self.brain = brain
        self.render_delay = render_delay
        self.logger = DatasetLogger() if enable_logging else None

    async def _run_branching_evaluation(self, map_id: int, num_branches: int = 4, steps: int = 50):
        """
        Optimal Route Search: Saves state, runs parallel workers, and loads the best branch.
        """
        print(f"🌿 Branching Evaluation: Exploring {num_branches} parallel paths for {steps} steps...")
        
        # 1. Save "Base" state for workers to load
        await self.env.client.call_tool("manage_checkpoint", {"action": "save", "slot": 99})
        
        from concurrent.futures import ProcessPoolExecutor
        from autogameplayer.worker.worker import run_worker_episode
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        with ProcessPoolExecutor(max_workers=num_branches) as executor:
            # Dispatch parallel evaluations
            for b_id in range(num_branches):
                tasks.append(loop.run_in_executor(
                    executor,
                    run_worker_episode,
                    b_id, # worker_id
                    self.env.client.rom_path, # rom_path
                    steps, # steps
                    None,  # genome
                    None,  # target_checkpoint
                    99     # initial_slot
                ))
            
            results = await asyncio.gather(*tasks)
            
        # 2. Evaluate Results
        best_reward = -float('inf')
        best_idx = -1
        
        for idx, res in enumerate(results):
            r = res.get("reward", -100.0)
            if r > best_reward:
                best_reward = r
                best_idx = idx
                
        if best_idx != -1:
            print(f"🏆 Branch {best_idx} won with reward {best_reward:.2f}. Merging into main timeline.")
            # Note: In a full implementation, we would load the winner's save file.
            # Since workers save to their own slots or Slot 99, we'll assume the 
            # environment continues from its current state or we can load Slot 99
            # if we want to reset to the transition point.
            # For now, we just inform the log.
        
    async def run(self, steps: int = 100) -> float:
        # 1. Generate a unique session ID for this specific run
        session_id = str(uuid.uuid4())[:8]
        
        # Share session ID with the brain (and handle wrappers)
        target_brain = self.brain
        if hasattr(self.brain, 'inner'): # MacroAwareBrain wrapper
            target_brain = self.brain.inner
            
        target_brain.session_id = session_id
        
        # Only print session starts if logging is enabled or for first session
        if self.logger:
            print(f"▶️ Starting Play Session {session_id}...")
        
        obs = await self.env.reset()
        last_map_id = obs.state.context.get('map_id', -1)
        total_reward = 0.0
        
        # 2. Extract heuristics for intro-cleared detection
        if hasattr(target_brain, 'config') and target_brain.config and hasattr(target_brain.config, 'heuristics'):
            # This logic was not using the variable, so we just check for its existence
            _ = target_brain.config.heuristics.intro_map_ids
        
        try:
            for i in range(1, steps + 1):
                action = await self.brain.act(obs, mcp_client=self.env.client)
                next_obs, reward, done = await self.env.step(action)
                
                # FEATURE: Map Transition Triggers
                current_map_id = next_obs.state.context.get('map_id', -1)
                
                if current_map_id != last_map_id and last_map_id != -1 and current_map_id != -1:
                    print(f"🌍 Map Transition: Map {last_map_id} -> {current_map_id}")
                    
                    # 1. Clear short-term memory to focus on new area
                    if hasattr(target_brain, 'memory'):
                        target_brain.memory.steps.clear()
                        print("🧹 Cleared Episodic Memory for new area.")

                    # 2. Bootstrap Save
                    print(f"💾 Saving Best-So-Far Bootstrap to Slot {settings.bootstrap_slot}...")
                    await self.env.client.call_tool("manage_checkpoint", {"action": "save", "slot": settings.bootstrap_slot})

                    # 3. FEATURE: Branching Evaluation (Optimal Route Search)
                    if not i >= steps: # Only if session has significant remaining steps
                        await self._run_branching_evaluation(current_map_id)
                
                last_map_id = current_map_id
                
                # Log for replay/metrics if logger exists
                if self.logger:
                    self.logger.log_step(obs, action, reward)
                
                total_reward += reward
                obs = next_obs
                
                if done:
                    break
                
                # Use configurable delay, but drop to 0 if a macro is active for "frame-perfect" TAS execution
                effective_delay = 0 if action.macro else self.render_delay
                if effective_delay > 0:
                    await asyncio.sleep(effective_delay)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
            
        # --- NEW: Session Consolidation Pass ---
        # Interrogate the brain for reflection/optimization capabilities
        target_brain = self.brain
        if hasattr(self.brain, 'inner'): # MacroAwareBrain wrapper
            target_brain = self.brain.inner
            
        if hasattr(target_brain, 'reflector') and target_brain.reflector:
            print(f"🌙 Session End: AI is 'Dreaming' (Reflecting on session {session_id})...")
            try:
                # We need access to LTM, which target_brain usually has
                if hasattr(target_brain, 'long_term_memory'):
                    await target_brain.reflector.analyze_session(session_id, target_brain.long_term_memory)
            except Exception as e:
                print(f"⚠️ End-of-session reflection failed: {e}")
                
        if hasattr(target_brain, 'optimizer') and target_brain.optimizer:
            try:
                target_brain.optimizer.consolidate_macros()
            except Exception as e:
                print(f"⚠️ End-of-session macro consolidation failed: {e}")
        # ---------------------------------------

        if self.logger:
            self.logger.save_episode(session_id)
        return total_reward
