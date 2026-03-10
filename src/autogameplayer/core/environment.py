import asyncio
from typing import List, Tuple
from autogameplayer.core.interfaces import Environment, RewardFunction
from autogameplayer.core.models import GameState, Action, Observation
from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.utils.state_tracker import StateTracker

class EmulatorEnvironment(Environment):
    """High-level Environment with State Deduplication and Macro support."""
    def __init__(self, client: MCPClient, reward_functions: List[RewardFunction] = None):
        self.client = client
        self.reward_functions = reward_functions or []
        self.state_tracker = StateTracker()
        self.last_observation: Observation = None

    async def _create_observation(self, state: GameState) -> Observation:
        state_hash = self.state_tracker.get_hash(state)
        
        # Fetch user guidance if available
        guidance = None
        try:
            guidance = await self.client.call_tool("get_guidance", {})
            if guidance == '""' or not guidance: guidance = None
        except Exception: pass

        obs = Observation(state=state, state_hash=state_hash, guidance=guidance)
        self.state_tracker.mark_visited(state_hash)
        return obs

    async def reset(self) -> Observation:
        state_json = await self.client.call_tool("get_game_state", {"include_ocr": True})
        state = GameState.model_validate_json(state_json)
        self.state_tracker.reset()
        self.last_observation = await self._create_observation(state)
        return self.last_observation

    async def step(self, action: Action) -> Tuple[Observation, float, bool]:
        hacker_tool_output = ""
        
        # --- Handle Hacker Tools ---
        if action.scan_memory:
            args = {"condition": action.scan_memory} if isinstance(action.scan_memory, str) else action.scan_memory
            res = await self.client.call_tool("scan_memory", args)
            hacker_tool_output += f"\nMEMORY SCAN RESULT: {res}\n"
            
        if action.poke_memory:
            res = await self.client.call_tool("poke_memory", action.poke_memory)
            hacker_tool_output += f"\nMEMORY POKE RESULT: {res}\n"
            
        if action.scan_neighborhood:
            res = await self.client.call_tool("scan_neighborhood", {"address": action.scan_neighborhood})
            hacker_tool_output += f"\nNEIGHBORHOOD RESULT: {res}\n"
            
        if action.save_discovered_address:
            res = await self.client.call_tool("save_discovered_address", action.save_discovered_address)
            hacker_tool_output += f"\nSAVE ADDRESS RESULT: {res}\n"
        # ---------------------------

        # --- Handle AI Save/Load Requests ---
        if action.save_state is not None:
            print(f"💾 AI requested SAVE to slot {action.save_state}")
            await self.save_state(action.save_state)
            
        if action.load_state is not None:
            print(f"⏪ AI requested LOAD from slot {action.load_state}")
            new_obs = await self.load_state(action.load_state)
            return new_obs, 0.0, False
        # -----------------------------------------

        # --- FEATURE: Emulator State Branching (Tree Search) ---
        if action.explore_branches:
            print(f"🌿 AI is exploring {len(action.explore_branches)} timeline branches...")
            # Save the 'root' state to a temporary slot (using slot 99 for internal branching)
            await self.save_state(99)
            
            best_branch_reward = -float('inf')
            best_branch_obs = None
            best_branch_index = -1
            
            for i, branch_actions in enumerate(action.explore_branches):
                # Restore the root state before each attempt
                await self.load_state(99)
                
                branch_total_reward = 0.0
                last_branch_obs = None
                
                for sub_action in branch_actions:
                    last_branch_obs, reward, done = await self.step(sub_action)
                    branch_total_reward += reward
                    if done: break
                
                print(f"  Branch {i+1} Result: Reward {branch_total_reward:.2f}")
                if branch_total_reward > best_branch_reward:
                    best_branch_reward = branch_total_reward
                    best_branch_obs = last_branch_obs
                    best_branch_index = i
            
            print(f"✅ Selected Best Branch {best_branch_index+1} (Reward: {best_branch_reward:.2f})")
            # The last branch execution already left the emulator in its final state
            # but if it wasn't the BEST branch, we need to re-run the best one or restore its state.
            # For simplicity, we re-load the root and re-execute the best branch one last time
            # to ensure all state trackers and loggers are in sync.
            await self.load_state(99)
            final_reward = 0.0
            final_obs = None
            for sub_action in action.explore_branches[best_branch_index]:
                final_obs, reward, _ = await self.step(sub_action)
                final_reward += reward
            
            return final_obs, final_reward, False
        # --------------------------------------------------------

        # Handle Macros (Batched server-side execution)
        if action.macro:
            print(f"🎬 Executing Macro: {action.reasoning[:50]} ({len(action.macro)} steps)")
            # Convert Action objects to dictionaries for JSON serialization
            macro_list = []
            for sub in action.macro:
                macro_list.append({
                    "button": sub.button,
                    "frames": sub.duration
                })
            
            await self.client.call_tool("send_input", {
                "macro": macro_list,
                "reasoning": action.reasoning
            })
            
            # Wait for the total estimated macro duration
            total_frames = sum(m.get('frames', 10) for m in macro_list)
            await asyncio.sleep(total_frames / 60.0 + 0.1)
            
            # Re-sync state after macro
            state_json = await self.client.call_tool("get_game_state", {"include_ocr": True})
            state = GameState.model_validate_json(state_json)
            new_obs = await self._create_observation(state)
            self.last_observation = new_obs
            return new_obs, 0.0, False # Reward will be calculated in next tick or by process_step_outcome

        # Handle Single Button
        if action.button:
            # Implement repeat and until_visual_change logic
            for r in range(action.repeat):
                # 1. Send the input (only pass recognized fields)
                input_args = {
                    "button": action.button,
                    "duration": action.duration,
                    "reasoning": action.reasoning
                }
                await self.client.call_tool("send_input", input_args)
                
                # 2. Wait for the action to complete
                await asyncio.sleep(action.duration / 60.0 + 0.05)
                
                # 3. If until_visual_change is set, check if we should break
                if action.until_visual_change:
                    # Get fresh state to check for change
                    state_json = await self.client.call_tool("get_game_state", {"include_ocr": False})
                    state = GameState.model_validate_json(state_json)
                    if state.vision_delta > 0.05: # Significant change detected
                        break
        
        state_json = await self.client.call_tool("get_game_state", {"include_ocr": True})
        state = GameState.model_validate_json(state_json)
        new_observation = await self._create_observation(state)
        
        # Inject hacker results into next guidance
        if hacker_tool_output:
            if new_observation.guidance:
                new_observation.guidance += f"\n{hacker_tool_output}"
            else:
                new_observation.guidance = hacker_tool_output
        
        # Compute standard rewards
        reward = 0.0
        for rf in self.reward_functions:
            reward += await rf.compute(self.last_observation, new_observation)

        # Phase 1: Grounding - Inject reward into context for the Brain
        # Suppress negative rewards if the engine is locked or overworld is inactive
        # (e.g. during intro, menus, or cutscenes)
        is_dialogue = new_observation.state.context.get('is_dialogue', False)
        is_overworld_active = new_observation.state.context.get('is_overworld_active', True)
        is_engine_locked = new_observation.state.context.get('is_engine_locked', False)
        map_id = new_observation.state.context.get('map_id', -1)
        intro_maps = new_observation.state.context.get('intro_map_ids', [0])
        vision_delta = new_observation.state.vision_delta

        if is_dialogue or not is_overworld_active or is_engine_locked or map_id in intro_maps:
            # Phase 1.1: Suppress stagnation penalties during intro/cutscenes
            reward = max(0.0, reward)

            # Phase 1.2: Add small reward for visual movement to encourage mashing through menus
            if vision_delta > 0.1:
                reward += 0.1

        new_observation.state.context['last_reward'] = reward
        new_observation.state.context['last_repeat'] = action.repeat
        new_observation.state.context['is_branching'] = action.explore_branches is not None
        new_observation.state.context['branch_count'] = len(action.explore_branches) if action.explore_branches else 0

        self.last_observation = new_observation
        
        return new_observation, reward, False

    # Feature 1: Save-state branching
    async def save_state(self, slot: int):
        await self.client.call_tool("manage_checkpoint", {"action": "save", "slot": slot})

    async def load_state(self, slot: int) -> Observation:
        await self.client.call_tool("manage_checkpoint", {"action": "load", "slot": slot})
        return await self.reset() # Re-sync observation after load
