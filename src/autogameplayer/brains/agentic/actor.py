import json
import random
from typing import Optional
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.utils.llm import LLMClientProtocol
from autogameplayer.utils.llm_utils import extract_json_from_llm_response
from .memory import EpisodicMemory

class BaseActor:
    """Base class for specialized actors."""
    def __init__(self, client: LLMClientProtocol, model: str, buttons: list, config: GameConfig = None, ltm = None):
        self.client = client
        self.model = model
        self.buttons = buttons
        self.config = config
        self.ltm = ltm
        self.capabilities = None

    async def _ensure_capabilities(self, mcp_client):
        """Lazy-load the emulator's agnostic capabilities."""
        if not self.capabilities and mcp_client:
            try:
                raw_caps = await mcp_client.call_tool("get_capabilities")
                self.capabilities = json.loads(raw_caps)
            except Exception as e:
                print(f"⚠️ Failed to get capabilities: {e}")
                self.capabilities = {"core": "Unknown", "supported_buttons": self.buttons}

    async def _call_llm(self, prompt: str, obs: Observation, drift: bool = False) -> Action:
        try:
            print(f"⏳ Actor ({self.__class__.__name__}): Running Inference...")
            content = [{"type": "text", "text": prompt}]

            # Pass the last 3 frames for temporal context (Movement perception)
            if obs.state.temporal_image_sequence:
                for i, img_b64 in enumerate(obs.state.temporal_image_sequence):
                    desc = "Current Frame" if i == len(obs.state.temporal_image_sequence)-1 else f"Frame {i+1} (History)"
                    content.append({"type": "text", "text": f"--- {desc} ---"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            elif obs.state.image_data:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{obs.state.image_data}"}})

            response_text = await self.client.acreate_completion(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=200,
                temperature=0.7 if drift else 0.0,
                timeout=90.0
            )
            
            result = extract_json_from_llm_response(response_text)
            if not result:
                raise ValueError(f"Failed to extract JSON from Actor response: {response_text}")

            # CRITICAL FIX: Sanitize hallucinated button strings
            raw_btn = str(result.get("button", "a")).lower()
            if "|" in raw_btn:
                # If model sends "A|A|A", pick the first valid button
                result["button"] = raw_btn.split("|")[0].strip()
            elif " " in raw_btn:
                result["button"] = raw_btn.split(" ")[0].strip()
            
            # Ensure the button exists in the controller's capabilities
            # Use self.navigator.buttons or self.tactician.buttons if self.buttons is not set
            valid_buttons = self.buttons or []
            if not valid_buttons and hasattr(self, 'navigator'):
                valid_buttons = self.navigator.buttons
            
            if valid_buttons and result.get("button") not in valid_buttons:
                # If it's a valid button but just needs lowercase/cleanup
                cleaned = str(result.get("button", "")).lower().strip()
                if cleaned in valid_buttons:
                    result["button"] = cleaned
                else:
                    # Fallback to 'a' or first valid button
                    result["button"] = valid_buttons[0] if valid_buttons else "a"

            return result
        except Exception as e:
            print(f"⚠️ Actor Inference Failed: {e}")
            return {"button": random.choice(self.buttons) if self.buttons else "a", "reasoning": f"Fallback: {e}"}

    def _post_process_action(self, result: dict, ctx: dict, map_id: int, ocr_text: str, drift: bool = False) -> Action:
        save_slot = result.get("save_state")
        load_slot = result.get("load_state")
        
        intro_maps = self.config.heuristics.intro_map_ids if self.config else [0]
        is_intro_phase = (map_id in intro_maps) or ("NEW GAME" in ocr_text)
        
        jitter = random.randint(-2, 3)
        suggested_duration = 15 if is_intro_phase else 5
        default_duration = result.get("duration", suggested_duration) + jitter
        default_duration = max(1, default_duration)
        
        if is_intro_phase and default_duration < 15:
            default_duration = 15 + random.randint(0, 5)

        drift_chance = 0.30 if drift else 0.05
        chosen_button = str(result.get("button", "a")).lower()
        if chosen_button in ["up", "down", "left", "right"] and random.random() < drift_chance:
            drift_map = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"]}
            chosen_button = random.choice(drift_map.get(chosen_button, ["a"]))
            print(f"🌀 Action Drift: {chosen_button.upper()}")

        if "macro" in result and isinstance(result["macro"], list) and len(result["macro"]) > 0:
            macro_actions = []
            for step in result["macro"]:
                if isinstance(step, dict) and "button" in step:
                    macro_actions.append(Action(button=str(step["button"]).lower(), duration=int(step.get("frames", default_duration)), repeat=step.get("repeat", 1), until_visual_change=step.get("until_visual_change", False)))
                elif isinstance(step, str):
                    macro_actions.append(Action(button=step.lower(), duration=default_duration))
            return Action(macro=macro_actions, save_state=int(save_slot) if save_slot is not None else None, load_state=int(load_slot) if load_slot is not None else None, reasoning=result.get("reasoning", "Executing macro sequence"))
        else:
            target_x = result.get("target_x")
            target_y = result.get("target_y")
            target_coords = None
            if target_x is not None and target_y is not None:
                target_coords = (int(target_x), int(target_y))

            return Action(
                button=chosen_button if "button" in result else None,
                duration=default_duration,
                repeat=result.get("repeat", 1),
                until_visual_change=result.get("until_visual_change", False),
                target_coords=target_coords,
                save_state=int(save_slot) if save_slot is not None else None,
                load_state=int(load_slot) if load_slot is not None else None,
                scan_memory=result.get("scan_memory"),
                poke_memory=result.get("poke_memory"),
                scan_neighborhood=result.get("scan_neighborhood"),
                save_discovered_address=result.get("save_discovered_address"),
                reasoning=result.get("reasoning", "Executing plan")
            )

class NavigatorAgent(BaseActor):
    """Specialized in X/Y movement, collision, and world navigation."""
    async def get_next_action(self, obs: Observation, plan: dict, memory: EpisodicMemory, mcp_client=None, drift: bool = False, session_metrics: dict = None) -> Action:
        await self._ensure_capabilities(mcp_client)
        ctx, ocr_text = obs.state.context, (obs.state.ocr_text or "").upper()
        core_name, buttons = self.capabilities.get("core", "Unknown"), self.capabilities.get("supported_buttons", self.buttons)
        
        intro_guidance = ""
        if self.config and self.config.profile:
            intro_guidance = self.config.profile.intro_guidance

        session_summary = ""
        if session_metrics:
            session_summary = f"SESSION: {session_metrics['total_steps']:,} steps | {session_metrics['maps_discovered']} maps discovered | {session_metrics['badges']} badges | Last save: {session_metrics['steps_since_save']} steps ago"

        # Get Map Name from config
        map_name = "Unknown Area"
        if self.config and self.config.profile and ctx.get('map_id', -1) in self.config.profile.maps:
            map_name = self.config.profile.maps[ctx.get('map_id', -1)]

        # Fetch confirmed blocked tiles for this map
        blocked_summary = ""
        if self.ltm:
            blocked_summary = self.ltm.get_blocked_tiles_summary(ctx.get('map_id', -1))

        prompt = f"""
        You are the NAVIGATOR for an AI playing {self.config.description}.
        TASK: Get to the target coordinates or room. Focus on movement and spatial reasoning.
        {session_summary}
        
        CORE: {core_name} | BUTTONS: {buttons}
        
        --- STRATEGIC PLAN ---
        GOAL: {plan.get('goal')}
        STEPS: {plan.get('steps')}
        ABORT IF: {plan.get('abort_condition')}
        
        {intro_guidance}
        
        --- SYMBOLIC STATE ---
        {json.dumps(ctx.get('symbolic', {}), indent=2)}

        --- SEMANTIC VISION ---
        Minimap: {obs.state.minimap_summary}
        Detected Sprites: {obs.state.detected_sprites}

        --- RAW CONTEXT ---
        Area: {map_name} (Map #{ctx.get('map_id')})
        Pos: ({ctx.get('x')}, {ctx.get('y')}) | Facing: {ctx.get('facing')}
        {blocked_summary}
        HISTORY: {memory.get_recent_summary(5)}
        
        GUIDELINE:
        - SET COORDINATE GOALS: Prefer providing "target_x" and "target_y" to use the automated A* pathfinder.
        - Look at the Minimap/Sprites to identify stairs, doors, or NPCs.
        - If the path is clear, set a goal 5-10 tiles away in your desired direction.
        - WARNING: If tiles are marked BLOCKED, do not set them as target_x/y. Provide an alternative route.

        CRITICAL: Respond ONLY with a valid JSON object. Do not include markdown formatting, preamble, or any conversational text.

        OUTPUT: JSON {{"target_x": int, "target_y": int, "reasoning": "..."}}
        """
        result = await self._call_llm(prompt, obs, drift)
        return self._post_process_action(result, ctx, ctx.get('map_id', 0), ocr_text, drift)

class TacticianAgent(BaseActor):
    """Specialized in combat, HP management, and menu logic."""
    async def get_next_action(self, obs: Observation, plan: dict, memory: EpisodicMemory, mcp_client=None, drift: bool = False, session_metrics: dict = None) -> Action:
        await self._ensure_capabilities(mcp_client)
        ctx, ocr_text = obs.state.context, (obs.state.ocr_text or "").upper()
        core_name, buttons = self.capabilities.get("core", "Unknown"), self.capabilities.get("supported_buttons", self.buttons)
        
        intro_guidance = ""
        if self.config and self.config.profile:
            intro_guidance = self.config.profile.intro_guidance

        session_summary = ""
        if session_metrics:
            session_summary = f"SESSION: {session_metrics['total_steps']:,} steps | {session_metrics['maps_discovered']} maps discovered | {session_metrics['badges']} badges | Last save: {session_metrics['steps_since_save']} steps ago"

        prompt = f"""
        You are the TACTICIAN for an AI playing {self.config.description}.
        TASK: Win the battle or navigate the menu efficiently. Focus on HP, moves, and UI logic.
        {session_summary}
        
        CORE: {core_name} | BUTTONS: {buttons}
        
        --- STRATEGIC PLAN ---
        GOAL: {plan.get('goal')}
        STEPS: {plan.get('steps')}
        ABORT IF: {plan.get('abort_condition')}
        
        {intro_guidance}
        
        --- SYMBOLIC STATE ---
        {json.dumps(ctx.get('symbolic', {}), indent=2)}

        --- SEMANTIC VISION ---
        Minimap: {obs.state.minimap_summary}
        Detected Sprites: {obs.state.detected_sprites}

        --- RAW BATTLE STATE ---
        {ctx}
        OCR Text: "{ocr_text[:200]}"
        DIALOGUE ARROW VISIBLE: {obs.state.has_dialogue_arrow}
        HISTORY: {memory.get_recent_summary(5)}
        
        GUIDELINE:
        - Use "duration": 2 for ultra-fast taps if mashing through dialogue.
        - Use "duration": 5 for standard menu navigation.
        - Use "duration": 30 for long presses if the menu seems unresponsive.
        
        CRITICAL:
        - If reading dialogue, ONLY press 'a' or 'b' if DIALOGUE ARROW VISIBLE is True. 
        - If the arrow is NOT visible, the game is still printing text. Wait by outputting a small duration or a different action.
        - Respond ONLY with a valid JSON object. Do not include markdown formatting, preamble, or any conversational text.

        OUTPUT: JSON {{"button": "a|b|up|down|start", "duration": int, "reasoning": "..."}}
        """
        result = await self._call_llm(prompt, obs, drift)
        return self._post_process_action(result, ctx, ctx.get('map_id', 0), ocr_text, drift)

class PolicyAgent:
    """Handles common, high-confidence rule-based decisions to save LLM tokens."""
    def get_action(self, obs: Observation, plan: dict) -> Optional[Action]:
        # 0. Respect the Critic: Abort hardcoded policy if stuck in a visual loop
        if obs.guidance and "LOOP DETECTED" in obs.guidance.upper():
            return None

        ctx = obs.state.context
        goal = plan.get('goal', '').lower()
        
        # 1. Dialogue Policy: If the arrow is visible, press A until the screen changes.
        if ctx.get('is_dialogue'):
            if obs.state.has_dialogue_arrow:
                return Action(
                    button="a", 
                    duration=5, 
                    until_visual_change=True,
                    reasoning="Policy: Dialogue arrow visible, advancing until text clears."
                )
            else:
                # Still printing, wait 15 frames
                return Action(button="none", duration=15, reasoning="Policy: Text still printing, waiting.")

        # 2. Simple Navigation Policy
        directions = ["up", "down", "left", "right"]
        for d in directions:
            if f"walk {d}" in goal or f"move {d}" in goal or f"go {d}" in goal:
                return Action(
                    button=d, 
                    duration=10, 
                    until_visual_change=True,
                    reasoning=f"Policy: Fulfilling directional goal '{d}' until environment change."
                )

        # 3. Menu Breakout Policy: Use stricter keywords to avoid spatial "exit" collisions
        menu_keywords = ["close menu", "exit menu", "escape menu", "close dialogue", "exit interface"]
        if any(k in goal for k in menu_keywords):
            if ctx.get('is_menu_open') or ctx.get('is_dialogue'):
                return Action(
                    button="b", 
                    duration=5, 
                    until_visual_change=True,
                    reasoning="Policy: Executing exit/close request until screen updates."
                )

        # 4. Naming Screen Policy
        naming_keywords = ["UPPER CASE", "LOWER CASE", "ED IT", "DELETE", " NAME "]
        ocr_upper = (obs.state.ocr_text or "").upper()
        if any(k in ocr_upper for k in naming_keywords):
            return Action(
                button="start", 
                duration=10, 
                until_visual_change=True,
                reasoning="Policy: Naming screen detected. Pressing START to accept default name."
            )

        return None

class LearnedPolicyAgent:
    """Uses the replay buffer to reuse high-reward actions, with anti-spam safeguards."""
    def __init__(self, ltm):
        self.ltm = ltm
        self.recently_used = [] # Track recent button presses to avoid infinite spam

    def get_action(self, obs: Observation) -> Optional[Action]:
        if not self.ltm: return None
        
        # 1. Respect the Critic: If a loop is detected, refuse to reuse the policy
        if obs.guidance and "LOOP DETECTED" in obs.guidance.upper():
            self.recently_used.clear()
            return None
            
        # Query LTM for a highly successful action in this visual state
        # Threshold 0.96 for high-confidence policy reuse
        best_data = self.ltm.get_successful_action(obs.state.vision_vector, threshold=0.96)
        
        if best_data and best_data.get('button') and best_data['button'] != "none":
            button = best_data['button']
            
            # 2. Anti-Spam Safeguard: Force exploration if the same learned action is abused
            if self.recently_used.count(button) >= 3:
                # print(f"🚫 Policy Safeguard: Detected spam of {button.upper()}. Forcing LLM re-think.")
                self.recently_used.clear()
                return None # Fallback to LLM Tactician/Navigator
                
            self.recently_used.append(button)
            if len(self.recently_used) > 5:
                self.recently_used.pop(0)
                
            return Action(
                button=button,
                duration=best_data['duration'],
                reasoning=f"Learned Policy: Reusing high-reward action (Reward: {best_data['reward']:.2f})."
            )
        return None

class ActorAgent:
    """Orchestrator that chooses between Policy, Navigator and Tactician."""
    def __init__(self, client: LLMClientProtocol, model: str, buttons: list, config: GameConfig = None, ltm = None, optimizer = None):
        self.navigator = NavigatorAgent(client, model, buttons, config, ltm=ltm)
        self.tactician = TacticianAgent(client, model, buttons, config, ltm=ltm)
        self.policy = PolicyAgent()
        self.learned_policy = LearnedPolicyAgent(ltm)
        self.optimizer = optimizer
        self.ltm = ltm

    async def get_next_action(self, obs: Observation, plan: dict, memory: EpisodicMemory, mcp_client=None, drift: bool = False, session_metrics: dict = None) -> Action:
        # 1. Try High-Confidence Manual Policy first (0 Token cost)
        policy_action = self.policy.get_action(obs, plan)
        if policy_action and not drift:
            print(f"⚡ Policy Action: {policy_action.button.upper() if policy_action.button else 'WAIT'} | {policy_action.reasoning}")
            return policy_action

        # 2. Try Learned Policy (Historical successful actions)
        learned_action = self.learned_policy.get_action(obs)
        if learned_action and not drift:
            print(f"🧠 Learned Policy: {learned_action.button.upper()} | {learned_action.reasoning}")
            return learned_action

        # 3. Otherwise fall back to Specialized LLM Agents
        from autogameplayer.core.models import GameStage
        stage = obs.state.stage
        
        if stage == GameStage.BATTLE or stage == GameStage.MENU:
            action = await self.tactician.get_next_action(obs, plan, memory, mcp_client, drift, session_metrics=session_metrics)
        else:
            action = await self.navigator.get_next_action(obs, plan, memory, mcp_client, drift, session_metrics=session_metrics)

        # 4. Resolve Named Skills (e.g. SKILL_DIALOGUE_SKIP)
        if action.button and action.button.upper().startswith("SKILL_") and self.optimizer:
            skill_name = action.button.upper()
            
            # --- FEATURE: Dialogue Busy Logic ---
            # If we are in dialogue or a naming screen, prioritize UI macros over exploration
            ocr_upper = (obs.state.ocr_text or "").upper()
            naming_keywords = ["UPPER CASE", "LOWER CASE", "ED IT", "DELETE", " NAME "]
            is_naming = any(k in ocr_upper for k in naming_keywords)
            is_busy = obs.state.has_dialogue_arrow or is_naming
            
            # If busy, only allow 'SKIP' or 'DIALOGUE' named skills
            if is_busy and not any(k in skill_name for k in ["SKIP", "DIALOGUE", "CONFIRM"]):
                print(f"🚫 Blocking exploration macro {skill_name} because Dialogue/UI is busy.")
                # Fallback to a simple A press or the model's base action
                return Action(button="a", duration=5, reasoning="Dialogue Busy: Falling back to A press instead of exploration macro.")
            
            seq = self.optimizer.resolve_skill(skill_name)
            if seq:
                print(f"🧩 Resolving named skill: {skill_name}")
                self.optimizer.record_macro_usage_by_name(skill_name)
                macro_actions = [Action(**step) for step in seq]
                # Update the action to be a macro
                action = Action(
                    macro=macro_actions,
                    reasoning=f"Executing named skill: {skill_name} | {action.reasoning}"
                )
        
        return action
