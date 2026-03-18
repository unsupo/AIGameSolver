import json
import random
from typing import Optional
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.utils.llm import LLMClientProtocol, extract_json_from_llm_response
from autogameplayer.utils.database import get_db_connection
from .memory import EpisodicMemory


def apply_universal_jitter(
    action: Action, buttons: list, chance: float = 0.05
) -> Action:
    """
    Periodically overrides the intended action with a random button press.

    Args:
        action: The original Action object from the brain.
        buttons: List of supported buttons (e.g., self.buttons).
        chance: Probability (0.0 to 1.0) of jittering this step.
    """
    if random.random() < chance:
        random_btn = random.choice(buttons)
        # Create a new action with a random duration to add variety
        jitter_duration = random.randint(5, 15)

        print(f"🎲 Universal Jitter: Overriding brain with random {random_btn.upper()}")

        return Action(
            button=random_btn,
            duration=jitter_duration,
            reasoning=f"Jitter Event: Forced random button '{random_btn}'",
        )

    return action


class BaseActor:
    """Base class for specialized actors."""

    def __init__(
        self,
        client: LLMClientProtocol,
        model: str,
        buttons: list,
        config: GameConfig = None,
        ltm=None,
    ):
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
                self.capabilities = {
                    "core": "Unknown",
                    "supported_buttons": self.buttons,
                }

    async def _call_llm(
        self, prompt: str, obs: Observation, drift: bool = False
    ) -> Action:
        try:
            print(f"⏳ Actor ({self.__class__.__name__}): Running Inference...")
            content = [{"type": "text", "text": prompt}]

            # Pass the last 3 frames for temporal context (Movement perception)
            if obs.state.temporal_image_sequence:
                for i, img_b64 in enumerate(obs.state.temporal_image_sequence):
                    desc = (
                        "Current Frame"
                        if i == len(obs.state.temporal_image_sequence) - 1
                        else f"Frame {i + 1} (History)"
                    )
                    content.append({"type": "text", "text": f"--- {desc} ---"})
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        }
                    )
            elif obs.state.image_data:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{obs.state.image_data}"
                        },
                    }
                )

            response_text = await self.client.acreate_completion(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=200,
                temperature=0.7 if drift else 0.0,
                timeout=90.0,
            )

            result = extract_json_from_llm_response(response_text)
            if not result:
                raise ValueError(
                    f"Failed to extract JSON from Actor response: {response_text}"
                )

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
            if not valid_buttons and hasattr(self, "navigator"):
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
            return {
                "button": random.choice(self.buttons) if self.buttons else "a",
                "reasoning": f"Fallback: {e}",
            }

    def _post_process_action(
        self, result: dict, ctx: dict, map_id: int, ocr_text: str, drift: bool = False
    ) -> Action:
        save_slot = result.get("save_state")
        load_slot = result.get("load_state")

        intro_maps = self.config.heuristics.intro_map_ids if self.config else [0]
        is_intro_phase = (map_id in intro_maps) or ("NEW GAME" in ocr_text)

        jitter = random.randint(-2, 3)
        suggested_duration = 15 if is_intro_phase else 5
        default_duration = result.get("duration", suggested_duration) + jitter
        default_duration = max(1, default_duration)

        # 2. THE UNIVERSAL JITTER (Interrupt Gatekeeper)
        # We check this BEFORE macro or standard action processing.
        # This allows it to break out of deterministic loops or "macro holes".
        # if random.random() < jitter_chance and self.buttons:
        #     chosen_button = random.choice(self.buttons)
        #     return Action(
        #         button=chosen_button.lower(),
        #         duration=random.randint(5, 15),
        #         reasoning="Universal Jitter: Periodic random button press."
        #     )

        if is_intro_phase and default_duration < 15:
            default_duration = 15 + random.randint(0, 5)

        chosen_button = str(result.get("button", "a")).lower()

        # --- FEATURE: Directional Inertia ---
        # If the last action was movement, and we are currently suggesting a movement,
        # add 'Inertia' to prevent jittering (Up-Down-Up-Down).
        last_button = ctx.get("last_button", "").lower()
        move_buttons = ["up", "down", "left", "right"]

        if last_button in move_buttons and chosen_button in move_buttons:
            if chosen_button != last_button:
                # 20% weight to continuing the old direction
                if random.random() < 0.2:
                    print(
                        f"🛤️ Directional Inertia: Staying with {last_button.upper()} to prevent jitter."
                    )
                    chosen_button = last_button
        # ------------------------------------

        drift_chance = 0.30 if drift else 0.05
        if (
            chosen_button in ["up", "down", "left", "right"]
            and random.random() < drift_chance
        ):
            drift_map = {
                "up": ["left", "right"],
                "down": ["left", "right"],
                "left": ["up", "down"],
                "right": ["up", "down"],
            }
            chosen_button = random.choice(drift_map.get(chosen_button, ["a"]))
            print(f"🌀 Action Drift: {chosen_button.upper()}")

        if (
            "macro" in result
            and isinstance(result["macro"], list)
            and len(result["macro"]) > 0
        ):
            macro_actions = []
            for step in result["macro"]:
                if isinstance(step, dict) and "button" in step:
                    macro_actions.append(
                        Action(
                            button=str(step["button"]).lower(),
                            duration=int(step.get("frames", default_duration)),
                            repeat=step.get("repeat", 1),
                            until_visual_change=step.get("until_visual_change", False),
                        )
                    )
                elif isinstance(step, str):
                    macro_actions.append(
                        Action(button=step.lower(), duration=default_duration)
                    )
            return Action(
                macro=macro_actions,
                save_state=int(save_slot) if save_slot is not None else None,
                load_state=int(load_slot) if load_slot is not None else None,
                reasoning=result.get("reasoning", "Executing macro sequence"),
            )
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
                reasoning=result.get("reasoning", "Executing plan"),
            )


class NavigatorAgent(BaseActor):
    """Specialized in X/Y movement, collision, and world navigation."""

    async def get_next_action(
        self,
        obs: Observation,
        plan: dict,
        memory: EpisodicMemory,
        mcp_client=None,
        drift: bool = False,
        session_metrics: dict = None,
        intrinsic_curiosity: float = 0.0,
    ) -> Action:
        await self._ensure_capabilities(mcp_client)
        ctx, ocr_text = obs.state.context, (obs.state.ocr_text or "").upper()
        core_name, buttons = (
            self.capabilities.get("core", "Unknown"),
            self.capabilities.get("supported_buttons", self.buttons),
        )

        intro_guidance = ""
        if self.config and self.config.profile:
            intro_guidance = self.config.profile.intro_guidance

        session_summary = ""
        if session_metrics:
            steps = session_metrics['total_steps']
            maps = session_metrics['maps_discovered']
            badges = session_metrics['badges']
            ss = session_metrics['steps_since_save']
            lock = session_metrics.get('transition_lock', 0)
            blocked_coords = session_metrics.get('collision_points', [])
            penalties = session_metrics.get('action_penalties', [])
            oscillating = session_metrics.get('is_oscillating', False)
            map_history = session_metrics.get('map_history', [])
            
            session_summary = f"SESSION: {steps:,} steps | {maps} maps discovered | {badges} badges | Last save: {ss} steps ago"
            if oscillating:
                session_summary += f"\n🚨 OSCILLATION DETECTED: You are bouncing between maps {map_history[-4:]}. You MUST break this loop by exploring DEEPER into the room. Do NOT go back through the door."
            if lock > 0:
                session_summary += "\n⚠️ TRANSITION LOCK: You just entered this map. Favor walking FORWARD/AWAY from the entrance to explore. Do NOT immediately turn back."
            if blocked_coords:
                session_summary += f"\n⛔ BLOCKED TILES (Recent Collisions): {', '.join([f'({x},{y})' for x, y in blocked_coords])}. Avoid these coordinates."
            if penalties:
                session_summary += f"\n🚫 FORBIDDEN ACTIONS (Doorway Inertia): {', '.join(penalties)}. Do NOT use these buttons; they lead back to the previous map."

        # Get Map Name from config
        map_name = "Unknown Area"
        if (
            self.config
            and self.config.profile
            and ctx.get("map_id", -1) in self.config.profile.maps
        ):
            map_name = self.config.profile.maps[ctx.get("map_id", -1)]

        # Fetch confirmed blocked tiles for this map
        blocked_summary = ""
        if self.ltm:
            blocked_summary = self.ltm.get_blocked_tiles_summary(ctx.get("map_id", -1))

        # Collision history from context
        collisions = ctx.get("collision_history", [])
        collision_text = ""
        if collisions:
            collision_text = f"\nRECENT COLLISIONS: {', '.join([f'({x},{y})' for x, y in collisions])}"

        stagnation_guidance = ""
        if "STAGNATION" in (obs.guidance or "") or "LOOP DETECTED" in (obs.guidance or ""):
            stagnation_guidance = f"\n⚠️ ALERT: You are currently STUCK or in a LOOP at ({ctx.get('x')}, {ctx.get('y')}). You MUST output a sequence of 3 to 10 actions (a 'macro') to aggressively break out of this state. For example: 'DOWN, DOWN, RIGHT, A' to navigate around a corner."
        
        if "CRITICAL_BOTTLENECK_PLAN_REQUIRED" in (obs.guidance or ""):
            stagnation_guidance = f"\n🚨 CRITICAL BOTTLENECK at ({ctx.get('x')}, {ctx.get('y')}): Standard recovery failed. You MUST provide a LONG-HORIZON plan (a macro of 8-15 steps) to reach a distant target or clear this area. Use the provided BLOCKED list to pathfind correctly."

        forbidden_buttons = []  # Optional: placeholder for future button disabling logic
        forbidden_text = ""
        if forbidden_buttons:
            forbidden_text = f"\n⚠️ FORBIDDEN ACTIONS: Do NOT use the following buttons: {', '.join(forbidden_buttons).upper()}. Choose a different path."

        prompt = f"""
        You are the NAVIGATOR for an AI playing {self.config.description if self.config else "Game"}.
        TASK: Get to the target coordinates or room. Focus on movement and spatial reasoning.
        {session_summary}
        {forbidden_text}
        
        CORE: {core_name} | BUTTONS: {buttons}
        
        --- STRATEGIC PLAN ---
        GOAL: {plan.get("goal")}
        STEPS: {plan.get("steps")}
        ABORT IF: {plan.get("abort_condition")}
        
        {intro_guidance}
        
        --- SYMBOLIC STATE ---
        {json.dumps(ctx.get("symbolic", {}), indent=2)}
        MENU_OPEN: {ctx.get("is_menu_open", False)}
        INTERFACE_MODE: {ctx.get("interface_mode", "Unknown")}

        --- SEMANTIC VISION ---
        Minimap: {obs.state.minimap_summary}
        Detected Sprites: {obs.state.detected_sprites}

        --- RAW CONTEXT ---
        Area: {map_name} (Map #{ctx.get("map_id")})
        Pos: ({ctx.get("x")}, {ctx.get("y")}) | Facing: {ctx.get("facing")}
        {blocked_summary}
        {collision_text}
        {stagnation_guidance}
        HISTORY: {memory.get_recent_summary(5)}
        
        TOOLS AVAILABLE:
        - list_waypoints: Use this to see all discovered Map IDs, (x, y) coordinates, and slot_ids for teleporting.
        - manage_checkpoint: Action 'load' with a slot_id from waypoints to teleport back to a known-good state.
        
        GUIDELINE:
        - MENU AWARENESS: If MENU_OPEN is True, do NOT suggest directional movement. Instead, use 'a', 'b', or 'start' to navigate the UI or exit.
        - STRATEGIC TELEPORT: If STAGNATION/LOOP is high, use 'list_waypoints' to find a nearby Discovery and 'manage_checkpoint' (load) to teleport back and try a new branch.
        - DIRECTIONAL INERTIA: If your last action was a movement (UP, DOWN, LEFT, RIGHT), favor continuing in that direction (+20% weight) unless you hit a wall or reached your goal. This prevents jittering.

        CRITICAL: Respond ONLY with a valid JSON object. Do not include markdown formatting, preamble, or any conversational text.

        OUTPUT FORMAT: 
        {{"button": "up|down|left|right|a|b|start|select", "reasoning": "...", "macro": [{{"button": "up", "frames": 10}}, {{"button": "right", "frames": 10}}]}}
        If a sequence is needed to break a loop, use the "macro" field.
        """
        result = await self._call_llm(prompt, obs, drift)
        return self._post_process_action(
            result, ctx, ctx.get("map_id", 0), ocr_text, drift
        )


class TacticianAgent(BaseActor):
    """Specialized in combat, HP management, and menu logic."""

    async def get_next_action(
        self,
        obs: Observation,
        plan: dict,
        memory: EpisodicMemory,
        mcp_client=None,
        drift: bool = False,
        session_metrics: dict = None,
        intrinsic_curiosity: float = 0.0,
    ) -> Action:
        await self._ensure_capabilities(mcp_client)
        ctx, ocr_text = obs.state.context, (obs.state.ocr_text or "").upper()
        core_name, buttons = (
            self.capabilities.get("core", "Unknown"),
            self.capabilities.get("supported_buttons", self.buttons),
        )

        intro_guidance = ""
        if self.config and self.config.profile:
            intro_guidance = self.config.profile.intro_guidance

        session_summary = ""
        if session_metrics:
            steps = session_metrics['total_steps']
            maps = session_metrics['maps_discovered']
            badges = session_metrics['badges']
            ss = session_metrics['steps_since_save']
            lock = session_metrics.get('transition_lock', 0)
            blocked_coords = session_metrics.get('collision_points', [])
            penalties = session_metrics.get('action_penalties', [])
            oscillating = session_metrics.get('is_oscillating', False)
            map_history = session_metrics.get('map_history', [])
            
            session_summary = f"SESSION: {steps:,} steps | {maps} maps discovered | {badges} badges | Last save: {ss} steps ago"
            if oscillating:
                session_summary += f"\n🚨 OSCILLATION DETECTED: You are bouncing between maps {map_history[-4:]}. You MUST break this loop by exploring DEEPER into the room. Do NOT go back through the door."
            if lock > 0:
                session_summary += "\n⚠️ TRANSITION LOCK: You just entered this map. Favor walking FORWARD/AWAY from the entrance to explore. Do NOT immediately turn back."
            if blocked_coords:
                session_summary += f"\n⛔ BLOCKED TILES (Recent Collisions): {', '.join([f'({x},{y})' for x, y in blocked_coords])}. Avoid these coordinates."
            if penalties:
                session_summary += f"\n🚫 FORBIDDEN ACTIONS (Doorway Inertia): {', '.join(penalties)}. Do NOT use these buttons; they lead back to the previous map."
        
        forbidden_buttons = []
        forbidden_text = ""
        if forbidden_buttons:
            forbidden_text = f"\n⚠️ FORBIDDEN ACTIONS: Do NOT use the following buttons: {', '.join(forbidden_buttons).upper()}."

        # Collision history from context
        collisions = ctx.get("collision_history", [])
        collision_text = ""
        if collisions:
            collision_text = f"\nRECENT COLLISIONS: {', '.join([f'({x},{y})' for x, y in collisions])}"

        stagnation_guidance = ""
        if "STAGNATION" in (obs.guidance or "") or "LOOP DETECTED" in (obs.guidance or ""):
            stagnation_guidance = f"\n⚠️ ALERT: You are currently STUCK or in a LOOP at ({ctx.get('x')}, {ctx.get('y')}). You MUST output a sequence of 3 to 10 actions (a 'macro') to aggressively break out of this state. For example: 'B, B, B, LEFT' to close a menu and walk away."

        if "CRITICAL_BOTTLENECK_PLAN_REQUIRED" in (obs.guidance or ""):
            stagnation_guidance = f"\n🚨 CRITICAL BOTTLENECK at ({ctx.get('x')}, {ctx.get('y')}): Standard recovery failed. Provide a LONG-HORIZON plan (8-15 actions) to break this interface lock or combat cycle."

        prompt = f"""
        You are the TACTICIAN for an AI playing {self.config.description if self.config else "Game"}.
        TASK: Win the battle or navigate the menu efficiently. Focus on HP, moves, and UI logic.
        {session_summary}
        {forbidden_text}
        
        CORE: {core_name} | BUTTONS: {buttons}
        
        --- STRATEGIC PLAN ---
        GOAL: {plan.get("goal")}
        STEPS: {plan.get("steps")}
        ABORT IF: {plan.get("abort_condition")}
        
        {intro_guidance}
        
        --- SYMBOLIC STATE ---
        {json.dumps(ctx.get("symbolic", {}), indent=2)}
        MENU_OPEN: {ctx.get("is_menu_open", False)}
        INTERFACE_MODE: {ctx.get("interface_mode", "Unknown")}

        --- SEMANTIC VISION ---
        Minimap: {obs.state.minimap_summary}
        Detected Sprites: {obs.state.detected_sprites}

        --- RAW BATTLE STATE ---
        {ctx}
        OCR Text: "{ocr_text[:200]}"
        DIALOGUE ARROW VISIBLE: {obs.state.has_dialogue_arrow}
        {collision_text}
        {stagnation_guidance}
        HISTORY: {memory.get_recent_summary(5)}
        
        TOOLS AVAILABLE:
        - list_waypoints: Use this to see all discovered Map IDs, (x, y) coordinates, and slot_ids for teleporting.
        - manage_checkpoint: Action 'load' with a slot_id from waypoints to teleport back to a known-good state.
        
        GUIDELINE:
        - MENU AWARENESS: If MENU_OPEN is True, focus on UI breakout (B spam) or targeted menu navigation.
        - STRATEGIC TELEPORT: If STAGNATION/LOOP is high, use 'list_waypoints' to find a nearby Discovery and 'manage_checkpoint' (load) to teleport back and try a new branch.
        - Use "duration": 2 for ultra-fast taps if mashing through dialogue.
        - Use "duration": 5 for standard menu navigation.
        - Use "duration": 30 for long presses if the menu seems unresponsive.
        
        CRITICAL:
        - If reading dialogue, ONLY press 'a' or 'b' if DIALOGUE ARROW VISIBLE is True. 
        - If the arrow is NOT visible, the game is still printing text. Wait by outputting a small duration or a different action.
        - Respond ONLY with a valid JSON object. Do not include markdown formatting, preamble, or any conversational text.

        OUTPUT FORMAT: 
        {{"button": "a|b|up|down|start", "reasoning": "...", "macro": [{{"button": "b", "frames": 5}}, {{"button": "left", "frames": 10}}]}}
        """
        result = await self._call_llm(prompt, obs, drift)
        return self._post_process_action(
            result, ctx, ctx.get("map_id", 0), ocr_text, drift
        )


class PolicyAgent:
    """Handles common, high-confidence rule-based decisions to save LLM tokens."""

    def get_action(self, obs: Observation, plan: dict) -> Optional[Action]:
        # 0. Respect the Critic: Abort hardcoded policy if stuck in a visual loop
        if obs.guidance and "LOOP DETECTED" in obs.guidance.upper():
            return None

        ctx = obs.state.context
        goal = plan.get("goal", "").lower()

        # 1. Simple Navigation Policy
        directions = ["up", "down", "left", "right"]
        for d in directions:
            if f"walk {d}" in goal or f"move {d}" in goal or f"go {d}" in goal:
                return Action(
                    button=d,
                    duration=10,
                    until_visual_change=True,
                    reasoning=f"Policy: Fulfilling directional goal '{d}' until environment change.",
                )

        # 3. Menu Breakout Policy: Use stricter keywords to avoid spatial "exit" collisions
        menu_keywords = [
            "close menu",
            "exit menu",
            "escape menu",
            "close dialogue",
            "exit interface",
        ]
        if any(k in goal for k in menu_keywords):
            if ctx.get("is_menu_open") or ctx.get("is_dialogue"):
                return Action(
                    button="b",
                    duration=5,
                    until_visual_change=True,
                    reasoning="Policy: Executing exit/close request until screen updates.",
                )

        # 4. Naming Screen Policy
        naming_keywords = ["UPPER CASE", "LOWER CASE", "ED IT", "DELETE", " NAME "]
        ocr_upper = (obs.state.ocr_text or "").upper()
        if any(k in ocr_upper for k in naming_keywords):
            return Action(
                button="start",
                duration=10,
                until_visual_change=True,
                reasoning="Policy: Naming screen detected. Pressing START to accept default name.",
            )

        return None


class LearnedPolicyAgent:
    """Uses the replay buffer to reuse high-reward actions, with anti-spam safeguards."""

    def __init__(self, ltm):
        self.ltm = ltm
        self.recently_used = []  # Track recent button presses to avoid infinite spam

    def get_action(self, obs: Observation) -> Optional[Action]:
        if not self.ltm:
            return None

        # 1. Respect the Critic: If a loop is detected, refuse to reuse the policy
        if obs.guidance and "LOOP DETECTED" in obs.guidance.upper():
            self.recently_used.clear()
            return None

        # --- NEW: Predictive Architecture (MuZero Hidden State) ---
        # Query LTM using the denoised latent vector if available.
        # This is much more robust than raw vision matching.
        best_data = None
        if obs.state.hidden_state:
            best_data = self.ltm.get_successful_action_by_hidden_state(
                obs.state.hidden_state, threshold=0.98
            )

        # Fallback to raw vision if hidden_state is missing or no match found
        if not best_data:
            best_data = self.ltm.get_successful_action(
                obs.state.vision_vector, threshold=0.96
            )

        if best_data and best_data.get("button") and best_data["button"] != "none":
            button = best_data["button"]

            # 2. Anti-Spam Safeguard: Force exploration if the same learned action is abused
            if self.recently_used.count(button) >= 3:
                # print(f"🚫 Policy Safeguard: Detected spam of {button.upper()}. Forcing LLM re-think.")
                self.recently_used.clear()
                return None  # Fallback to LLM Tactician/Navigator

            self.recently_used.append(button)
            if len(self.recently_used) > 5:
                self.recently_used.pop(0)

            return Action(
                button=button,
                duration=best_data["duration"],
                reasoning=f"Learned Policy: Reusing high-reward action (Reward: {best_data['reward']:.2f}).",
            )
        return None


class ActorAgent:
    """Orchestrator that chooses between Policy, Navigator and Tactician."""

    def __init__(
        self,
        client: LLMClientProtocol,
        model: str,
        buttons: list,
        config: GameConfig = None,
        ltm=None,
        optimizer=None,
    ):
        self.muzero_ratio = 0.5
        self.random_ratio = 0.5
        self.navigator = NavigatorAgent(client, model, buttons, config, ltm=ltm)
        self.tactician = TacticianAgent(client, model, buttons, config, ltm=ltm)
        self.policy = PolicyAgent()
        self.learned_policy = LearnedPolicyAgent(ltm)
        self.optimizer = optimizer
        self.ltm = ltm
        self.button_usage_map = {}  # map_id -> {button -> count}
        self.action_penalties = {}  # button -> remaining_steps

    def apply_action_penalty(self, button: str, duration: int = 500):
        """Applies a temporary weight penalty to a specific button."""
        self.action_penalties[button.lower()] = duration
        print(f"⛔ Action Penalty: {button.upper()} biased for {duration} steps.")

    def get_random_action(self, roll: float, map_id: int = -1, curr_x: int = 0, curr_y: int = 0):
        if map_id not in self.button_usage_map:
            self.button_usage_map[map_id] = {b.lower(): 0 for b in self.navigator.buttons}
        
        counts = self.button_usage_map[map_id]
        
        # --- FEATURE: Action Penalties (Doorway Inertia) ---
        # Reduce 'duration' of active penalties
        for btn in list(self.action_penalties.keys()):
            if self.action_penalties[btn] > 0:
                self.action_penalties[btn] -= 1
            else:
                del self.action_penalties[btn]

        # Filter buttons that are not heavily penalized
        penalized = [b for b, dur in self.action_penalties.items() if dur > 0]
        
        # --- FEATURE: Frontier Bias (Centroid weight) ---
        # If we have LTM, find the 'average' direction of exploration and push AWAY from it
        frontier_weights = {b: 1.0 for b in ["up", "down", "left", "right"]}
        if self.ltm:
            try:
                # Query local SLAM centroid
                with get_db_connection(self.ltm.storage_path) as conn:
                    cursor = conn.execute("SELECT AVG(x), AVG(y) FROM explored_locations WHERE map_id = ?", (map_id,))
                    avg_x, avg_y = cursor.fetchone()
                    if avg_x is not None and avg_y is not None:
                        # If average explored is West (avg_x < curr_x), weight EAST (right) higher
                        if avg_x < curr_x:
                            frontier_weights["right"] += 1.0
                        else:
                            frontier_weights["left"] += 1.0
                        
                        if avg_y < curr_y:
                            frontier_weights["down"] += 1.0
                        else:
                            frontier_weights["up"] += 1.0
            except Exception:
                pass

        # Choose from buttons with the minimum usage count to force exploration
        min_count = min(counts.values())
        least_used = [b for b, c in counts.items() if c == min_count]
        
        # --- ENHANCEMENT: Spatial Discovery Bias ---
        # If START or SELECT are in the candidates, only pick them 10% of the time
        # to prevent menu-loops during spatial discovery.
        utility_btns = ["start", "select"]
        if any(b in least_used for b in utility_btns):
            # If we have movement options that are also least used, prioritize them
            movement_candidates = [b for b in least_used if b not in utility_btns]
            if movement_candidates and random.random() < 0.90:
                least_used = movement_candidates
        # --------------------------------------------

        # If possible, pick a least used button that is NOT penalized
        candidates = [b for b in least_used if b not in penalized]
        if not candidates:
            candidates = least_used # Fallback if everything is used equally or penalized
        
        # Apply Frontier Weights to candidates if they are movement buttons
        if len(candidates) > 1:
            # Weighted random choice based on frontier_weights
            weights = [frontier_weights.get(b, 1.0) for b in candidates]
            chosen_btn = random.choices(candidates, weights=weights, k=1)[0]
        else:
            chosen_btn = random.choice(candidates)

        counts[chosen_btn] += 1
        
        return Action(
            button=chosen_btn.lower(),
            duration=random.randint(5, 12),
            reasoning=f"Directed Random: Minimum usage discovery ({roll:.2f} < {self.random_ratio})",
        )

    async def get_next_action(
        self,
        obs: Observation,
        plan: dict,
        memory: EpisodicMemory,
        mcp_client=None,
        drift: bool = False,
        session_metrics: dict = None,
        intrinsic_curiosity: float = 0.0,
    ) -> Action:
        ctx = obs.state.context
        map_id = ctx.get("map_id", -1)
        curr_x, curr_y = ctx.get("x", 0), ctx.get("y", 0)

        # PRIORITY 1: Physical Stuck State or Loop Detection (Route to LLM)
        is_stuck_or_loop = (
            "STAGNATION" in (obs.guidance or "")
            or "LOOP DETECTED" in (obs.guidance or "")
        )
        
        # If we are stuck, bypass random/MuZero and go straight to LLM
        if is_stuck_or_loop:
            print("🧠 Bottleneck/Collision detected. Routing to LLM Meta-Reasoner.")
            return await self._get_llm_action(obs, plan, memory, mcp_client, drift, session_metrics, intrinsic_curiosity)

        roll = random.random()

        if roll < self.random_ratio:
            return self.get_random_action(roll, map_id=map_id, curr_x=curr_x, curr_y=curr_y)
        # 2. OPTION: MUZERO (50% - Fast Path)
        # 0.40 to 0.90 (0.40 + 0.50)
        elif roll < (self.random_ratio + self.muzero_ratio):
            learned_action = self.learned_policy.get_action(obs)
            if learned_action:
                # If the learned action is penalized, ignore it and force random
                if learned_action.button and learned_action.button.lower() in self.action_penalties:
                    print(f"💨 Learned Policy blocked by penalty ({learned_action.button.upper()}).")
                else:
                    return learned_action

            # If MuZero has no data for this specific state
            print("💨 MuZero is clueless on this new map. Defaulting to Directed Random.")
            return self.get_random_action(roll, map_id=map_id, curr_x=curr_x, curr_y=curr_y)

        # Inject penalties into session metrics for LLM visibility
        if session_metrics:
            session_metrics["action_penalties"] = [b.upper() for b, d in self.action_penalties.items() if d > 0]

        return await self._get_llm_action(obs, plan, memory, mcp_client, drift, session_metrics, intrinsic_curiosity)

    async def _get_llm_action(
        self,
        obs: Observation,
        plan: dict,
        memory: EpisodicMemory,
        mcp_client=None,
        drift: bool = False,
        session_metrics: dict = None,
        intrinsic_curiosity: float = 0.0,
    ) -> Action:
        print("🐢 LLM Thinking Mode Triggered")
        # If MuZero doesn't have data for this specific screen,
        # we fall through to the LLM or a default.
        # --- FEATURE: Mental Dream Execution ---
        # If the planning phase found a high-reward mental sequence, execute it
        ctx = obs.state.context
        if "mental_dream" in ctx and ctx["mental_dream"]:
            dream = ctx["mental_dream"]
            print(f"🔮 Executing Mental Sequence: {' -> '.join(dream)}")
            macro_actions = [Action(button=btn, duration=5) for btn in dream]
            # Clear it so we don't repeat infinitely
            del ctx["mental_dream"]
            return Action(
                macro=macro_actions,
                reasoning="Executing high-reward sequence from internal dynamics model (Mental Dream).",
            )
        # ----------------------------------------

        # --- FEATURE: Text-Box Priority Mode (Dialogue Cruise Control) ---
        # if ctx.get('is_dialogue'): # todo make this optional
        #     print("🗨️ Dialogue Cruise Control: Auto-advancing text...")
        #     return Action(
        #         button="a",
        #         duration=2, # High-speed tap
        #         until_visual_change=True,
        #         reasoning="Dialogue Cruise Control: Progressing text automatically via RAM flag."
        #     )
        # ------------------------------------------------------------------

        # 1. Try High-Confidence Manual Policy first (0 Token cost)
        policy_action = self.policy.get_action(obs, plan)
        if policy_action:
            # UI policies (Dialogue/Naming) should ALWAYS trigger even during drift
            is_ui_policy = any(
                r in policy_action.reasoning
                for r in ["Dialogue", "Naming", "Interface"]
            )
            if is_ui_policy or not drift:
                print(
                    f"⚡ Policy Action: {policy_action.button.upper() if policy_action.button else 'WAIT'} | {policy_action.reasoning}"
                )
                return policy_action

        # 2. Try Learned Policy (Historical successful actions)
        learned_action = self.learned_policy.get_action(obs)
        if learned_action and not drift:
            print(
                f"🧠 Learned Policy: {learned_action.button.upper()} | {learned_action.reasoning}"
            )
            return learned_action

        # 3. Dynamic Curiosity Calculation
        # If last_reward was very negative (like from interaction boredom), spike curiosity
        last_reward = obs.state.context.get("last_reward", 0.0)
        current_curiosity = intrinsic_curiosity
        if last_reward < -1.0:
            current_curiosity = max(
                current_curiosity, 0.8
            )  # High pressure to find fresh dopamine
            print(
                f"🕵️ Curiosity Spike: reward={last_reward}, curiosity={current_curiosity:.2f}"
            )

        # 4. Otherwise fall back to Specialized LLM Agents
        from autogameplayer.core.models import GameStage

        stage = obs.state.stage

        if stage == GameStage.COMBAT or stage == GameStage.UI:
            action = await self.tactician.get_next_action(
                obs,
                plan,
                memory,
                mcp_client,
                drift,
                session_metrics=session_metrics,
                intrinsic_curiosity=current_curiosity,
            )
        else:
            action = await self.navigator.get_next_action(
                obs,
                plan,
                memory,
                mcp_client,
                drift,
                session_metrics=session_metrics,
                intrinsic_curiosity=current_curiosity,
            )

        # 4. Resolve Named Skills (e.g. SKILL_DIALOGUE_SKIP)
        if (
            action.button
            and action.button.upper().startswith("SKILL_")
            and self.optimizer
        ):
            skill_name = action.button.upper()

            # --- FEATURE: Dialogue Busy Logic ---
            # If we are in dialogue or a naming screen, prioritize UI macros over exploration
            ocr_upper = (obs.state.ocr_text or "").upper()
            naming_keywords = ["UPPER CASE", "LOWER CASE", "ED IT", "DELETE", " NAME "]
            is_naming = any(k in ocr_upper for k in naming_keywords)
            is_busy = obs.state.has_dialogue_arrow or is_naming

            # If busy, only allow 'SKIP' or 'DIALOGUE' named skills
            if is_busy and not any(
                k in skill_name for k in ["SKIP", "DIALOGUE", "CONFIRM"]
            ):
                print(
                    f"🚫 Blocking exploration macro {skill_name} because Dialogue/UI is busy."
                )
                # Fallback to a simple A press or the model's base action
                return Action(
                    button="a",
                    duration=5,
                    reasoning="Dialogue Busy: Falling back to A press instead of exploration macro.",
                )

            seq = self.optimizer.resolve_skill(skill_name)
            if seq:
                print(f"🧩 Resolving named skill: {skill_name}")
                self.optimizer.record_macro_usage_by_name(skill_name)
                macro_actions = [Action(**step) for step in seq]
                # Update the action to be a macro
                action = Action(
                    macro=macro_actions,
                    reasoning=f"Executing named skill: {skill_name} | {action.reasoning}",
                )

        return action
