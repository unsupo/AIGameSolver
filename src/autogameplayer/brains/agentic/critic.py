import json
import asyncio
from autogameplayer.core.models import Observation
from .memory import EpisodicMemory

class CriticAgent:
    """Agent responsible for evaluating performance and detecting stagnation (stuck states)."""
    def __init__(self, ltm=None, session_id: str = "default"):
        self.milestones = set()
        self.last_states = {} # Tracks previous values for transition detection
        self.ltm = ltm
        self.session_id = session_id
        self._load_global_milestones()
        
    def _load_global_milestones(self):
        """Hydrate milestones from SQLite so the AI knows what it already achieved across runs."""
        if not self.ltm: return
        try:
            import sqlite3
            with sqlite3.connect(str(self.ltm.storage_path)) as conn:
                # Look for memories that represent critical achievements
                cursor = conn.execute("SELECT metadata FROM memories WHERE type = 'milestone' OR text LIKE 'CRITICAL OBJECTIVE ACHIEVED%'")
                for row in cursor:
                    try:
                        meta = json.loads(row[0])
                        if 'milestone_key' in meta:
                            self.milestones.add(meta['milestone_key'])
                    except (json.JSONDecodeError, KeyError):
                        continue
            if self.milestones:
                print(f"🧠 Critic: Loaded {len(self.milestones)} global milestones from LTM.")
        except Exception as e:
            print(f"⚠️ Failed to load global milestones: {e}")

    def _register_milestone(self, key: str, reward: float, guidance: str):
        """Helper to record permanent milestones to LTM."""
        if key in self.milestones:
            return 0.0, ""
            
        self.milestones.add(key)
        if self.ltm:
            asyncio.create_task(self.ltm.add_memory(
                guidance, 
                {"type": "milestone", "milestone_key": key}
            ))
        return reward, guidance

    def evaluate(self, memory: EpisodicMemory, current_obs: Observation) -> tuple[float, bool, str, bool]:
        """Returns (reward_delta, is_stuck, guidance, is_loop)."""
        # BULLETPROOF COOLDOWN: Total immunity during escape maneuvers
        if memory.steps:
            last_reason = list(memory.steps)[-1].reasoning or ""
            if any(r in last_reason for r in ["Breaking stuck state", "Mash Breakout", "Interface Breakout"]):
                return 0.0, False, "Cooldown: Waiting for escape maneuver to finish.", False

        ctx = current_obs.state.context
        last_reward = ctx.get('last_reward', 0.0)
        map_id = ctx.get('map_id', -1)
        ocr = (current_obs.state.ocr_text or "").upper()
        curr_coords = (ctx.get('x', 0), ctx.get('y', 0))
        
        is_dialogue = ctx.get('is_dialogue', False) or current_obs.state.has_dialogue_box or map_id == -1
        is_overworld = ctx.get("is_overworld_active", False)
        
        reward_delta = 0.0
        is_stuck = False
        is_loop = False
        guidance = ""

        # --- 6. SELF-GENERATED CURRICULUM (Discovery Rewards) ---
        from autogameplayer.core.models import GameStage
        
        # A. Stage Discovery (Title -> Menu -> Overworld -> Battle)
        stage = current_obs.state.stage
        if stage != self.last_states.get('stage', GameStage.UNKNOWN):
            s_key = f"stage_{stage.value}"
            if s_key not in self.milestones:
                r, g = self._register_milestone(s_key, 2.0, f"STAGE ACHIEVED: You entered the {stage.value} phase of the game!")
                reward_delta += r
                guidance = g
        self.last_states['stage'] = stage

        # B. Map Discovery (Entering Building/New Area)
        if map_id != self.last_states.get('map_id', -1) and self.last_states.get('map_id', -1) != -1:
            m_key = f"reach_map_{map_id}"
            if m_key not in self.milestones:
                r, g = self._register_milestone(m_key, 5.0, f"OBJECTIVE ACHIEVED: You discovered a new area (Map #{map_id})!")
                reward_delta += r
                guidance = g
        self.last_states['map_id'] = map_id

        # B. Battle Discovery (Starting Combat)
        battle_state = ctx.get('battle_state', 0)
        if battle_state > 0 and self.last_states.get('battle_state', 0) == 0:
            if "start_battle" not in self.milestones:
                r, g = self._register_milestone("start_battle", 3.0, "OBJECTIVE ACHIEVED: You successfully engaged in a battle!")
                reward_delta += r
                guidance = g
        self.last_states['battle_state'] = battle_state

        # C. Menu Discovery (Learning to use the UI)
        is_menu = ctx.get('is_menu_open', False)
        if is_menu and not self.last_states.get('is_menu_open', False):
            if "open_menu" not in self.milestones:
                r, g = self._register_milestone("open_menu", 1.0, "OBJECTIVE ACHIEVED: You successfully opened the menu!")
                reward_delta += r
                guidance = g
        self.last_states['is_menu_open'] = is_menu

        # D. Wealth Discovery (Gain money)
        money = ctx.get('money', 0)
        if money > self.last_states.get('money', 0) and self.last_states.get('money', 0) > 0:
            reward_delta += 2.0
            guidance = f"OBJECTIVE ACHIEVED: You earned wealth ({money} Yen)!"
        self.last_states['money'] = money

        # E. Starter Pokemon Discovery
        party_count = ctx.get('party_count', 0)
        if party_count > 0 and self.last_states.get('party_count', 0) == 0:
            if "get_starter" not in self.milestones:
                r, g = self._register_milestone("get_starter", 10.0, "CRITICAL OBJECTIVE ACHIEVED: You obtained your first Pokemon! The real game begins.")
                reward_delta += r
                guidance = g
        self.last_states['party_count'] = party_count

        # F. Badge Discovery
        badges = ctx.get('badges_count', 0)
        if badges > self.last_states.get('badges_count', 0):
            b_key = f"badge_{badges}"
            r, g = self._register_milestone(b_key, 20.0, f"CRITICAL OBJECTIVE ACHIEVED: You earned a Gym Badge (Total: {badges})!")
            reward_delta += r
            guidance = g
        self.last_states['badges_count'] = badges

        # 0. MAP-SPECIFIC STAGNATION PENALTY (Session-based)
        if not hasattr(self, 'map_step_counts'): self.map_step_counts = {}
        self.map_step_counts[map_id] = self.map_step_counts.get(map_id, 0) + 1
        
        if self.map_step_counts.get(map_id, 0) >= 100:
            map_stagnation = (self.map_step_counts[map_id] // 100) * 0.1
            reward_delta -= map_stagnation
            if self.map_step_counts[map_id] % 100 == 0:
                guidance = f"MAP STAGNATION: You have spent {self.map_step_counts[map_id]} steps on Map #{map_id}. Move to a new area."

        if last_reward < -0.5:
            is_stuck = True

        if len(memory.steps) < 5: 
            return reward_delta, is_stuck, guidance, is_loop
        
        recent = list(memory.steps)[-8:]
        
        # --- 7. PROGRESS DETECTION ---
        last_step = memory.steps[-1] if memory.steps else None
        has_new_text = (current_obs.state.ocr_text != last_step.ocr) if last_step else False
        has_new_hash = (current_obs.state_hash != last_step.vision_hash) if last_step else False
        
        # If progress is zero for multiple steps, we are stuck
        if len(memory.steps) >= 5:
            recent_progress = []
            for i in range(1, 6):
                s = memory.steps[-i]
                p = s.vision_delta + (1.0 if i < len(memory.steps) and s.ocr != memory.steps[-i-1].ocr else 0.0)
                recent_progress.append(p)
            
            if sum(recent_progress) < 0.05: # Threshold for "Zero Progress"
                is_stuck = True
                guidance = "STAGNATION: Zero progress detected (Vision, OCR, and Hash are static). Switch strategy."

        # 1. Static Screen Detection (Frustration)
        if current_obs.state.vision_delta < 0.01:
            intro_maps = ctx.get("intro_map_ids", [0])
            is_intro_map = (map_id in intro_maps)
            
            if (is_dialogue or is_intro_map) and any(s.button in ['a', 'b', 'start'] for s in recent[-3:]):
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
                recent_fingerprints.append(f"{step.coords[0]}_{step.coords[1]}_{step_ocr}")
                
            loop_count = recent_fingerprints.count(state_fingerprint)
            
            if loop_count > 3:
                reward_delta -= 2.0
                is_stuck = True
                is_loop = True
                guidance = "STATE LOOP DETECTED: You keep returning to the exact same state."

        # 3. Cutscene Awareness
        if ctx.get("is_cutscene_locked"):
            is_stuck = False
            reward_delta = 0.0 
            guidance = "Game is executing a scripted event. Wait for control to be returned."

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
            
            if curr_ocr == last_ocr and not has_arrow and last_button in ['a', 'b', 'start']:
                reward_delta -= 0.5
                guidance = "DIALOGUE MASHING: You are pressing buttons while text is still printing. Wait for the arrow."
            
            if curr_ocr != last_ocr and last_button in ['a', 'b', 'start']:
                reward_delta += 1.0

        # 7. SPATIAL STAGNATION (Long-term)
        if len(memory.steps) >= 20:
            long_recent = list(memory.steps)[-20:]
            spatial_static = all(s.coords == curr_coords and s.map_id == map_id for s in long_recent)
            
            if spatial_static and not is_dialogue and not ctx.get("is_cutscene_locked"):
                is_stuck = True
                guidance = "LONG-TERM SPATIAL STAGNATION: Position and Map have not changed for 20 steps. Strategy is failing to move the player."

        # Short-term coordinate stagnation
        if not is_stuck and curr_coords != (0, 0) and not is_dialogue:
            static = all(s.coords == curr_coords for s in recent[-5:])
            directional = all(s.button in ["up", "down", "left", "right"] for s in recent[-5:])
            if directional and static:
                is_stuck = True
                guidance = "MOVEMENT BLOCKED: You are walking but your coordinates are not changing. You are hitting an obstacle."
        
        # --- FEATURE: Dead End Persistence ---
        if is_stuck and self.ltm:
            # Record this visual state as a 'Dead End' in persistent memory
            asyncio.create_task(self.ltm.record_dead_end(current_obs.state_hash, self.session_id))
        # ------------------------------------
            
        return reward_delta, is_stuck, guidance, is_loop
