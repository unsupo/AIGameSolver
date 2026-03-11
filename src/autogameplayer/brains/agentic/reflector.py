import json
import numpy as np
import time
from pathlib import Path
from typing import Any
from autogameplayer.utils.llm import LLMClientProtocol
from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.utils.database import database_write
from autogameplayer.core.models import Observation

class ReflectionAgent:
    """Agent that analyzes session history to distill and evolve macros."""
    def __init__(self, client: LLMClientProtocol, model: str, optimizer: StrategyOptimizer, knowledge=None):
        self.client = client
        self.model = model
        self.optimizer = optimizer
        self.knowledge = knowledge

    async def ingest_json_episode(self, episode_path: str, ltm: Any):
        """Processes an external JSON episode file into the replay buffer and reflects on it."""
        try:
            with open(episode_path, "r") as f:
                episode_data = json.load(f)
            
            session_id = f"imported_{Path(episode_path).stem}"
            print(f"📥 Ingesting episode {session_id} into learning pipeline...")
            
            # 1. Manually add steps to LTM replay buffer if they aren't there
            # (In a real system we'd check if session already exists)
            from autogameplayer.brains.agentic.memory import StepRecord
            for i, step in enumerate(episode_data):
                record = StepRecord(
                    map_id=step['context'].get('map_id', 0),
                    coords=(step['context'].get('x', 0), step['context'].get('y', 0)),
                    button=step['action'].get('button', 'none'),
                    ocr=step.get('ocr', ''),
                    vision_hash=step.get('vision_hash', ''),
                    reward=step.get('reward', 0.0),
                    vision_vector=step.get('vision_vector')
                )
                await ltm.add_step_to_replay(session_id, i, record)
            
            # 2. Run standard reflection
            await self.analyze_session(session_id, ltm)
            
        except Exception as e:
            print(f"⚠️ Episode ingestion failed: {e}")

    def calculate_visual_delta(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates Euclidean distance between two vision vectors."""
        if v1 is None or v2 is None:
            return 0.0
        return float(np.linalg.norm(v1 - v2))

    async def analyze_session(self, session_id: str, ltm: Any, limit: int = 100):
        """Reviews recent history to identify breakthroughs and traps."""
        steps = ltm.get_steps_by_session(session_id, limit=limit)
        if len(steps) < 10:
            return
        
        # Sort by step index ascending for chronological analysis
        steps.sort(key=lambda x: x['step_index'])
        
        # Calculate adaptive breakthrough threshold
        # Base is 2.0, but we scale it if the average reward is high
        avg_reward = sum(s['reward'] for s in steps) / len(steps)
        breakthrough_threshold = max(2.0, avg_reward * 3.0)
        
        is_deep_scan = limit > 100
        scan_type = "DEEP SCAN" if is_deep_scan else "standard"
        print(f"🕵️ Reflection: Running {scan_type} analysis on last {len(steps)} steps (Breakthrough Threshold: {breakthrough_threshold:.1f})")

        # Format steps for LLM analysis
        formatted_history = []
        # If deep scan, we subsample to avoid context bloat
        stride = max(1, len(steps) // 50) if is_deep_scan else 1
        for i in range(0, len(steps), stride):
            s = steps[i]
            formatted_history.append({
                "index": s['step_index'],
                "map": s['map_id'],
                "pos": s['coords'],
                "action": s['button'],
                "reward": s['reward'],
                "stuck": s['stuck']
            })

        prompt = f"""
        Analyze these actions from a {scan_type} of a play session. 
        GOAL: Summarize, identify breakthroughs (Map Changes or Reward > {breakthrough_threshold:.1f}), and identify traps (Zero progress).
        
        HISTORY ({len(formatted_history)} samples):
        {json.dumps(formatted_history, indent=2)}
        
        TASK: 
        1. Summarize exactly what happened in this segment in one sentence.
        2. Identify "Breakthroughs" (Gold Runs): sequences of 2-10 buttons that led to a Map Change, Reward > {breakthrough_threshold:.1f}, or a Visual Landmark (Significant change in screen content/UI).
        3. Prioritize sequences with the highest REWARD-TO-STEP ratio (Maximum efficiency).
        4. Identify any "Traps" (sequences that stayed in the same coordinate for > 10 steps with 0 reward).
        5. Output Breakthroughs as compressed JSON macro sequences.
        
        CRITICAL: Respond ONLY with a valid JSON object. Do not include markdown formatting, preamble, or any conversational text.
        
        Output format:
        {{
            "summary": "...",
            "breakthroughs": [
                {{ "reason": "...", "start_index": int, "end_index": int, "reward_efficiency": float, "sequence": [{{ "button": "...", "frames": int }}] }}
            ],
            "trap_indices": [int, ...]
        }}
        """
        
        try:
            response = await self.client.acreate_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.2
            )
            from autogameplayer.utils.llm import extract_json_from_llm_response
            analysis = extract_json_from_llm_response(response)
            
            if not analysis:
                print(f"⚠️ Reflection analysis failed: Could not extract JSON from response: {response[:100]}...")
                return

            summary = analysis.get("summary", "Exploration continued.")
            print(f"📝 Reflection Summary: {summary}")
            
            # Record summary to LTM as a high-level milestone
            if ltm:
                await ltm.add_memory(f"Session Reflection: {summary}", {"type": "reflection", "session_id": session_id})

            # --- FEATURE: Permanent Knowledge Distillation ---
            # Save the summary into the Knowledge Base (RAG) so it persists across all future sessions
            if self.knowledge:
                knowledge_note = f"SESSION INSIGHT ({session_id}): {summary}"
                await self.knowledge.ingest_text(knowledge_note, source=f"reflection_{session_id}")
                print(f"📒 Permanent Knowledge Recorded: {summary[:50]}...")
            # --------------------------------------------------

            # 1. Save new breakthrough macros
            for b in analysis.get("breakthroughs", []):
                start_idx = b.get("start_index")
                if start_idx is None:
                    continue
                
                # Get vision vector from the DB at the start step
                # We can find the step in our current 'steps' list
                start_step = next((s for s in steps if s['step_index'] == start_idx), None)
                if not start_step:
                    continue
                
                vision_vec = np.frombuffer(start_step['vision_vector'], dtype=np.float32).tolist() if start_step['vision_vector'] else None
                if not vision_vec:
                    continue
                
                # Find steps in this range to calculate reward and metadata
                range_steps = [s for s in steps if start_idx <= s['step_index'] <= b.get('end_index', start_idx + 10)]
                if not range_steps:
                    continue
                
                reward_sum = sum(s['reward'] for s in range_steps)
                efficiency = reward_sum / max(1, len(range_steps))
                
                map_id = start_step['map_id']
                end_map_id = range_steps[-1]['map_id']
                coords = json.loads(start_step['coords']) if isinstance(start_step['coords'], str) else start_step['coords']
                vision_hash = start_step['vision_hash']
                ocr_text = start_step['ocr']
                
                # Check for intro breakthrough
                is_intro_breakthrough = (map_id == 0 and end_map_id != 0)
                skill_name = "SKILL_SKIP_INTRO" if is_intro_breakthrough else None
                
                print(f"✨ Gold Run Discovery: {b['reason']} (Efficiency: {efficiency:.2f}, Total Reward: {reward_sum:.1f}){f' -> {skill_name}' if skill_name else ''}")
                
                self.optimizer.save_macro(
                    vision_vector=vision_vec,
                    sequence=b['sequence'],
                    map_id=map_id,
                    coords=coords,
                    description=b['reason'],
                    score=reward_sum * efficiency, # Boost score by efficiency
                    vision_hash=vision_hash,
                    ocr_text=ocr_text
                )
                
                # If it's a critical breakthrough, give it a permanent name immediately
                if skill_name:
                    self._update_skill_name(vision_hash, skill_name)

                # --- FEATURE: Visual Landmark Recording ---
                # If the breakthrough caused a massive visual change but NO map change,
                # record it as a landmark in RAG so the Planner knows what happened.
                end_step = range_steps[-1]
                v_end = np.frombuffer(end_step['vision_vector'], dtype=np.float32) if end_step['vision_vector'] else None
                v_start = np.frombuffer(start_step['vision_vector'], dtype=np.float32) if start_step['vision_vector'] else None
                
                v_delta = self.calculate_visual_delta(v_start, v_end)
                if v_delta > 1.5 and map_id == end_map_id: # Threshold for 'significant change'
                    landmark_desc = f"Visual Landmark on Map {map_id}: {b['reason']} at {coords}. This sequence significantly changed the screen state."
                    if ltm:
                        await ltm.add_memory(landmark_desc, {
                            "type": "landmark", 
                            "map_id": map_id, 
                            "x": coords[0], "y": coords[1],
                            "vision_vector": v_end.tolist() if v_end is not None else None
                        })
                        print(f"📍 Recorded Visual Landmark: {b['reason']}")
                    
                    # ALSO record to permanent knowledge
                    if self.knowledge:
                        await self.knowledge.ingest_text(landmark_desc, source="visual_landmarks")
                # ------------------------------------------
                
            # 2. Identify and Penalize Traps
            for trap_idx in analysis.get("trap_indices", []):
                trap_vision = ltm.get_vision_vector(session_id, trap_idx)
                if trap_vision:
                    print(f"🗑️ Reflection identified TRAP at step {trap_idx}. Penalizing existing macro...")
                    self.optimizer.penalize_macro(trap_vision)
            
        except Exception as e:
            print(f"⚠️ Reflection analysis failed: {e}")

    @database_write("optimizer.db_path")
    def _update_skill_name(self, vision_hash: str, skill_name: str):
        """Internal helper to promote a macro to a permanent named skill."""
        try:
            import sqlite3
            with sqlite3.connect(str(self.optimizer.db_path), timeout=10) as conn:
                conn.execute(
                    "UPDATE skills SET name = ?, compressed = 1, reliability = 1.0 WHERE vision_hash = ?",
                    (skill_name, vision_hash)
                )
                conn.commit()
        except Exception:
            pass

    async def analyze_failure(self, session_id: str, ltm: Any, map_id: int, pos: tuple, observation: Observation):
        """Instant reflection for real-time debugging of stuck states."""
        steps = ltm.get_steps_by_session(session_id, limit=10)
        if not steps:
            return
        
        # Chronological order
        steps.sort(key=lambda x: x['step_index'])
        
        history_text = "\n".join([
            f"- Step {s['step_index']}: Pressed {s['button'].upper()} at {s['coords']}. Result: {'STUCK' if s['stuck'] else 'MOVED'}"
            for s in steps
        ])

        prompt = f"""
        DEBUG MODE: The AI is STUCK at {pos} on Map #{map_id}. 
        The last 10 actions resulted in zero progress.
        
        RECENT HISTORY:
        {history_text}
        
        TASK:
        Analyze the screenshot and the history. Propose a new SKILL (sequence of 2-5 buttons) specifically designed to bypass this obstacle or break the loop.
        
        CRITICAL: Respond ONLY with a valid JSON object.
        
        Output format:
        {{
            "analysis": "Brief explanation of why we are stuck",
            "proposed_skill_name": "SKILL_BYPASS_...",
            "sequence": [{{ "button": "...", "frames": int }}]
        }}
        """
        
        try:
            from autogameplayer.utils.llm import extract_json_from_llm_response
            
            # Use image if available for visual debugging
            content = [{"type": "text", "text": prompt}]
            if observation.state.image_data:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{observation.state.image_data}"}})

            response = await self.client.acreate_completion(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=400,
                temperature=0.1
            )
            
            debug_info = extract_json_from_llm_response(response)
            if debug_info and "sequence" in debug_info:
                name = debug_info.get("proposed_skill_name", f"SKILL_DEBUG_{int(time.time())}")
                print(f"🛠️ Debug Reflection: {debug_info['analysis']} -> Creating {name}")
                
                self.optimizer.save_macro(
                    vision_vector=observation.state.vision_vector,
                    sequence=debug_info["sequence"],
                    map_id=map_id,
                    coords=pos,
                    description=f"Debug Skill: {debug_info['analysis']}",
                    score=15.0, # High priority
                    vision_hash=observation.state_hash,
                    ocr_text=observation.state.ocr_text,
                    reliability=0.8 # Moderate initial trust
                )
                
                # Update name
                self._update_skill_name(observation.state_hash, name)
                
        except Exception as e:
            print(f"⚠️ Debug reflection failed: {e}")
