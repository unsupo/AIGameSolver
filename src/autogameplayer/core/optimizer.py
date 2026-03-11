import json
import sqlite3
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from autogameplayer.core.config import settings

from autogameplayer.utils.llm import LLMClientProtocol

from autogameplayer.utils.vector import cosine_similarity
from autogameplayer.utils.timing import frames_to_seconds
from autogameplayer.utils.database import ensure_ltm_schema, get_db_connection, database_write

class StrategyOptimizer:
    """Consolidates individual experiences into reusable 'Skills' stored in SQLite."""
    def __init__(self, client: LLMClientProtocol = None, model: str = None, db_path: str = None):
        self.client = client
        self.model = model
        if db_path is None:
            self.db_path = settings.models_dir / "long_term_memory.db"
        else:
            self.db_path = Path(db_path)
            
        self._init_db()

    def _init_db(self):
        print(f"🗄️ StrategyOptimizer initializing DB at: {self.db_path}")
        ensure_ltm_schema(self.db_path)
        self._seed_global_skills()

    @database_write("db_path")
    def _seed_global_skills(self):
        """Seeds the database with game-agnostic default macros from data/skills."""
        import yaml
        
        # 1. Default Hardcoded Seeds
        seeds = [
            {
                "name": "SKILL_CONFIRM",
                "description": "Standard confirmation: Pressing A to select/confirm.",
                "macro": [{"button": "a", "frames": 10}]
            },
            {
                "name": "SKILL_SKIP_INTRO",
                "description": "Pokemon Red skip: Title, New Game, Naming, and exit bedroom.",
                "macro": [
                    {"button": "start", "frames": 20},
                    {"button": "a", "frames": 30},
                    {"button": "a", "frames": 30},
                    {"button": "a", "frames": 30},
                    {"button": "a", "frames": 30},
                    {"button": "a", "frames": 30},
                    {"button": "down", "frames": 10},
                    {"button": "down", "frames": 10},
                    {"button": "down", "frames": 10}
                ]
            }
        ]
        
        # 2. Load from data/skills directory
        skills_dir = settings.skills_dir
        if skills_dir.exists():
            for f_path in skills_dir.glob("*.yaml"):
                try:
                    with open(f_path, "r") as f:
                        data = yaml.safe_load(f)
                        if data and "name" in data and "sequence" in data:
                            seeds.append({
                                "name": data["name"],
                                "description": data.get("description", f"Loaded from {f_path.name}"),
                                "macro": data["sequence"]
                            })
                except Exception as e:
                    print(f"⚠️ Failed to load skill file {f_path.name}: {e}")

        try:
            with get_db_connection(self.db_path) as conn:
                for s in seeds:
                    # Check if already exists by name
                    cursor = conn.execute("SELECT id FROM skills WHERE name = ?", (s['name'],))
                    if not cursor.fetchone():
                        print(f"🌱 Seeding agnostic skill: {s['name']}")
                        conn.execute(
                            "INSERT INTO skills (map_id, name, description, macro_json, score, reliability) VALUES (?, ?, ?, ?, ?, ?)",
                            (-1, s['name'], s['description'], json.dumps(s['macro']), 10.0, 0.5)
                        )
                conn.commit()
        except Exception as e:
            print(f"⚠️ Failed to seed global skills: {e}")

    @database_write("db_path")
    def save_macro(self, vision_vector: list, sequence: list, map_id: int, coords: tuple, description: str = "Discovered successful sequence.", score: float = 5.0, vision_hash: str = None, ocr_text: str = None, is_hierarchical: bool = False, reliability: float = 0.5):
        """Writes distilled skills directly to the DB instead of JSON."""
        v_bytes = np.array(vision_vector, dtype=np.float32).tobytes() if vision_vector else None
        
        # Check for similar visual state to prevent duplicates (Similarity threshold 0.98)
        best_match_id = None
        if vision_vector:
            q_vec = np.array(vision_vector)
            try:
                with get_db_connection(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("SELECT id, vision_vector, score FROM skills WHERE map_id = ?", (map_id,))
                    for row in cursor:
                        sid, v_blob, old_score = row['id'], row['vision_vector'], row['score']
                        if v_blob:
                            m_vec = np.frombuffer(v_blob, dtype=np.float32)
                            sim = cosine_similarity(q_vec, m_vec)
                            if sim > 0.98:
                                best_match_id = sid
                                break
            except Exception: pass

        try:
            with get_db_connection(self.db_path) as conn:
                if best_match_id:
                    # Update existing macro if new score is better
                    conn.execute(
                        "UPDATE skills SET macro_json = ?, description = ?, score = MAX(score, ?), vision_hash = ?, ocr_text = ?, is_hierarchical = ? WHERE id = ?",
                        (json.dumps(sequence), description, score, vision_hash, ocr_text, 1 if is_hierarchical else 0, best_match_id)
                    )
                else:
                    # Insert new macro
                    conn.execute(
                        "INSERT INTO skills (map_id, vision_vector, vision_hash, ocr_text, macro_json, description, score, is_hierarchical, reliability) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (map_id, v_bytes, vision_hash, ocr_text, json.dumps(sequence), description, score, 1 if is_hierarchical else 0, reliability)
                    )
                conn.commit()
        except Exception as e:
            print(f"⚠️ Failed to save macro to DB: {e}")

    @database_write("db_path")
    def record_macro_usage_by_name(self, name: str):
        """Increments the run count for a skill by its name or description."""
        try:
            with get_db_connection(self.db_path) as conn:
                conn.execute(
                    "UPDATE skills SET times_run = times_run + 1 WHERE name = ? OR description = ?",
                    (name, name)
                )
                conn.commit()
        except Exception: pass

    @database_write("db_path")
    def promote_macro(self, vision_vector: List[float], outcome: float):
        """Increase reliability score on success, decrease on failure."""
        if not vision_vector: return
        q_vec = np.array(vision_vector)
        
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT id, map_id, vision_vector, reliability, times_run, times_succeeded, name FROM skills")
                for row in cursor:
                    sid, v_blob, reliability, times_run, times_succeeded = row['id'], row['vision_vector'], row['reliability'], row['times_run'], row['times_succeeded']
                    
                    is_match = False
                    if v_blob:
                        m_vec = np.frombuffer(v_blob, dtype=np.float32)
                        sim = cosine_similarity(q_vec, m_vec)
                        if sim > 0.95:
                            is_match = True
                    elif row['map_id'] == -1:
                        # For global agnostic skills without vision vectors, 
                        # we rely on the Brain to report success/failure of the skill it just ran by name.
                        pass

                    if is_match:
                        # Outcome should be 1.0 for success, 0.0 for failure
                        new_reliability = reliability * 0.9 + outcome * 0.1
                        new_succeeded = times_succeeded + (1 if outcome > 0.5 else 0)
                        
                        if new_reliability < 0.2 and times_run > 3:
                            # IMPORTANT: Don't delete global seeds (map_id -1), just let them have low scores
                            if row['map_id'] != -1:
                                print(f"🗑️ Purging unreliable macro {sid} (Reliability: {new_reliability:.2f})")
                                conn.execute("DELETE FROM skills WHERE id = ?", (sid,))
                            else:
                                conn.execute(
                                    "UPDATE skills SET reliability = ?, times_run = ?, times_succeeded = ? WHERE id = ?",
                                    (new_reliability, times_run + 1, new_succeeded, sid)
                                )
                        else:
                            conn.execute(
                                "UPDATE skills SET reliability = ?, times_run = ?, times_succeeded = ? WHERE id = ?",
                                (new_reliability, times_run + 1, new_succeeded, sid)
                            )
                        conn.commit()
                        print(f"📈 Promoted Macro {sid} ({row['name'] or 'Unnamed'}): Reliability {new_reliability:.2f} (Run {times_run+1})")
                        break
        except Exception as e:
            print(f"⚠️ Failed to promote macro: {e}")

    def _resolve_hierarchical_macro(self, macro_json: str) -> List[Dict[str, Any]]:
        """Recursively flattens a hierarchical skill into a base action sequence."""
        sub_skill_names = json.loads(macro_json)
        flat_sequence = []
        
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                for name in sub_skill_names:
                    cursor = conn.execute("SELECT macro_json, is_hierarchical FROM skills WHERE name = ? OR description = ? LIMIT 1", (name, name))
                    row = cursor.fetchone()
                    if row:
                        if row['is_hierarchical']:
                            flat_sequence.extend(self._resolve_hierarchical_macro(row['macro_json']))
                        else:
                            flat_sequence.extend(json.loads(row['macro_json']))
        except Exception as e:
            print(f"⚠️ Failed to resolve hierarchical macro: {e}")
            
        return flat_sequence

    def get_relevant_macros(self, vision_vector: List[float], threshold: float = 0.95, vision_hash: str = None, map_id: int = -1) -> List[Dict[str, Any]]:
        """Returns all macros matching the current state ranked by similarity * reliability."""
        q_vec = np.array(vision_vector) if vision_vector else None
        q_norm = np.linalg.norm(q_vec) if q_vec is not None else 0
        is_vision_active = q_norm > 0.001
        
        matches = []
        
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # 1. Filter by map context first for efficiency
                cursor = conn.execute(
                    "SELECT vision_vector, vision_hash, ocr_text, macro_json, description, score, reliability, times_run, times_succeeded, is_hierarchical, map_id FROM skills WHERE map_id = ? OR map_id = -1",
                    (map_id,)
                )
                
                for row in cursor:
                    sim = 0.0
                    is_match = False
                    
                    # MATCHING LOGIC
                    v_blob = row['vision_vector']
                    
                    # Case A: Visual Match (Primary)
                    if is_vision_active and v_blob:
                        m_vec = np.frombuffer(v_blob, dtype=np.float32)
                        sim = cosine_similarity(q_vec, m_vec)
                        if sim > threshold:
                            is_match = True
                    
                    # Case B: Hash Match (Fallback for exact same screen or disabled vision)
                    if not is_match and vision_hash and row['vision_hash'] == vision_hash:
                        sim = 1.0 # Perfect match for same hash
                        is_match = True
                        
                    # Case C: Global Skill Match (Always relevant if name exists)
                    if not is_match and row['map_id'] == -1 and not row['vision_vector']:
                        pass

                    if is_match:
                        # Resolve hierarchical sequence if needed
                        if row['is_hierarchical']:
                            sequence = self._resolve_hierarchical_macro(row['macro_json'])
                        else:
                            sequence = json.loads(row['macro_json'])
                            
                        matches.append({
                            "sequence": sequence,
                            "description": row['description'],
                            "vision_hash": row['vision_hash'],
                            "ocr_text": row['ocr_text'],
                            "score": row['score'],
                            "reliability": row['reliability'],
                            "times_run": row['times_run'],
                            "times_succeeded": row['times_succeeded'],
                            "sim": float(sim),
                            "rank_score": float(sim) * row['reliability']
                        })
        except Exception as e:
            print(f"⚠️ Failed to query macros: {e}")
        
        # Sort by rank_score (sim * reliability) then absolute score
        matches.sort(key=lambda x: (x["rank_score"], x["score"]), reverse=True)
        return matches

    def get_best_macro_for_context(self, vision_vector: List[float], map_id: int, vision_hash: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Returns top matching macros for the current context."""
        return self.get_relevant_macros(vision_vector, vision_hash=vision_hash, map_id=map_id)[:top_k]

    def get_best_macro(self, vision_vector: List[float], threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        matches = self.get_relevant_macros(vision_vector, threshold)
        return matches[0] if matches else None

    @database_write("db_path")
    def penalize_macro(self, vision_vector: List[float]):
        """Reduces the reliability of a macro in the DB using multiplicative decay."""
        if not vision_vector: return
        q_vec = np.array(vision_vector)
        
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT id, vision_vector, reliability, times_run FROM skills")
                for row in cursor:
                    sid, v_blob, reliability, times_run = row['id'], row['vision_vector'], row['reliability'], row['times_run']
                    if v_blob:
                        m_vec = np.frombuffer(v_blob, dtype=np.float32)
                        sim = cosine_similarity(q_vec, m_vec)
                        if sim > 0.95:
                            # MULTIPLICATIVE DECAY
                            new_reliability = reliability * 0.9
                            
                            if new_reliability < 0.2 and times_run > 3:
                                print(f"🗑️ Purging failed macro {sid} (Reliability dropped too low).")
                                conn.execute("DELETE FROM skills WHERE id = ?", (sid,))
                            else:
                                conn.execute("UPDATE skills SET reliability = ? WHERE id = ?", (new_reliability, sid))
                                conn.commit()
                                break
        except Exception as e:
            print(f"⚠️ Failed to penalize macro: {e}")

    @database_write("db_path")
    def penalize_macro_by_name(self, name: str):
        """Targeted demotion of a skill by its name/ID."""
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT id, reliability FROM skills WHERE name = ?", (name,))
                row = cursor.fetchone()
                if row:
                    new_reliability = row['reliability'] * 0.8 # Slightly harsher for explicit name penalty
                    conn.execute("UPDATE skills SET reliability = ? WHERE id = ?", (new_reliability, row['id']))
                    conn.commit()
                    print(f"📉 Penalized Skill by Name: {name} -> New Reliability: {new_reliability:.2f}")
        except Exception: pass

    @database_write("db_path")
    def evolve_macro(self, vision_vector: List[float], new_sequence: List[Dict[str, Any]]):
        """Replaces a macro's sequence with a more efficient one if similar visual match found."""
        if not vision_vector: return
        q_vec = np.array(vision_vector)
        
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT id, vision_vector, macro_json FROM skills")
                for row in cursor:
                    sid, v_blob, old_macro_json = row['id'], row['vision_vector'], row['macro_json']
                    if v_blob:
                        m_vec = np.frombuffer(v_blob, dtype=np.float32)
                        sim = cosine_similarity(q_vec, m_vec)
                        if sim > 0.98:
                            try:
                                old_seq = json.loads(old_macro_json)
                                # If the new sequence is shorter (more efficient)
                                if len(new_sequence) < len(old_seq):
                                    print(f"📈 Evolving macro {sid}: Compressed from {len(old_seq)} to {len(new_sequence)} steps.")
                                    conn.execute("UPDATE skills SET macro_json = ? WHERE id = ?", (json.dumps(new_sequence), sid))
                                    conn.commit()
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break
        except Exception: pass

    def _is_high_entropy(self, macro_json: str) -> bool:
        """Returns True if the macro has more than one unique button (strategic)."""
        try:
            seq = json.loads(macro_json)
            if not seq: return False
            unique_buttons = set(step.get('button', 'none') for step in seq if step.get('button'))
            return len(unique_buttons) > 1
        except Exception:
            return False

    @database_write("db_path")
    async def async_compress_skills(self):
        """Periodically reviews new macros and gives them high-level semantic names."""
        if not self.client or not self.model: return

        try:
            uncompressed = []
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT id, map_id, macro_json, description FROM skills WHERE compressed = 0 LIMIT 5")
                uncompressed = [dict(row) for row in cursor.fetchall()]
            
            if not uncompressed: return

            print(f"🧠 AI is 'Dreaming': Compressing {len(uncompressed)} new skills...")
            for skill in uncompressed:
                # --- FEATURE: Entropy Filtering ---
                is_strategic = self._is_high_entropy(skill['macro_json'])
                
                if not is_strategic:
                    # Categorize simple button mashing as utility
                    try:
                        seq = json.loads(skill['macro_json'])
                        btn = seq[0].get('button', 'UNKNOWN').upper()
                        new_name = f"UTILITY_MASH_{btn}"
                    except Exception:
                        new_name = "UTILITY_MACRO"
                    
                    print(f"🛠️ Categorized as Utility: {skill['description']} -> {new_name}")
                else:
                    # Use LLM only for high-entropy strategic skills
                    prompt = f"""
                    You are a Skill Distiller. You take low-level movement macros and turn them into high-level SKILLS.
                    
                    DATA:
                    - Map ID: {skill['map_id']}
                    - Macro: {skill['macro_json']}
                    - Original Description: {skill['description']}
                    
                    TASK:
                    Create a high-level, unique name for this skill (e.g., SKILL_EXIT_ROOM, SKILL_WIN_BATTLE).
                    
                    OUTPUT: ONLY the skill name in all caps (starting with SKILL_).
                    """
                    
                    response = await self.client.acreate_completion(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=20
                    )
                    new_name = response.strip().replace(" ", "_").upper()
                    if not new_name.startswith("SKILL_"):
                        new_name = f"SKILL_{new_name}"
                    
                    print(f"✨ Distilled Strategic Skill: {skill['description']} -> {new_name}")
                
                with get_db_connection(self.db_path) as conn:
                    conn.execute(
                        "UPDATE skills SET name = ?, compressed = 1 WHERE id = ?",
                        (new_name, skill['id'])
                    )
                    conn.commit()

        except Exception as e:
            print(f"⚠️ Skill compression failed: {e}")

    @database_write("db_path")
    def consolidate_macros(self, threshold: float = 0.95):
        """Identifies near-duplicate macros and merges them, keeping the highest quality ones."""
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM skills")
                all_skills = [dict(row) for row in cursor.fetchall()]
                
                if len(all_skills) < 2: return
                
                print(f"🧹 Running Macro Consolidation on {len(all_skills)} skills...")
                to_delete = set()
                
                # Sort by map_id to group comparisons
                all_skills.sort(key=lambda x: x['map_id'] if x['map_id'] is not None else -1)

                for i in range(len(all_skills)):
                    if all_skills[i]['id'] in to_delete: continue
                    
                    v_blob_i = all_skills[i].get('vision_vector')
                    if not v_blob_i: continue
                    
                    vec_i = np.frombuffer(v_blob_i, dtype=np.float32)
                    map_i = all_skills[i]['map_id']
                    
                    for j in range(i + 1, len(all_skills)):
                        if all_skills[j]['id'] in to_delete: continue
                        if all_skills[j]['map_id'] != map_i: continue 
                        
                        v_blob_j = all_skills[j].get('vision_vector')
                        if not v_blob_j: continue
                        
                        vec_j = np.frombuffer(v_blob_j, dtype=np.float32)
                        
                        if True:
                            sim = cosine_similarity(vec_i, vec_j)
                            
                            if sim > threshold:
                                # Keep the one with better reliability * score
                                rel_i = all_skills[i].get('reliability', 1.0)
                                rel_j = all_skills[j].get('reliability', 1.0)
                                score_i = rel_i * all_skills[i]['score']
                                score_j = rel_j * all_skills[j]['score']
                                
                                if score_i >= score_j:
                                    to_delete.add(all_skills[j]['id'])
                                else:
                                    to_delete.add(all_skills[i]['id'])
                                    break 
                
                if to_delete:
                    print(f"✅ Consolidating skills: Merging {len(to_delete)} near-duplicates (Similarity > {threshold}).")
                    for sid in to_delete:
                        conn.execute("DELETE FROM skills WHERE id = ?", (sid,))
                    conn.commit()
                else:
                    print("✨ No near-duplicate macros found.")
            
            # Sync to YAML for human oversight
            self.sync_to_yaml()
                    
        except Exception as e:
            print(f"⚠️ Skill consolidation failed: {e}")

    def sync_to_yaml(self):
        """Exports the skills table to a human-readable YAML file."""
        yaml_path = self.db_path.parent / "macros.yaml"
        try:
            import yaml
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT name, description, macro_json, score, reliability FROM skills ORDER BY map_id, score DESC")
                skills = []
                for row in cursor:
                    skills.append({
                        "name": row['name'] or "Unnamed",
                        "description": row['description'],
                        "sequence": json.loads(row['macro_json']),
                        "score": round(row['score'], 2),
                        "reliability": round(row['reliability'], 2)
                    })
                
                with open(yaml_path, "w") as f:
                    yaml.dump({"macros": skills}, f, sort_keys=False)
        except Exception as e:
            print(f"⚠️ Failed to sync macros to YAML: {e}")

    @database_write("db_path")
    def mine_sequences(self, top_k: int = 5):
        """Mines the replay buffer for high-scoring frequent subsequences and promotes them to macros."""
        print("⛏️ Mining replay buffer for frequent subsequences...")
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # 1. Fetch all steps from replay buffer
                cursor = conn.execute("SELECT * FROM replay_buffer ORDER BY session_id, step_index")
                all_steps = [dict(row) for row in cursor.fetchall()]
                
                if not all_steps: return
                
                # 2. Group by session
                episodes = {}
                for s in all_steps:
                    sid = s['session_id']
                    if sid not in episodes: episodes[sid] = []
                    episodes[sid].append(s)
                
                sequences = {} 
                
                # 3. Mine subsequences (size 2-5)
                for sid, steps in episodes.items():
                    for i in range(len(steps)):
                        for k in range(2, 6):
                            if i + k > len(steps): break
                            
                            sub = steps[i : i+k]
                            if any(s['map_id'] != sub[0]['map_id'] for s in sub): continue
                            
                            btns = tuple(s['button'] for s in sub)
                            if "none" in btns: continue
                            
                            reward_sum = sum(s['reward'] for s in sub)
                            v_delta_avg = sum(s['vision_delta'] for s in sub) / k
                            
                            if btns not in sequences: sequences[btns] = []
                            sequences[btns].append({
                                "reward": reward_sum,
                                "v_delta": v_delta_avg,
                                "start_step": sub[0]
                            })
                
                # 4. Score sequences: (F * mean(R)) / total_frames
                scored = []
                for btns, occs in sequences.items():
                    freq = len(occs)
                    if freq < 3: continue 
                    
                    mean_r = sum(o['reward'] for o in occs) / freq
                    total_frames = sum(5 for _ in btns) 
                    
                    # Reward Speed Scoring
                    score = (freq * mean_r) / frames_to_seconds(total_frames)
                    if score > 5.0: 
                        scored.append({
                            "btns": btns,
                            "score": score,
                            "occ": occs[0] 
                        })
                
                scored.sort(key=lambda x: x['score'], reverse=True)
                
                # 5. Promote Top K to Skills
                promoted_count = 0
                for item in scored[:top_k]:
                    occ = item['occ']
                    start_step = occ['start_step']
                    v_vec = np.frombuffer(start_step['vision_vector'], dtype=np.float32).tolist() if start_step['vision_vector'] else None
                    
                    if v_vec:
                        macro_seq = [{"button": b, "frames": 5} for b in item['btns']]
                        
                        desc = f"Pattern Discovery: {' -> '.join(item['btns'])}"
                        print(f"✨ Macro Mined: {desc} (Score: {item['score']:.2f})")
                        
                        self.save_macro(
                            vision_vector=v_vec,
                            sequence=macro_seq,
                            map_id=start_step['map_id'],
                            coords=(0,0),
                            description=desc,
                            score=item['score'],
                            vision_hash=start_step['vision_hash'],
                            ocr_text=start_step['ocr'],
                            reliability=0.5
                        )
                        promoted_count += 1
                
                if promoted_count > 0:
                    print(f"✅ Successfully mined and promoted {promoted_count} sequences.")
                    self.consolidate_macros()
                else:
                    print("ℹ️ No high-scoring patterns found in current buffer.")

        except Exception as e:
            print(f"⚠️ Sequence mining failed: {e}")

    def _evolve_macro(self, parent_a: Optional[Dict[str, Any]], parent_b: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Handles the logic of mutating or crossing over macros with safety checks."""
        if not parent_a:
            return None
            
        try:
            # 1. Sequence Validation
            seq_a = json.loads(parent_a['macro_json']) if isinstance(parent_a['macro_json'], str) else parent_a['macro_json']
            if not seq_a:
                return None
                
            if parent_b:
                seq_b = json.loads(parent_b['macro_json']) if isinstance(parent_b['macro_json'], str) else parent_b['macro_json']
                if not seq_b:
                    # Fallback to mutation of parent_a if parent_b is invalid
                    new_seq = self._mutate_macro(seq_a)
                    desc = f"Mutation of {parent_a['name'] or parent_a['description']}"
                else:
                    new_seq = self._crossover_macros(seq_a, seq_b)
                    desc = f"Crossover: {parent_a['name'] or 'Macro'} x {parent_b['name'] or 'Macro'}"
            else:
                new_seq = self._mutate_macro(seq_a)
                desc = f"Mutation of {parent_a['name'] or parent_a['description']}"

            # 2. Vision Vector Validation (The 'Genome')
            v_blob = parent_a.get('vision_vector')
            if not v_blob:
                return None
                
            return {
                "sequence": new_seq,
                "vision_vector": np.frombuffer(v_blob, dtype=np.float32).tolist(),
                "desc": desc,
                "parent": parent_a
            }
        except Exception as e:
            print(f"⚠️ Internal evolution step failed: {e}")
            return None

    @database_write("db_path")
    def evolve_population(self, top_k: int = 5, variants_per_macro: int = 2):
        """Evolves the current macro population using mutations and crossover."""
        print("🧬 Running Macro Evolution (Genetic Algorithm)...")
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Select top-performing macros to act as parents
                cursor = conn.execute(
                    "SELECT * FROM skills ORDER BY (reliability * score) DESC LIMIT ?", 
                    (top_k,)
                )
                parents = [dict(row) for row in cursor.fetchall()]
                
                if len(parents) < 1: return

                new_variants = []
                
                # 1. Mutation
                for parent in parents:
                    # ADAPTIVE MUTATION RATE
                    rel = parent.get('reliability', 1.0)
                    actual_variants = variants_per_macro
                    if rel < 0.8: actual_variants += 2 
                    
                    for _ in range(actual_variants):
                        variant = self._evolve_macro(parent)
                        if variant:
                            new_variants.append(variant)
                
                # 2. Crossover (if we have at least 2 parents)
                if len(parents) >= 2:
                    for _ in range(variants_per_macro):
                        p1, p2 = random.sample(parents, 2)
                        variant = self._evolve_macro(p1, p2)
                        if variant:
                            new_variants.append(variant)

                # 3. Save new variants
                for var in new_variants:
                    p = var['parent']
                    self.save_macro(
                        vision_vector=var['vision_vector'],
                        sequence=var['sequence'],
                        map_id=p['map_id'],
                        coords=(0,0),
                        description=var['desc'],
                        score=p['score'] * 0.8, 
                        vision_hash=p['vision_hash'],
                        ocr_text=p['ocr_text'],
                        reliability=0.5
                    )
                
                print(f"✅ Evolution complete: Generated {len(new_variants)} new variants.")
                self.consolidate_macros()

        except Exception as e:
            print(f"⚠️ Macro evolution failed: {e}")

    def _mutate_macro(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Applies random mutations: insert, delete, or change duration (jitter)."""
        res = [dict(s) for s in sequence]
        if not res: return [{"button": "a", "frames": 5}]
        
        # Jittering logic: 30% chance to perform a global jitter across the whole sequence
        if random.random() < 0.3:
            for step in res:
                jitter = random.randint(-2, 2)
                step["frames"] = max(1, step.get("frames", 5) + jitter)
            return res

        mutation_type = random.choice(["insert", "delete", "duration", "button"])
        
        if mutation_type == "insert" and len(res) < 10:
            idx = random.randint(0, len(res))
            new_btn = random.choice(["a", "b", "start", "up", "down", "left", "right"])
            res.insert(idx, {"button": new_btn, "frames": 5})
        
        elif mutation_type == "delete" and len(res) > 1:
            idx = random.randint(0, len(res) - 1)
            res.pop(idx)
            
        elif mutation_type == "duration":
            idx = random.randint(0, len(res) - 1)
            res[idx]["frames"] = max(2, res[idx]["frames"] + random.choice([-5, 5]))
            
        elif mutation_type == "button":
            idx = random.randint(0, len(res) - 1)
            res[idx]["button"] = random.choice(["a", "b", "start", "up", "down", "left", "right"])
            
        return res

    def _crossover_macros(self, seq1: List[Dict[str, Any]], seq2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combines two sequences at a random midpoint."""
        if not seq1: return seq2
        if not seq2: return seq1
        
        cp1 = random.randint(1, len(seq1))
        cp2 = random.randint(0, len(seq2) - 1)
        
        return seq1[:cp1] + seq2[cp2:]

    def compose_hierarchical_skill(self, name: str, sub_skill_names: List[str], description: str = ""):
        """Manually composes a new hierarchical skill from existing ones."""
        print(f"🧩 Composing hierarchical skill: {name} ({len(sub_skill_names)} sub-skills)")
        
        v_vec = None
        map_id = -1
        v_hash = None
        ocr = None
        
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT vision_vector, map_id, vision_hash, ocr_text FROM skills WHERE name = ? OR description = ? LIMIT 1", 
                    (sub_skill_names[0], sub_skill_names[0])
                )
                row = cursor.fetchone()
                if row:
                    v_vec = np.frombuffer(row['vision_vector'], dtype=np.float32).tolist() if row['vision_vector'] else None
                    map_id = row['map_id']
                    v_hash = row['vision_hash']
                    ocr = row['ocr_text']
        except Exception: pass

        if v_vec:
            self.save_macro(
                vision_vector=v_vec,
                sequence=sub_skill_names,
                map_id=map_id,
                coords=(0,0),
                description=description or f"Hierarchical skill: {name}",
                vision_hash=v_hash,
                ocr_text=ocr,
                is_hierarchical=True,
                reliability=0.5
            )
            try:
                with get_db_connection(self.db_path) as conn:
                    conn.execute("UPDATE skills SET name = ? WHERE vision_hash = ? AND is_hierarchical = 1", (name, v_hash))
                    conn.commit()
            except Exception: pass

    def resolve_skill(self, name: str) -> List[Dict[str, Any]]:
        """Resolves a skill name into a flat action sequence."""
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT macro_json, is_hierarchical FROM skills WHERE name = ? LIMIT 1", (name,))
                row = cursor.fetchone()
                if row:
                    if row['is_hierarchical']:
                        return self._resolve_hierarchical_macro(row['macro_json'])
                    else:
                        return json.loads(row['macro_json'])
        except Exception: pass
        return []

    def load_raw_memories(self) -> List[Dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.execute("SELECT text, metadata FROM memories")
                return [{"text": row[0], "metadata": json.loads(row[1])} for row in cursor]
        except Exception as e:
            print(f"⚠️ Failed to load memories for optimization: {e}")
            return []

    def optimize(self):
        """Identifies repetitive success patterns and creates a Skill in the DB."""
        memories = self.load_raw_memories()
        if not memories: return

        grouped_skills = {}
        for mem in memories:
            meta = mem.get("metadata", {})
            map_id = meta.get("map_id", "global")
            
            if "MAJOR SUCCESS" in mem["text"]:
                if map_id not in grouped_skills:
                    grouped_skills[map_id] = []
                grouped_skills[map_id].append(mem)

        for map_id, items in grouped_skills.items():
            if len(items) >= 3:
                for item in items:
                    meta = item.get("metadata", {})
                    v_vec = meta.get("vision_vector")
                    v_hash = meta.get("vision_hash")
                    ocr_text = meta.get("ocr_text")
                    macro_json = meta.get("macro")
                    description = meta.get("description", f"Map {map_id} Success Sequence")
                    
                    if v_vec and macro_json:
                        self.save_macro(v_vec, macro_json, map_id, (0,0), description, vision_hash=v_hash, ocr_text=ocr_text, reliability=0.5)

        print("✨ Strategy Optimization Complete: Syncing successes to Skills table.")

    def _macro_to_nl(self, macro_json: str) -> str:
        """Translates a raw macro JSON string into a short natural language description."""
        try:
            seq = json.loads(macro_json)
            if not seq: return "No actions"
            
            parts = []
            current_btn = None
            current_count = 0
            
            for step in seq:
                btn = step.get('button', 'none').upper()
                if btn == current_btn:
                    current_count += 1
                else:
                    if current_btn:
                        parts.append(f"Press {current_btn}" + (f" {current_count} times" if current_count > 1 else ""))
                    current_btn = btn
                    current_count = 1
            
            if current_btn:
                parts.append(f"Press {current_btn}" + (f" {current_count} times" if current_count > 1 else ""))
                
            return " then ".join(parts)
        except Exception:
            return "Unknown sequence"

    def get_skills_for_map(self, map_id: Any) -> str:
        """Helper for agents to query relevant skills for a specific map, including global agnostic skills."""
        try:
            with get_db_connection(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT name, description, macro_json, reliability, score, times_succeeded FROM skills WHERE map_id = ? OR map_id = -1 ORDER BY (reliability * score) DESC LIMIT 5", 
                    (map_id,)
                )
                rows = cursor.fetchall()
                if not rows: return ""
                
                output = f"AVAILABLE PROVEN STRATEGIES (Map #{map_id} & Global):\n"
                for row in rows:
                    name = row['name'] or "Unnamed Macro"
                    desc = row['description']
                    macro_nl = self._macro_to_nl(row['macro_json'])
                    rel, succ = row['reliability'], row['times_succeeded']
                    
                    output += f"- {name}: {desc}\n  PROVEN ACTION: {macro_nl} (Succeeded {succ} times, Reliability: {rel:.2f})\n"
                return output
        except Exception as e:
            print(f"⚠️ Failed to get skills for map: {e}")
            return ""

# Compatibility class if any other module expects it
class SkillRegistry:
    def __init__(self, registry_path=None):
        self.optimizer = StrategyOptimizer()
    def get_best_macro(self, vision_vector, threshold=0.95):
        return self.optimizer.get_best_macro(vision_vector, threshold)
    def penalize_macro(self, vision_vector):
        self.optimizer.penalize_macro(vision_vector)
