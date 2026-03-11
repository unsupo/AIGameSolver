import json
import sqlite3
import random
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import deque
import numpy as np
from pydantic import BaseModel

from autogameplayer.core.config import settings
from autogameplayer.core.models import Observation, Action
from autogameplayer.utils.llm import LLMClientProtocol
from autogameplayer.utils.vector import cosine_similarity
from autogameplayer.utils.database import ensure_ltm_schema, self_healing_db, get_db_connection, database_write

class StepRecord(BaseModel):
    map_id: int
    coords: tuple[int, int]
    button: str
    duration: int = 5
    ocr: str
    vision_hash: str = ""
    reward: float = 0.0
    vision_delta: float = 1.0
    vision_vector: Optional[List[float]] = None
    result_coords: Optional[tuple[int, int]] = None
    stuck: bool = False
    reasoning: str = ""

class ActionCompressor:
    """Detects repeating button patterns and suggests compressed macros."""
    def __init__(self, window_size: int = 20):
        self.buffer = deque(maxlen=window_size)

    def push(self, button: str):
        self.buffer.append(button)

    def detect_repeats(self) -> Optional[Dict[str, Any]]:
        """If the tail of the buffer is a repeating pattern, returns compression info."""
        seq = list(self.buffer)
        if len(seq) < 4:
            return None
        
        # Check for simple single-button spam (e.g., A A A A)
        last_btn = seq[-1]
        if all(b == last_btn for b in seq[-4:]):
            # Found at least 4 repeats
            count = 0
            for b in reversed(seq):
                if b == last_btn:
                    count += 1
                else:
                    break
            if count >= 4:
                return {"type": "spam", "button": last_btn, "count": count}
        
        # Check for 2-button patterns (e.g., A B A B)
        if len(seq) >= 6:
            p2 = seq[-2:]
            if seq[-4:-2] == p2 and seq[-6:-4] == p2:
                return {"type": "pattern", "pattern": p2, "repeats": 3}
                
        return None

class EpisodicMemory:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.steps = deque(maxlen=capacity)
        self.milestones: List[Dict[str, Any]] = []
        self.seen_dialogues: set[str] = set()
        self.compressor = ActionCompressor()

    def record_step(self, obs: Observation, action: Action):
        ctx = obs.state.context
        btn = action.button or "none"
        self.compressor.push(btn)
        record = StepRecord(
            map_id=ctx.get('map_id', 0),
            coords=(ctx.get('x', 0), ctx.get('y', 0)),
            button=action.button or "none",
            duration=action.duration or 5,
            ocr=obs.state.ocr_text or "",
            vision_hash=obs.state_hash,
            reward=ctx.get('last_reward', 0.0),
            vision_delta=obs.state.vision_delta,
            vision_vector=obs.state.vision_vector,
            reasoning=action.reasoning or ""
        )
        self.steps.append(record)

    def update_last_step(self, obs: Observation, stuck: bool):
        if not self.steps:
            return
        ctx = obs.state.context
        self.steps[-1].result_coords = (ctx.get('x', 0), ctx.get('y', 0))
        self.steps[-1].stuck = stuck
        self.steps[-1].reward = ctx.get('last_reward', 0.0)

    def get_recent_summary(self, n: int = 10) -> str:
        recent = list(self.steps)[-n:]
        summary = []
        for s in recent:
            if s.stuck:
                summary.append(f"Failed Action: Pressed {s.button.upper()} but hit a wall or obstacle. (Reward: {s.reward:.2f})")
            else:
                summary.append(f"Action: Pressed {s.button.upper()}. (Reward: {s.reward:.2f})")
        return "\n".join(summary) if summary else "No history."

    def warmup(self, historical_steps: List[Dict[str, Any]]):
        """Populates the episodic buffer with historical steps from a previous session."""
        if not historical_steps:
            return
        
        print(f"🔥 Warming up Episodic Memory with {len(historical_steps)} historical steps.")
        for step in reversed(historical_steps):
            try:
                record = StepRecord(
                    map_id=step.get('map_id', 0),
                    coords=json.loads(step['coords']) if isinstance(step['coords'], str) else step['coords'],
                    button=step.get('button', 'none'),
                    duration=step.get('duration', 5),
                    ocr=step.get('ocr', ""),
                    vision_hash=step.get('vision_hash', ""),
                    reward=step.get('reward', 0.0),
                    vision_delta=step.get('vision_delta', 1.0),
                    vision_vector=np.frombuffer(step['vision_vector'], dtype=np.float32).tolist() if step.get('vision_vector') else None,
                    stuck=bool(step.get('stuck', 0)),
                    reasoning=step.get('reasoning', "Historical memory")
                )
                self.steps.append(record)
                self.compressor.push(record.button)
            except Exception as e:
                print(f"⚠️ Warmup error for step: {e}")

class LongTermMemory:
    """A RAG system backed by SQLite for efficiency and persistence."""
    def __init__(self, client: LLMClientProtocol, model: str = None, storage_path: str = None):
        self.client = client
        if storage_path is None:
            self.storage_path = settings.models_dir / "long_term_memory.db"
        else:
            self.storage_path = Path(storage_path)
            
        self.model = model or settings.default_embedding_model
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        print(f"🗄️ Initializing RAG Database at: {self.storage_path}")
        ensure_ltm_schema(self.storage_path)

    @database_write("storage_path")
    @self_healing_db("storage_path")
    async def add_step_to_replay(self, session_id: str, step_index: int, record: StepRecord):
        """Persists every action and outcome to the replay buffer."""
        try:
            v_vec = np.array(record.vision_vector, dtype=np.float32).tobytes() if record.vision_vector else None
            with get_db_connection(self.storage_path) as conn:
                conn.execute(
                    """INSERT INTO replay_buffer 
                    (session_id, step_index, map_id, coords, button, duration, reward, vision_delta, vision_vector, vision_hash, ocr, stuck, reasoning, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id, step_index, record.map_id, json.dumps(record.coords), 
                        record.button, record.duration, record.reward, record.vision_delta, 
                        v_vec, record.vision_hash, record.ocr, 1 if record.stuck else 0, record.reasoning, time.time()
                    )
                )
        except Exception:
            pass 

    @self_healing_db("storage_path")
    def get_steps_by_session(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves history for analysis. If session_id is None, gets the latest steps globally."""
        try:
            with get_db_connection(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                if session_id:
                    cursor = conn.execute(
                        "SELECT * FROM replay_buffer WHERE session_id = ? ORDER BY step_index DESC LIMIT ?",
                        (session_id, limit)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM replay_buffer ORDER BY timestamp DESC LIMIT ?",
                        (limit,)
                    )
                return [dict(row) for row in cursor]
        except Exception:
            return []

    @self_healing_db("storage_path")
    def get_vision_vector(self, session_id: str, step_index: int) -> Optional[List[float]]:
        """Retrieves the vision vector for a specific step in a session."""
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute(
                    "SELECT vision_vector FROM replay_buffer WHERE session_id = ? AND step_index = ?",
                    (session_id, step_index)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return np.frombuffer(row[0], dtype=np.float32).tolist()
        except Exception:
            pass
        return None

    @self_healing_db("storage_path")
    def get_successful_action(self, vision_vector: List[float], threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        """Queries the replay buffer for highly successful actions in similar visual states."""
        if not vision_vector:
            return None
        q_vec = np.array(vision_vector)
        
        try:
            with get_db_connection(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT button, duration, reward, vision_vector FROM replay_buffer WHERE reward > 0.1 AND stuck = 0 ORDER BY reward DESC LIMIT 50"
                )
                for row in cursor:
                    if row['vision_vector']:
                        m_vec = np.frombuffer(row['vision_vector'], dtype=np.float32)
                        if len(m_vec) == len(q_vec):
                            sim = cosine_similarity(q_vec, m_vec)
                            if sim > threshold:
                                return {
                                    "button": row['button'],
                                    "duration": row['duration'],
                                    "reward": row['reward']
                                }
        except Exception as e:
            print(f"⚠️ Failed to query successful action: {e}")
        return None

    @database_write("storage_path")
    @self_healing_db("storage_path")
    async def add_event_pattern(self, state_hash: str, action: Action, result_hash: str, reward: float, map_id: int):
        """Records a procedural event: state A + action -> state B."""
        try:
            with get_db_connection(self.storage_path) as conn:
                action_json = action.model_dump_json()
                cursor = conn.execute(
                    "SELECT id, frequency FROM event_patterns WHERE state_hash = ? AND action_json = ? AND result_hash = ?",
                    (state_hash, action_json, result_hash)
                )
                row = cursor.fetchone()
                if row:
                    conn.execute(
                        "UPDATE event_patterns SET frequency = frequency + 1, reward = (reward + ?) / 2.0, timestamp = ? WHERE id = ?",
                        (reward, time.time(), row[0])
                    )
                else:
                    conn.execute(
                        """INSERT INTO event_patterns 
                        (state_hash, action_json, result_hash, reward, map_id, frequency, timestamp)
                        VALUES (?, ?, ?, ?, ?, 1, ?)""",
                        (state_hash, action_json, result_hash, reward, map_id, time.time())
                    )
                conn.commit()
        except Exception:
            pass

    @database_write("storage_path")
    @self_healing_db("storage_path")
    async def record_collision(self, map_id: int, x: int, y: int):
        """Gradually increments the impassable score for a tile. Starts at 0.5, capped at 1.0."""
        try:
            with get_db_connection(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT impassable_score FROM explored_locations WHERE map_id = ? AND x = ? AND y = ?", (map_id, x, y))
                row = cursor.fetchone()
                
                if row:
                    current_score = row['impassable_score'] or 0.0
                    # Increment by 0.2, capped at 1.0
                    new_score = min(1.0, current_score + 0.2)
                    if current_score == 0.0: new_score = 0.5 # Initial penalty
                else:
                    new_score = 0.5
                
                conn.execute(
                    """INSERT INTO explored_locations (map_id, x, y, impassable_score, last_seen) 
                    VALUES (?, ?, ?, ?, ?) 
                    ON CONFLICT(map_id, x, y) DO UPDATE SET impassable_score = ?, last_seen = ?""",
                    (map_id, x, y, new_score, time.time(), new_score, time.time())
                )
                conn.commit()
        except Exception: pass

    @database_write("storage_path")
    @self_healing_db("storage_path")
    async def record_walkable(self, map_id: int, x: int, y: int):
        """Reduces the impassable score for a tile (confidence building)."""
        try:
            with get_db_connection(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT impassable_score FROM explored_locations WHERE map_id = ? AND x = ? AND y = ?", (map_id, x, y))
                row = cursor.fetchone()
                
                if row:
                    current_score = row['impassable_score'] or 0.0
                    # Build confidence: reduce penalty by 0.2
                    new_score = max(0.0, current_score - 0.2)
                    
                    conn.execute(
                        "UPDATE explored_locations SET impassable_score = ?, last_seen = ? WHERE map_id = ? AND x = ? AND y = ?",
                        (new_score, time.time(), map_id, x, y)
                    )
                    conn.commit()
                else:
                    # Brand new walkable tile
                    await self.record_location(map_id, x, y, impassable_score=0.0)
        except Exception: pass

    @database_write("storage_path")
    @self_healing_db("storage_path")
    async def record_warp(self, map_id: int, x: int, y: int):
        """Mark a tile as a warp point (is_warp=1)."""
        await self.record_location(map_id, x, y, is_warp=1)

    @database_write("storage_path")
    @self_healing_db("storage_path")
    async def record_location(self, map_id: int, x: int, y: int, state: int = 1, impassable_score: float = None, is_warp: int = 0):
        """Persists a visited coordinate to the global world map, incrementing visit_count."""
        try:
            if impassable_score is None:
                impassable_score = 1.0 if state == 2 else 0.0
                
            with get_db_connection(self.storage_path) as conn:
                conn.execute(
                    """INSERT INTO explored_locations (map_id, x, y, state, impassable_score, is_warp, visit_count, last_seen) 
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?) 
                    ON CONFLICT(map_id, x, y) DO UPDATE SET 
                        last_seen = ?, 
                        state = ?, 
                        impassable_score = ?, 
                        is_warp = MAX(is_warp, ?),
                        visit_count = visit_count + 1""",
                    (map_id, x, y, state, impassable_score, is_warp, time.time(), time.time(), state, impassable_score, is_warp)
                )
                conn.commit()
        except Exception as e:
            print(f"⚠️ Failed to record location: {e}")

    @self_healing_db("storage_path")
    def get_visit_count(self, map_id: int, x: int, y: int) -> int:
        """Returns the number of times a specific tile has been visited (Heat Map)."""
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute(
                    "SELECT visit_count FROM explored_locations WHERE map_id = ? AND x = ? AND y = ?",
                    (map_id, x, y)
                )
                row = cursor.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    @self_healing_db("storage_path")
    def get_blocked_tiles_summary(self, map_id: int) -> str:
        """Returns a list of confirmed obstacles (score > 0.4) for the current map."""
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute(
                    "SELECT x, y, impassable_score FROM explored_locations WHERE map_id = ? AND impassable_score > 0.4 ORDER BY last_seen DESC LIMIT 10",
                    (map_id,)
                )
                obstacles = []
                for row in cursor:
                    severity = "BLOCKED" if row[2] >= 0.9 else "POTENTIALLY BLOCKED"
                    obstacles.append(f"({row[0]}, {row[1]}) [{severity}]")
                
                if not obstacles:
                    return "No confirmed obstacles."
                return f"KNOWN OBSTACLES: {', '.join(obstacles)}"
        except Exception:
            return "Obstacle data unavailable."

    @self_healing_db("storage_path")
    def get_explored_summary(self, map_id: int) -> str:
        """Returns a human-readable summary of explored tiles on a specific map."""
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute(
                    "SELECT x, y, is_warp FROM explored_locations WHERE map_id = ? ORDER BY x, y",
                    (map_id,)
                )
                tiles = cursor.fetchall()
                if not tiles:
                    return "This map is unexplored."
                
                xs = [t[0] for t in tiles]
                ys = [t[1] for t in tiles]
                warps = [f"({t[0]}, {t[1]})" for t in tiles if t[2] == 1]
                
                summary = f"EXPLORED: {len(tiles)} unique tiles. Range: X({min(xs)}-{max(xs)}), Y({min(ys)}-{max(ys)})."
                if warps:
                    summary += f" WARP TILES: {', '.join(warps[:5])}"
                return summary
        except Exception:
            return "Exploration data unavailable."

    @database_write("storage_path")
    @self_healing_db("storage_path")
    async def add_memory(self, text: str, metadata: Dict[str, Any]):
        """Embeds and stores a milestone or discovery in SQLite."""
        try:
            emb_list = await self.client.acreate_embedding(text, model=self.model)
            embedding = np.array(emb_list, dtype=np.float32)
            
            with get_db_connection(self.storage_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO memories (text, metadata, embedding) VALUES (?, ?, ?)",
                    (text, json.dumps(metadata), embedding.tobytes())
                )
            print(f"💾 Milestone Recorded: {text[:50]}...")
        except Exception as e:
            print(f"⚠️ RAG Storage failed: {e}")

    @self_healing_db("storage_path")
    async def query(self, text: str, current_map_id: Optional[int] = None, top_k: int = 3) -> List[str]:
        """Retrieves relevant memories using Vector Similarity + Metadata Filtering."""
        try:
            emb_list = await self.client.acreate_embedding(text, model=self.model)
            query_embedding = np.array(emb_list, dtype=np.float32)
            
            results = []
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute("SELECT text, metadata, embedding FROM memories")
                for row in cursor:
                    m_text, m_meta_json, m_emb_bytes = row
                    m_meta = json.loads(m_meta_json)
                    if m_emb_bytes:
                        m_emb = np.frombuffer(m_emb_bytes, dtype=np.float32)
                        sim = cosine_similarity(query_embedding, m_emb)
                        sim += random.uniform(-0.05, 0.05)
                        if current_map_id is not None and m_meta.get("map_id") == current_map_id:
                            sim += 0.2
                        if m_meta.get("type") == "rule":
                            sim += 0.3
                        results.append((sim, m_text))
            
            results.sort(key=lambda x: x[0], reverse=True)
            return [t for sim, t in results[:top_k] if sim > 0.6]
        except Exception as e:
            print(f"⚠️ RAG Retrieval failed: {e}")
            return []

    @self_healing_db("storage_path")
    async def query_by_vision(self, vision_vector: List[float], top_k: int = 1) -> List[str]:
        """Retrieves memories by matching the Vision Encoder latent vector stored in metadata."""
        if not vision_vector:
            return []
        q_vec = np.array(vision_vector)
        
        results = []
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute("SELECT text, metadata FROM memories")
                for row in cursor:
                    m_text, m_meta_json = row
                    m_meta = json.loads(m_meta_json)
                    v_vec = m_meta.get("vision_vector")
                    if v_vec:
                        m_vec = np.array(v_vec)
                        sim = cosine_similarity(q_vec, m_vec)
                        results.append((sim, m_text))
            
            results.sort(key=lambda x: x[0], reverse=True)
            return [t for sim, t in results[:top_k] if sim > 0.90]
        except Exception as e:
            print(f"⚠️ Vision query failed: {e}")
            return []

    def _save_to_disk(self):
        """No-op for SQLite implementation, kept for compatibility."""
        pass

    @database_write("storage_path")
    @self_healing_db("storage_path")
    def delete_redundant_memories(self):
        """Deduplicates 'experience' type memories while keeping 'rule' and 'curriculum' unique."""
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute("SELECT MIN(id), text FROM memories GROUP BY text")
                keep_ids = [row[0] for row in cursor]
                placeholders = ','.join(['?'] * len(keep_ids))
                conn.execute(f"DELETE FROM memories WHERE id NOT IN ({placeholders})", keep_ids)
                print("🧹 SQLite Pruning Complete.")
        except Exception as e:
            print(f"⚠️ SQLite Pruning Failed: {e}")

    @property
    def memories(self) -> List[Dict[str, Any]]:
        """Compatibility property for pruning logic."""
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute("SELECT text, metadata FROM memories")
                return [{"text": row[0], "metadata": json.loads(row[1])} for row in cursor]
        except Exception:
            return []

    @memories.setter
    @database_write("storage_path")
    def memories(self, value: List[Dict[str, Any]]):
        """Setter for compatibility with pruning logic. Wipes and re-inserts."""
        try:
            with get_db_connection(self.storage_path) as conn:
                conn.execute("DELETE FROM memories")
                for m in value:
                    conn.execute("INSERT INTO memories (text, metadata) VALUES (?, ?)", (m['text'], json.dumps(m['metadata'])))
        except Exception:
            pass

    @property
    def embeddings(self) -> List[np.ndarray]:
        """Compatibility property."""
        try:
            with get_db_connection(self.storage_path) as conn:
                cursor = conn.execute("SELECT embedding FROM memories")
                return [np.frombuffer(row[0], dtype=np.float32) if row[0] else np.zeros(1) for row in cursor]
        except Exception:
            return []

    @embeddings.setter
    def embeddings(self, value: List[np.ndarray]):
        """Compatibility setter."""
        pass
