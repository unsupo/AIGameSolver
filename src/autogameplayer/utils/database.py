import sqlite3
import time
from pathlib import Path
from typing import Callable, Any
import functools
import asyncio

def get_db_connection(db_path: Path, timeout: int = 30) -> sqlite3.Connection:
    """Returns a SQLite connection with WAL mode enabled and a long timeout."""
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    # Enable WAL mode for better concurrency
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass
    return conn

def database_write(db_path_attr: str, retries: int = 5, delay: float = 0.5):
    """
    Decorator for database write operations that implements a retry loop 
    specifically for 'database is locked' errors.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            last_err = None
            for i in range(retries):
                try:
                    return func(self, *args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_err = e
                    if "locked" in str(e).lower():
                        time.sleep(delay * (i + 1))
                        continue
                    raise e
            raise last_err

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            last_err = None
            for i in range(retries):
                try:
                    return await func(self, *args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_err = e
                    if "locked" in str(e).lower():
                        await asyncio.sleep(delay * (i + 1))
                        continue
                    raise e
            raise last_err

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator

def ensure_ltm_schema(db_path: Path):
    """Ensures that all necessary tables exist in the Long Term Memory database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Retry logic for shared DB initialization
    last_err = None
    for i in range(5):
        try:
            # Use get_db_connection to ensure WAL mode
            with get_db_connection(db_path, timeout=30) as conn:
                # 1. Memories Table (RAG)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT UNIQUE,
                        metadata TEXT,
                        embedding BLOB
                    )
                """)
                
                # 2. Replay Buffer Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS replay_buffer (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        step_index INTEGER,
                        map_id INTEGER,
                        coords TEXT,
                        button TEXT,
                        duration INTEGER,
                        reward REAL,
                        vision_delta REAL,
                        vision_vector BLOB,
                        vision_hash TEXT,
                        ocr TEXT,
                        stuck INTEGER,
                        reasoning TEXT,
                        timestamp REAL
                    )
                """)
                
                # 3. Skills/Macros Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS skills (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        map_id INTEGER,
                        vision_vector BLOB,
                        vision_hash TEXT,
                        ocr_text TEXT,
                        macro_json TEXT,
                        description TEXT,
                        name TEXT,
                        compressed INTEGER DEFAULT 0,
                        is_hierarchical INTEGER DEFAULT 0,
                        score REAL DEFAULT 5.0,
                        reliability REAL DEFAULT 0.5,
                        times_run INTEGER DEFAULT 0,
                        times_succeeded INTEGER DEFAULT 0
                    )
                """)
                
                # 4. Stuck Hashes Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stuck_hashes (
                        vision_hash TEXT PRIMARY KEY,
                        count INTEGER DEFAULT 1,
                        last_seen REAL
                    )
                """)
                
                # 5. Event Patterns (Procedural Memory)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS event_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        state_hash TEXT,
                        action_json TEXT,
                        result_hash TEXT,
                        reward REAL,
                        macro_id INTEGER,
                        map_id INTEGER,
                        frequency INTEGER DEFAULT 1,
                        timestamp REAL,
                        FOREIGN KEY(macro_id) REFERENCES skills(id)
                    )
                """)
                
                # 6. Explored Locations (Global World Map)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS explored_locations (
                        map_id INTEGER,
                        x INTEGER,
                        y INTEGER,
                        state INTEGER DEFAULT 1, -- Deprecated, use impassable_score
                        impassable_score REAL DEFAULT 0.0, -- 0.0 (Walkable) to 1.0 (Static Wall)
                        is_warp INTEGER DEFAULT 0, -- 1 if this tile triggers a map transition
                        visit_count INTEGER DEFAULT 1, -- Tracks how many times this specific tile was entered
                        last_seen REAL,
                        PRIMARY KEY(map_id, x, y)
                    )
                """)
                
                # --- MIGRATION: Add columns if they don't exist ---
                try:
                    conn.execute("ALTER TABLE explored_locations ADD COLUMN state INTEGER DEFAULT 1")
                except sqlite3.OperationalError: pass
                
                try:
                    conn.execute("ALTER TABLE explored_locations ADD COLUMN impassable_score REAL DEFAULT 0.0")
                except sqlite3.OperationalError: pass

                try:
                    conn.execute("ALTER TABLE explored_locations ADD COLUMN is_warp INTEGER DEFAULT 0")
                except sqlite3.OperationalError: pass

                try:
                    conn.execute("ALTER TABLE explored_locations ADD COLUMN visit_count INTEGER DEFAULT 1")
                except sqlite3.OperationalError: pass
                # -------------------------------------------------------
                
                conn.commit()
            return
        except sqlite3.OperationalError as e:
            last_err = e
            if "locked" in str(e).lower():
                time.sleep(1.0)
                continue
            raise e
    
    print(f"❌ Failed to initialize LTM Database at {db_path}: {last_err}")

def self_healing_db(db_path_attr: str):
    """
    Decorator that catches 'no such table' errors and runs ensure_ltm_schema.
    Expected to be used on methods of classes that have a 'self.[db_path_attr]' Path.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            db_path = getattr(self, db_path_attr)
            try:
                return func(self, *args, **kwargs)
            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    print(f"🛠️ Self-Healing: Missing table detected. Reconstructing schema for {db_path}...")
                    ensure_ltm_schema(db_path)
                    # Retry once after healing
                    return func(self, *args, **kwargs)
                raise e
        
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            db_path = getattr(self, db_path_attr)
            try:
                return await func(self, *args, **kwargs)
            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    print(f"🛠️ Self-Healing: Missing table detected. Reconstructing schema for {db_path}...")
                    ensure_ltm_schema(db_path)
                    # Retry once after healing
                    return await func(self, *args, **kwargs)
                raise e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator
