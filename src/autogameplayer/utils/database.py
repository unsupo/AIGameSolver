import sqlite3
import time
from pathlib import Path
from typing import Callable
import functools
import asyncio
import threading

# Thread-local storage for SQLite connections
_local = threading.local()


def check_and_add_column(
    conn: sqlite3.Connection, table: str, column: str, definition: str
):
    """Checks if a column exists in a table, and adds it if not."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        print(f"🛠️ DB Migration: Adding column '{column}' to table '{table}'")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def get_db_connection(db_path: Path, timeout: int = 30) -> sqlite3.Connection:
    """Returns a SQLite connection with WAL mode enabled and a long timeout, using a thread-local pool."""
    db_key = str(db_path)

    # Initialize thread-local storage if needed
    if not hasattr(_local, "connections"):
        _local.connections = {}

    # Reuse existing connection for this thread if available
    if db_key in _local.connections:
        try:
            # Simple ping to see if connection is alive
            _local.connections[db_key].execute("SELECT 1")
            return _local.connections[db_key]
        except sqlite3.Error:
            # Connection died, create a new one
            del _local.connections[db_key]

    # Create new connection
    conn = sqlite3.connect(db_key, timeout=timeout)
    # Enable WAL mode for better concurrency
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass

    _local.connections[db_key] = conn
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
                        type TEXT, -- milestone, reasoning, reflection, etc.
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
                        next_vision_vector BLOB,
                        hidden_state BLOB,
                        search_statistics BLOB,
                        vision_hash TEXT,
                        ocr TEXT,
                        stuck INTEGER,
                        reasoning TEXT,
                        timestamp REAL,
                        lstm_h BLOB,
                        lstm_c BLOB,
                        state_metadata TEXT,
                        personality_id INTEGER
                    )
                """)

                check_and_add_column(
                    conn, "replay_buffer", "episode_done", "INTEGER DEFAULT 0"
                )
                check_and_add_column(
                    conn, "replay_buffer", "priority", "REAL DEFAULT 1.0"
                )
                check_and_add_column(
                    conn, "replay_buffer", "rnd_error", "REAL DEFAULT 0.0"
                )
                check_and_add_column(
                    conn, "replay_buffer", "intrinsic_reward", "REAL DEFAULT 0.0"
                )
                check_and_add_column(
                    conn, "replay_buffer", "solver_name", "TEXT"
                )

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
                        state INTEGER DEFAULT 1,
                        impassable_score REAL DEFAULT 0.0,
                        is_warp INTEGER DEFAULT 0,
                        visit_count INTEGER DEFAULT 1,
                        last_seen REAL,
                        PRIMARY KEY(map_id, x, y)
                    )
                """)

                # 7. Dead Ends Table (State hashes that lead to stagnation across sessions)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dead_ends (
                        vision_hash TEXT PRIMARY KEY,
                        occurrence_count INTEGER DEFAULT 1,
                        last_session_id TEXT,
                        timestamp REAL
                    )
                """)

                # 8. Spatial Anchors Table (Teleport/Waypoint Persistence)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS spatial_anchors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        map_id INTEGER,
                        x INTEGER,
                        y INTEGER,
                        state_hash TEXT,
                        slot_id INTEGER,
                        description TEXT,
                        timestamp REAL
                    )
                """)

                # --- MIGRATIONS ---
                check_and_add_column(
                    conn, "explored_locations", "state", "INTEGER DEFAULT 1"
                )
                check_and_add_column(
                    conn, "explored_locations", "impassable_score", "REAL DEFAULT 0.0"
                )
                check_and_add_column(
                    conn, "explored_locations", "is_warp", "INTEGER DEFAULT 0"
                )
                check_and_add_column(
                    conn, "explored_locations", "visit_count", "INTEGER DEFAULT 1"
                )
                check_and_add_column(conn, "memories", "type", "TEXT")
                check_and_add_column(
                    conn, "replay_buffer", "next_vision_vector", "BLOB"
                )
                check_and_add_column(conn, "replay_buffer", "hidden_state", "BLOB")
                check_and_add_column(conn, "replay_buffer", "search_statistics", "BLOB")
                check_and_add_column(conn, "replay_buffer", "lstm_h", "BLOB")
                check_and_add_column(conn, "replay_buffer", "lstm_c", "BLOB")
                check_and_add_column(conn, "replay_buffer", "state_metadata", "TEXT")
                check_and_add_column(conn, "replay_buffer", "personality_id", "INTEGER")

                # --- NEW: MuZero Consistency Index ---
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_replay_hidden ON replay_buffer(hidden_state)"
                )

                # --- NEW: Agent57 Personality Index ---
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_replay_personality ON replay_buffer(personality_id)"
                )

                # --- NEW: SLAM Optimizations ---
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_explored_locations ON explored_locations(map_id, x, y)"
                )
                # ------------------

                conn.commit()
            return
        except sqlite3.OperationalError as e:
            last_err = e
            if "locked" in str(e).lower():
                time.sleep(1.0)
                continue
            raise e

    print(f"❌ Failed to initialize LTM Database at {db_path}: {last_err}")


def prune_replay_buffer(db_path: Path, max_rows: int = 100000):
    """
    Priority-ordered eviction policy. 
    Deletes rows with the lowest TD-priority first to preserve critical context.
    """
    try:
        with get_db_connection(db_path) as conn:
            # Check row count
            cursor = conn.execute("SELECT COUNT(*) FROM replay_buffer")
            count = cursor.fetchone()[0]
            
            if count > max_rows:
                to_delete = count - max_rows
                print(f"🗑️ Replay Buffer Full ({count} rows). Evicting {to_delete} low-priority steps...")
                
                # Delete by priority ascending (lowest first)
                conn.execute(f"""
                    DELETE FROM replay_buffer 
                    WHERE id IN (
                        SELECT id FROM replay_buffer 
                        ORDER BY priority ASC 
                        LIMIT {to_delete}
                    )
                """)
                conn.commit()
    except Exception as e:
        print(f"⚠️ Replay buffer pruning failed: {e}")


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
                    print(
                        f"🛠️ Self-Healing: Missing table detected. Reconstructing schema for {db_path}..."
                    )
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
                    print(
                        f"🛠️ Self-Healing: Missing table detected. Reconstructing schema for {db_path}..."
                    )
                    ensure_ltm_schema(db_path)
                    # Retry once after healing
                    return await func(self, *args, **kwargs)
                raise e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
