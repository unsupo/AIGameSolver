import sqlite3
import numpy as np
from pathlib import Path
from typing import List
from autogameplayer.core.config import settings
from autogameplayer.utils.llm import LLMClientProtocol
from autogameplayer.utils.vector import cosine_similarity


class KnowledgeBase:
    """A RAG system for external game knowledge (walkthroughs, wikis)."""

    def __init__(
        self, client: LLMClientProtocol, model: str = None, storage_path: str = None
    ):
        self.client = client
        if storage_path is None:
            self.storage_path = settings.models_dir / "external_knowledge.db"
        else:
            self.storage_path = Path(storage_path)

        self.model = model or settings.default_embedding_model
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Trigger async auto-ingest in a background task or synchronously check
        if not self.is_ready:
            print("📚 Knowledge Base empty. Searching for data to ingest...")
            self.sync_auto_ingest()

    def sync_auto_ingest(self):
        """Synchronous wrapper for auto_ingest to be called from __init__."""
        import asyncio

        try:
            # We use a temporary event loop if one isn't running, or run_coroutine_threadsafe
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.auto_ingest())
            else:
                loop.run_until_complete(self.auto_ingest())
        except Exception:
            pass

    async def auto_ingest(self):
        """Finds text files in data/knowledge and ingests them."""
        knowledge_dir = settings.base_dir / "data" / "knowledge"
        if not knowledge_dir.exists():
            return

        files = list(knowledge_dir.glob("*.txt")) + list(knowledge_dir.glob("*.md"))
        for file in files:
            print(f"📖 Auto-ingesting: {file.name}")
            with open(file, "r") as f:
                content = f.read()
                await self.ingest_text(content, source=file.name)

    @property
    def is_ready(self) -> bool:
        """Checks if there are any ingested documents in the knowledge base."""
        try:
            with sqlite3.connect(str(self.storage_path), timeout=5) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge")
                count = cursor.fetchone()[0]
                return count > 0
        except Exception:
            return False

    def _init_db(self):
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT,
                        content TEXT,
                        embedding BLOB
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"❌ Failed to initialize Knowledge DB: {e}")

    async def ingest_text(self, text: str, source: str = "walkthrough"):
        """Chunks and embeds a text file into the DB."""
        # Simple chunking by paragraph or fixed length
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]

        print(f"📚 Ingesting {len(chunks)} knowledge chunks from {source}...")
        for chunk in chunks:
            try:
                emb_list = await self.client.acreate_embedding(chunk, model=self.model)
                embedding = np.array(emb_list, dtype=np.float32)

                with sqlite3.connect(str(self.storage_path)) as conn:
                    conn.execute(
                        "INSERT INTO knowledge (source, content, embedding) VALUES (?, ?, ?)",
                        (source, chunk, embedding.tobytes()),
                    )
            except Exception as e:
                print(f"⚠️ Knowledge ingestion failed for chunk: {e}")

    async def query(self, query_text: str, top_k: int = 2) -> List[str]:
        """Retrieves relevant knowledge snippets."""
        try:
            emb_list = await self.client.acreate_embedding(query_text, model=self.model)
            query_embedding = np.array(emb_list, dtype=np.float32)

            results = []
            with sqlite3.connect(str(self.storage_path)) as conn:
                cursor = conn.execute("SELECT content, embedding FROM knowledge")
                for row in cursor:
                    content, emb_bytes = row
                    if emb_bytes:
                        m_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                        sim = cosine_similarity(query_embedding, m_emb)
                        results.append((sim, content))

            results.sort(key=lambda x: x[0], reverse=True)
            snippets = [t for sim, t in results[:top_k] if sim > 0.6]
            if snippets:
                print(f"📖 RAG: Found {len(snippets)} relevant knowledge snippets.")
            return snippets
        except Exception:
            return []
