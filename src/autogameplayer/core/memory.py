import json
from abc import ABC, abstractmethod
from autogameplayer.core.config import settings
from autogameplayer.core.interfaces import IMemoryReader


class MemoryReader(IMemoryReader, ABC):
    """Legacy alias for IMemoryReader."""

    @abstractmethod
    async def read_byte(self, address: int) -> int:
        pass


class GlobalKnowledgeBase:
    """Persists discovered RAM addresses across sessions for any ROM."""

    def __init__(self, rom_hash: str):
        self.storage_dir = settings.base_dir / "datasets" / "knowledge"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"{rom_hash}_knowledge.json"
        self.knowledge = self._load()

    def _load(self) -> dict:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {"discovered_addresses": {}}
        return {"discovered_addresses": {}}

    def save_address(self, address: int, label: str):
        """Saves a confirmed RAM correlation to disk."""
        self.knowledge["discovered_addresses"][hex(address)] = label
        with open(self.file_path, "w") as f:
            json.dump(self.knowledge, f, indent=2)

    def get_known_vars(self) -> dict:
        return self.knowledge.get("discovered_addresses", {})
