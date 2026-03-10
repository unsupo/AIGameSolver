from pathlib import Path
from autogameplayer.core.config import settings
from autogameplayer.brains.agentic.memory import LongTermMemory
from autogameplayer.brains.agentic.reflector import ReflectionAgent

class ReplayProcessor:
    """Processes historical JSON episodes and ingests them into the learning system."""
    def __init__(self, ltm: LongTermMemory, reflector: ReflectionAgent):
        self.ltm = ltm
        self.reflector = reflector

    async def process_all(self, dataset_dir: Path = None):
        if dataset_dir is None:
            dataset_dir = settings.datasets_dir
            
        if not dataset_dir.exists():
            print(f"⚠️ Dataset directory {dataset_dir} does not exist.")
            return

        episodes = list(dataset_dir.glob("episode_*.json"))
        if not episodes:
            print("ℹ️ No JSON episodes found to process.")
            return

        print(f"🎬 Processing {len(episodes)} historical episodes...")
        for ep_path in episodes:
            await self.reflector.ingest_json_episode(str(ep_path), self.ltm)
            # Mark as processed or move to an 'ingested' subfolder
            ingested_dir = dataset_dir / "ingested"
            ingested_dir.mkdir(exist_ok=True)
            ep_path.rename(ingested_dir / ep_path.name)
            
        print("✅ Finished reprocessing historical data.")
