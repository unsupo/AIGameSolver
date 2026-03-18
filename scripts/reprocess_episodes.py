import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from autogameplayer.core.config import settings
from autogameplayer.utils.llm import get_llm_client
from autogameplayer.brains.agentic.memory import LongTermMemory
from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.brains.agentic.reflector import ReflectionAgent
from autogameplayer.core.replay_processor import ReplayProcessor


async def main():
    print("🔄 Initializing Replay Processor...")
    client = get_llm_client()
    model = settings.llm_model

    ltm = LongTermMemory(client)
    optimizer = StrategyOptimizer(client, model)
    reflector = ReflectionAgent(client, model, optimizer)
    processor = ReplayProcessor(ltm, reflector)

    await processor.process_all()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
