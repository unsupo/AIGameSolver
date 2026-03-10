import asyncio
from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.runner import GameRunner
from autogameplayer.core.controllers import GBAController
from autogameplayer.brains.random_brain import RandomBrain
from autogameplayer.rewards.exploration import ExplorationReward
from autogameplayer.rewards.ocr import OcrKeywordReward

async def main():
    # 1. Low-level Network Client
    client = MCPClient("http://localhost:8000/sse")
    await client.connect()
    
    try:
        # 2. High-level Environment with modular rewards
        rewards = [
            ExplorationReward(),
            OcrKeywordReward()
        ]
        env = EmulatorEnvironment(client, reward_functions=rewards)
        
        # 3. Decision Logic (Brain)
        controller = GBAController()
        brain = RandomBrain(controller)
        
        # 4. Standardized Runner
        runner = GameRunner(env, brain)
        await runner.run(steps=100)
        
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
