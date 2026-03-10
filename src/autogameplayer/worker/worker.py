import asyncio
import os
import numpy as np
import base64
import logging
from PIL import Image
from io import BytesIO

from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.runner import GameRunner
from autogameplayer.core.config import settings
from autogameplayer.core.models import GameState
from autogameplayer.core.registry import Registry
from autogameplayer.utils.launcher import ServerLauncher
from autogameplayer.utils.process import get_base_env, port_allocator
from autogameplayer.rewards.exploration import ExplorationReward

import autogameplayer.core.controllers # noqa: F401

# Import modules to trigger registration
from autogameplayer.brains.macro_wrapper import MacroAwareBrain # noqa: F401
import autogameplayer.brains.agentic_brain # noqa: F401
import autogameplayer.brains.llm_brain # noqa: F401
import autogameplayer.brains.random_brain # noqa: F401
import autogameplayer.brains.walk_brain # noqa: F401
import autogameplayer.rewards.exploration # noqa: F401
import autogameplayer.rewards.dialogue # noqa: F401
import autogameplayer.rewards.ocr # noqa: F401
import autogameplayer.rewards.pokemon # noqa: F401

class Worker:
    """Encapsulates a single evaluation instance."""
    def __init__(self, worker_id: int, rom_path: str):
        self.worker_id = worker_id
        self.rom_path = rom_path
        # Use global allocator to avoid conflict with main dashboard
        self.port = port_allocator.allocate(offset=100, worker_id=worker_id) 
        self.launcher = ServerLauncher(rom_path, self.port)

    def start_server(self):
        self.launcher.start(env_vars=get_base_env())

    def stop_server(self):
        self.launcher.stop()

    async def run_episode(self, genome: np.ndarray = None, steps=100, target_checkpoint=None, initial_slot: int = 0) -> dict:
        if not await self.launcher.wait_until_healthy():
            return {"reward": -100.0, "reached_target": False}
            
        client = MCPClient(f"http://{settings.server_host}:{self.port}/sse")
        await client.connect()
        try:
            # Load the specified starting state (e.g. for Timeline Branching)
            try:
                await client.call_tool("manage_checkpoint", {"action": "load", "slot": initial_slot})
            except Exception:
                # Fallback to Slot 0 if initial_slot fails
                if initial_slot != 0:
                    try:
                        await client.call_tool("manage_checkpoint", {"action": "load", "slot": 0})
                    except Exception:
                        pass
                
            # Fetch config to create rewards
            # Search for a config that matches the ROM or use a default
            # For simplicity in workers, we might need a better way to pass the config
            # But usually settings.llm_model etc are enough for the brain
            
            rewards = [ExplorationReward()]
            env = EmulatorEnvironment(client, reward_functions=rewards)
            
            # Determine controller
            controller_name = "gba" if self.rom_path.lower().endswith(".gba") else "gb"
            controller = Registry.create_controller(controller_name)

            from autogameplayer.utils.llm import get_llm_client
            llm_client = get_llm_client()

            if genome is not None:
                brain = Registry.create_brain("evolution", controller, genome=genome)
            else:
                brain = Registry.create_brain(settings.brain, controller, llm_client=llm_client)
            
            # --- FEATURE: Parallel Macro Interception ---
            if hasattr(brain, 'optimizer') and brain.optimizer:
                from autogameplayer.brains.macro_wrapper import MacroAwareBrain
                brain = MacroAwareBrain(brain, brain.optimizer)
            # --------------------------------------------
                
            # Disable logging and delay for training performance
            runner = GameRunner(env, brain, enable_logging=False, render_delay=0)
            total_reward = await runner.run(steps=steps)
            
            # Save final frame for dashboard visualization
            try:
                state_resp = await client.call_tool("get_game_state", {"include_ocr": False})
                state = GameState.model_validate_json(state_resp)
                img_data = base64.b64decode(state.image_data)
                img = Image.open(BytesIO(img_data))
                frame_path = settings.base_dir / "logs" / "training_frames" / f"worker_{self.worker_id}_final.png"
                os.makedirs(frame_path.parent, exist_ok=True)
                img.save(frame_path)
            except Exception as e:
                # Log error but don't fail the episode
                logging.debug(f"Worker {self.worker_id} failed to save frame: {e}")

            # CHECK MILESTONE
            reached_target = False
            if target_checkpoint:
                val = await client.call_tool("read_memory", {"address": target_checkpoint.address})
                if int(val) >= target_checkpoint.target_value:
                    reached_target = True
                    # Save THIS successful state as a potential new global start
                    await client.call_tool("manage_checkpoint", {"action": "save", "slot": 99})

            return {"reward": total_reward, "reached_target": reached_target}
        finally:
            await client.disconnect()

def run_worker_episode(worker_id: int, rom_path: str, steps: int = 100, genome: np.ndarray = None, target_checkpoint=None, initial_slot: int = 0) -> dict:
    """Entry point for the ProcessPoolExecutor."""
    worker = Worker(worker_id, rom_path)
    worker.start_server()
    try:
        # Use a fresh event loop for each process worker
        result = asyncio.run(worker.run_episode(genome=genome, steps=steps, target_checkpoint=target_checkpoint, initial_slot=initial_slot))
        return result
    except Exception:
        return {"reward": -100.0, "reached_target": False}
    finally:
        worker.stop_server()
