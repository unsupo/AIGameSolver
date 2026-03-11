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
    def __init__(self, worker_id: int, rom_path: str, config: GameConfig = None):
        self.worker_id = worker_id
        self.rom_path = rom_path
        self.config = config
        # Use global allocator to avoid conflict with main dashboard
        self.port = port_allocator.allocate(offset=100, worker_id=worker_id) 
        self.launcher = ServerLauncher(rom_path, self.port)

    def start_server(self):
        self.launcher.start(env_vars=get_base_env())

    def stop_server(self):
        self.launcher.stop()

    async def run_episode(self, genome: np.ndarray = None, steps=100, target_checkpoint=None, initial_slot: int = None) -> dict:
        """Encapsulates the lifecycle of a single episode."""
        if not await self.launcher.wait_until_healthy():
            return {"reward": -100.0, "reached_target": False}
            
        client = MCPClient(f"http://{settings.server_host}:{self.port}/sse")
        await client.connect()
        try:
            # 1. Initialize State
            await self._initialize_state(client, initial_slot)
            
            # 2. Setup Brain & Controller
            brain = self._setup_brain(genome)
            env = EmulatorEnvironment(client, reward_functions=[ExplorationReward()])
            
            # 3. Execute Episode
            runner = GameRunner(env, brain, enable_logging=False, render_delay=0)
            total_reward = await runner.run(steps=steps)
            
            # 4. Finalize & Validate
            await self._save_final_frame(client)
            reached_target = await self._check_milestone(client, target_checkpoint)

            return {"reward": total_reward, "reached_target": reached_target}
        finally:
            await client.disconnect()

    async def _initialize_state(self, client, initial_slot):
        slot = initial_slot if initial_slot is not None else settings.bootstrap_slot
        try:
            await client.call_tool("manage_checkpoint", {"action": "load", "slot": slot})
        except Exception:
            if slot != settings.bootstrap_slot:
                await client.call_tool("manage_checkpoint", {"action": "load", "slot": settings.bootstrap_slot})

    def _setup_brain(self, genome):
        # Determine controller from profile or fallback to extension
        controller_name = "gb"
        if hasattr(self, 'config') and self.config and self.config.profile:
            controller_name = self.config.profile.controller_type
        elif self.rom_path.lower().endswith(".gba"):
            controller_name = "gba"
            
        controller = Registry.create_controller(controller_name)

        if genome is not None:
            brain = Registry.create_brain("evolution", controller, genome=genome)
        else:
            from autogameplayer.utils.llm import get_llm_client
            brain = Registry.create_brain(settings.brain, controller, llm_client=get_llm_client())
        
        if hasattr(brain, 'optimizer') and brain.optimizer:
            from autogameplayer.brains.macro_wrapper import MacroAwareBrain
            brain = MacroAwareBrain(brain, brain.optimizer)
        return brain

    async def _save_final_frame(self, client):
        try:
            state_resp = await client.call_tool("get_game_state", {"include_ocr": False})
            state = GameState.model_validate_json(state_resp)
            img_data = base64.b64decode(state.image_data)
            img = Image.open(BytesIO(img_data))
            frame_path = settings.base_dir / "logs" / "training_frames" / f"worker_{self.worker_id}_final.png"
            os.makedirs(frame_path.parent, exist_ok=True)
            img.save(frame_path)
        except Exception as e:
            logging.debug(f"Worker {self.worker_id} failed to save frame: {e}")

    async def _check_milestone(self, client, target):
        if not target:
            return False
        try:
            val = await client.call_tool("read_memory", {"address": target.address})
            if int(val) >= target.target_value:
                await client.call_tool("manage_checkpoint", {"action": "save", "slot": settings.milestone_tmp_slot})
                return True
        except Exception:
            pass
        return False

def run_worker_episode(worker_id: int, rom_path: str, steps: int = 100, genome: np.ndarray = None, target_checkpoint=None, initial_slot: int = None, config: GameConfig = None) -> dict:
    """Entry point for the ProcessPoolExecutor."""
    if initial_slot is None:
        initial_slot = settings.bootstrap_slot
    worker = Worker(worker_id, rom_path, config=config)
    worker.start_server()
    try:
        # Use a fresh event loop for each process worker
        result = asyncio.run(worker.run_episode(genome=genome, steps=steps, target_checkpoint=target_checkpoint, initial_slot=initial_slot))
        return result
    except Exception:
        return {"reward": -100.0, "reached_target": False}
    finally:
        worker.stop_server()
