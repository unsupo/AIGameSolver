import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import os

from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.models import Action
from autogameplayer.core.config_loader import load_game_config
from autogameplayer.core.registry import Registry
from autogameplayer.utils.launcher import ServerLauncher
from autogameplayer.utils.process import get_base_env

class UniversalRLWrapper(gym.Env):
    """A game-agnostic Gymnasium wrapper around the asynchronous MCP EmulatorEnvironment."""
    
    def __init__(self, config_path: str, port: int = None):
        super().__init__()
        self.config_path = config_path
        self.config = load_game_config(config_path)
        self.rom_path = self.config.rom
        
        # Determine port safely for multiprocessing
        if port is None:
            from autogameplayer.utils.process import port_allocator
            worker_idx = int(os.environ.get("AGP_WORKER_ID", "0"))
            self.port = port_allocator.allocate(offset=100, worker_id=worker_idx)
        else:
            self.port = port
        
        # Setup spaces based on config controller
        self.controller = Registry.create_controller(self.config.controller)
        self.buttons = self.controller.buttons
        
        self.action_space = spaces.Discrete(len(self.buttons))
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(384,), dtype=np.float32)
        
        self.launcher = ServerLauncher(self.rom_path, self.port)
        self.client = None
        self.env = None
        self.loop = asyncio.new_event_loop()
        
        # Start the background server
        self._start_server()
        
    def _start_server(self):
        self.launcher.start(env_vars=get_base_env(), config_path=self.config_path)
        
        # Wait for health check
        if not self.loop.run_until_complete(self.launcher.wait_until_healthy()):
            raise RuntimeError(f"RL Worker failed to start server on port {self.port}")
                
        # Initialize client and env
        url = f"http://localhost:{self.port}/sse"
        self.client = MCPClient(url)
        self.loop.run_until_complete(self.client.connect())
        
        # Create rewards dynamically from config
        rewards = []
        for r_cfg in self.config.rewards:
            rewards.append(Registry.create_reward(r_cfg.type, client=self.client, config=self.config, **r_cfg.params))
            
        self.env = EmulatorEnvironment(self.client, reward_functions=rewards)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Attempt to load bootstrap state 0 if exists
        try:
            self.loop.run_until_complete(self.client.call_tool("manage_checkpoint", {"action": "load", "slot": 0}))
        except Exception:
            pass
        
        obs = self.loop.run_until_complete(self.env.reset())
        vec = np.array(obs.state.vision_vector, dtype=np.float32)
        return vec, {}

    def step(self, action_idx):
        button = self.buttons[action_idx]
        action = Action(button=button, duration=5)
        
        obs, reward, done = self.loop.run_until_complete(self.env.step(action))
        vec = np.array(obs.state.vision_vector, dtype=np.float32)
        
        # Truncate handled by external wrappers or trainers if needed
        truncated = False
        
        return vec, float(reward), done, truncated, {}

    def close(self):
        if self.client:
            try:
                self.loop.run_until_complete(self.client.disconnect())
            except Exception:
                pass
        self.launcher.stop()
        self.loop.close()
