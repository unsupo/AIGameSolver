import asyncio
import sys
import os
import time
import subprocess
from pathlib import Path
from sb3_contrib import RecurrentPPO
import numpy as np

from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.config import settings
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.controllers import StandardController, GBAController
from autogameplayer.brains.llm_brain import LLMBrain
from autogameplayer.rewards.exploration import ExplorationReward
from autogameplayer.core.models import Action

async def llm_coach_session(rom_path: str, model_path: str):
    print("🎓 Starting LLM Coach Session...")
    
    port = 8600
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = str(settings.base_dir / "src") + os.pathsep + env_vars.get("PYTHONPATH", "")
    
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "autogameplayer.server.mcp_server", "--rom", rom_path, "--port", str(port)],
        env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    
    time.sleep(5) # Wait for boot
    client = MCPClient(f"http://localhost:{port}/sse")
    await client.connect()
    
    try:
        # 1. Setup
        controller = GBAController() if rom_path.lower().endswith(".gba") else StandardController()
        env = EmulatorEnvironment(client, reward_functions=[ExplorationReward()])
        
        # Load latest model if it exists
        rl_model = None
        if os.path.exists(model_path):
            print(f"Loading RL Model: {model_path}")
            rl_model = RecurrentPPO.load(model_path)
            
        llm_brain = LLMBrain(controller)
        
        # Load latest state
        try: await client.call_tool("manage_checkpoint", {"action": "load", "slot": 0})
        except: pass
        
        obs = await env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        stagnation_counter = 0
        
        print("▶️ RL Agent is playing...")
        while True:
            # If we have an RL model and we aren't stuck, let RL play
            if rl_model and stagnation_counter < 10:
                vec = np.array(obs.state.vision_vector, dtype=np.float32).reshape(1, -1)
                action_idx, lstm_states = rl_model.predict(vec, state=lstm_states, episode_start=episode_starts, deterministic=True)
                button = controller.buttons[action_idx[0]]
                action = Action(button=button, duration=5)
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                # LLM takes over!
                if stagnation_counter == 10:
                    print("\n🛑 RL Agent is stuck! LLM Coach taking over...")
                    stagnation_counter = 11 # Prevent re-triggering message
                
                action = llm_brain.act(obs)
            
            # Step environment
            next_obs, reward, done = await env.step(action)
            
            # Check for stagnation (Exploration reward handles this)
            if reward <= 0:
                stagnation_counter += 1
            else:
                if stagnation_counter >= 10:
                    print("✅ LLM Coach got us unstuck! Saving new curriculum checkpoint.")
                    await client.call_tool("manage_checkpoint", {"action": "save", "slot": 0})
                    stagnation_counter = 0
                    print("▶️ Handing control back to RL Agent...")
                else:
                    stagnation_counter = 0
                    
            obs = next_obs
            
    except KeyboardInterrupt:
        print("\nEnding coaching session.")
    finally:
        await client.disconnect()
        server_proc.terminate()

def main():
    rom_dir = Path("./roms")
    roms = list(rom_dir.glob("*.gb*")) + list(Path(".").glob("*.gb*"))
    if not roms: return
    
    model_path = str(settings.base_dir / "models" / "rl_best_model.zip")
    asyncio.run(llm_coach_session(str(roms[0]), model_path))

if __name__ == "__main__":
    main()
