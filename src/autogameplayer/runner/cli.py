import argparse
import asyncio
import sys
import os

from autogameplayer.core.config_loader import load_game_config, GameConfig
from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.runner import GameRunner
from autogameplayer.core.config import settings
from autogameplayer.core.registry import Registry
import autogameplayer.core.controllers

# Import modules to trigger registration
from autogameplayer.brains.macro_wrapper import MacroAwareBrain
import autogameplayer.brains.agentic_brain
import autogameplayer.brains.llm_brain
import autogameplayer.brains.random_brain
import autogameplayer.brains.walk_brain
import autogameplayer.rewards

def get_rewards(config: GameConfig, client: MCPClient):
    rewards = []
    for r_cfg in config.rewards:
        rewards.append(Registry.create_reward(r_cfg.type, config=config, **r_cfg.params))
            
    return rewards

import signal

async def run_autogame(config_path: str):
    config = load_game_config(config_path)
    print(f"🚀 Launching {config.name} stack...")

    server_url = f"http://{settings.server_host}:{settings.server_port}/sse"
    client = MCPClient(server_url)
    brain = None
    
    # Setup shutdown event
    shutdown_event = asyncio.Event()
    
    def handle_signal():
        shutdown_event.set()
        
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)
    
    try:
        await client.connect()
        
        from autogameplayer.utils.llm import get_llm_client
        llm_client = get_llm_client()
        
        env = EmulatorEnvironment(client, reward_functions=get_rewards(config, client))
        controller = Registry.create_controller(config.controller)
        brain = Registry.create_brain(config.brain, controller, config=config, llm_client=llm_client)
        
        # --- FEATURE: Macro Interception ---
        # If the brain has an optimizer, wrap it to allow visual macro shortcuts
        if hasattr(brain, 'optimizer') and brain.optimizer:
            print(f"🔌 Enabling Macro Interception for {config.brain} brain.")
            brain = MacroAwareBrain(brain, brain.optimizer)
        # -----------------------------------
        
        # --- FEATURE: Auto-Resume (Bootstrap Slot 0) ---
        # If we have an intro-cleared checkpoint, start there!
        try:
            print("💾 Checking for bootstrap checkpoint in slot 0...")
            await client.call_tool("manage_checkpoint", {"action": "load", "slot": 0})
            print("🚀 Bootstrap checkpoint loaded! (Pallet Town Start)")
        except Exception:
            # Slot 0 might not exist if it's the first run
            print("✨ No bootstrap found. Starting from Title Screen.")
        # ---------------------------------------------

        runner = GameRunner(env, brain)        
        print("🎮 AI is now playing! (Press Ctrl+C to stop safely and save)")
        while not shutdown_event.is_set():
            # Run in small increments to check shutdown_event frequently
            await runner.run(steps=min(config.steps, 10))
            if shutdown_event.is_set(): break
            await asyncio.sleep(0.1)
            
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as e:
        msg = str(e).lower()
        # Common connection errors during shutdown or interruption
        conn_errors = ["peer closed connection", "connection closed", "cancel scope", "broken pipe", 
                       "all connection attempts failed", "remoteprotocolerror", "incomplete chunked read"]
        if any(err in msg for err in conn_errors):
            pass 
        elif msg.strip():
            print(f"⚠️ AI Loop Error: {e}")
    finally:
        # --- NEW: AUTO-SAVE ON EXIT ---
        # If we reached here due to signal or loop exit
        try:
            if client.session and (shutdown_event.is_set() or sys.exc_info()[0] is KeyboardInterrupt):
                print("\n🛑 Interrupted! Saving game state to Slot 1 before exiting...")
                # We need a fresh connection or reuse existing if alive
                await asyncio.wait_for(client.call_tool("manage_checkpoint", {"action": "save", "slot": 1}), timeout=3.0)
                print("✅ Game saved successfully.")
        except Exception:
            # Silent failure during shutdown save
            pass
        # ------------------------------
        
        # Cleanup brain tasks
        if brain:
            try:
                await asyncio.wait_for(brain.close(), timeout=2.0)
            except Exception: pass
        
        # Final silent cleanup
        try:
            await asyncio.wait_for(client.disconnect(), timeout=2.0)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="AutoGamePlayer Turnkey CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found at {args.config}")
        return

    try:
        # Use a single-entry run to avoid loop shutdown noise
        asyncio.run(run_autogame(args.config))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception:
        pass # Final catch-all for silent exit

if __name__ == "__main__":
    main()
