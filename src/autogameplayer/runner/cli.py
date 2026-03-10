import argparse
import asyncio
import signal
from pathlib import Path

from autogameplayer.core.config_loader import load_game_config, GameConfig
from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.runner import GameRunner
from autogameplayer.core.config import settings
from autogameplayer.core.registry import Registry
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

def get_rewards(config: GameConfig, client: MCPClient):
    rewards = []
    for r_cfg in config.rewards:
        rewards.append(Registry.create_reward(r_cfg.type, config=config, **r_cfg.params))
            
    return rewards

async def run_autogame(config_path: str):
    config = load_game_config(config_path)
    print(f"🚀 Launching {config.name} stack...")

    server_url = f"http://{settings.server_host}:{settings.server_port}/sse"
    client = MCPClient(server_url)
    await client.connect()

    try:
        # Load Slot 0 if it exists
        try:
            print("💾 Checking for bootstrap checkpoint in slot 0...")
            await client.call_tool("manage_checkpoint", {"action": "load", "slot": 0})
            print("🚀 Bootstrap checkpoint loaded! (Pallet Town Start)")
        except Exception:
            print("ℹ️ No bootstrap checkpoint found. Starting from current state.")

        # Prepare Env
        rewards = get_rewards(config, client)
        env = EmulatorEnvironment(client, reward_functions=rewards)
        
        # Create Controller
        controller = Registry.create_controller(config.controller)
        
        # Create Brain
        from autogameplayer.utils.llm import get_llm_client
        llm_client = get_llm_client()
        brain = Registry.create_brain(config.brain, controller, config=config, llm_client=llm_client)
        
        # Enable Macro Interception
        print(f"🔌 Enabling Macro Interception for {config.brain} brain.")
        brain = MacroAwareBrain(brain, brain.optimizer)

        # Handle Knowledge Ingestion
        if config.profile and config.profile.known_locations:
            from autogameplayer.core.knowledge import KnowledgeBase
            kb = KnowledgeBase(llm_client)
            for loc in config.profile.known_locations:
                if loc.endswith(".md"):
                    await kb.ingest_file(Path(loc))

        # Runner
        runner = GameRunner(env, brain, render_delay=config.render_delay)
        
        print("🎮 AI is now playing! (Press Ctrl+C to stop safely and save)")
        
        # Main Loop with Shutdown Hook
        shutdown_event = asyncio.Event()
        
        def handle_interrupt(*args):
            print("\n✨ Initiating Graceful Shutdown...")
            shutdown_event.set()
            
        signal.signal(signal.SIGINT, handle_interrupt)
        
        while not shutdown_event.is_set():
            # Run in small increments to check shutdown_event frequently
            await runner.run(steps=min(config.steps, 10))
            if shutdown_event.is_set():
                break
            await asyncio.sleep(0.1)
            
    finally:
        # Cleanup
        if 'brain' in locals():
            try:
                await asyncio.wait_for(brain.close(), timeout=2.0)
            except Exception:
                pass
        
        # Final silent cleanup
        try:
            await client.disconnect()
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="AutoGamePlayer CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    try:
        # Filter asyncio logs to reduce shutdown noise
        asyncio.run(run_autogame(args.config))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception:
        pass # Final catch-all for silent exit

if __name__ == "__main__":
    main()
