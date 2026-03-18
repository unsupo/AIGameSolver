import argparse
import asyncio
import signal
import sys
from pathlib import Path

from autogameplayer.core.config_loader import load_game_config, GameConfig
from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.orchestrator import SessionOrchestrator
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.factory import SolverFactory
from autogameplayer.core.config import settings
from autogameplayer.core.registry import Registry
import autogameplayer.core.controllers  # noqa: F401

# Import modules to trigger registration
import autogameplayer.solvers.composite  # noqa: F401
import autogameplayer.solvers.decorators  # noqa: F401
import autogameplayer.solvers.macros  # noqa: F401
import autogameplayer.solvers.tree_search  # noqa: F401
import autogameplayer.solvers.mcts_solver  # noqa: F401
import autogameplayer.solvers.llm  # noqa: F401
import autogameplayer.solvers.llm_meta  # noqa: F401
import autogameplayer.solvers.skill_solver  # noqa: F401
import autogameplayer.solvers.router  # noqa: F401
import autogameplayer.solvers.adapter  # noqa: F401
import autogameplayer.brains.agentic_brain  # noqa: F401
import autogameplayer.brains.random_brain  # noqa: F401
import autogameplayer.brains.walk_brain  # noqa: F401
import autogameplayer.brains.llm_brain  # noqa: F401
import autogameplayer.muzero.brain  # noqa: F401
import autogameplayer.rewards.exploration  # noqa: F401
import autogameplayer.rewards.dialogue  # noqa: F401
import autogameplayer.rewards.ocr  # noqa: F401


def get_rewards(config: GameConfig, client: MCPClient):
    rewards = []
    for r_cfg in config.rewards:
        rewards.append(
            Registry.create_reward(r_cfg.type, config=config, **r_cfg.params)
        )

    return rewards


async def run_autogame(config_path: str, dry_run: bool = False):
    try:
        config = load_game_config(config_path)
    except Exception as e:
        print(f"❌ Config Validation Failed: {e}")
        return

    print(f"🚀 Launching {config.name} stack...")

    server_url = f"http://{settings.server_host}:{settings.server_port}/sse"
    client = MCPClient(server_url)
    
    if not dry_run:
        await client.connect()

    try:
        # Load bootstrap state if it exists
        if not dry_run:
            try:
                print(
                    f"💾 Checking for bootstrap checkpoint in slot {settings.bootstrap_slot}..."
                )
                await client.call_tool(
                    "manage_checkpoint", {"action": "load", "slot": settings.bootstrap_slot}
                )
                print("🚀 Bootstrap checkpoint loaded!")
            except Exception:
                print("ℹ️ No bootstrap checkpoint found. Starting from current state.")

        # Prepare Env
        rewards = get_rewards(config, client)
        env = EmulatorEnvironment(
            client, 
            reward_functions=rewards, 
            reward_schedule=config.reward_schedule
        )

        # Create Controller
        controller = Registry.create_controller(config.controller)
        supported_buttons = controller.buttons

        # Create Solver from Pipeline
        if config.agent_pipeline:
            print(f"🔌 Building Solver Pipeline: {config.agent_pipeline.type}")
            # We'll need StrategyOptimizer if macros are involved
            from autogameplayer.core.optimizer import StrategyOptimizer

            optimizer = StrategyOptimizer(config=config)

            solver = SolverFactory.create_solver(
                config.agent_pipeline,
                supported_buttons=supported_buttons,
                optimizer=optimizer,
            )
        else:
            # Fallback to legacy brain-to-solver adapter or default random
            try:
                from autogameplayer.solvers.adapter import BrainSolverAdapter

                print(f"🧠 No agent_pipeline found. Attempting to use Brain: {config.brain}")
                brain = Registry.create_brain(
                    config.brain, controller=controller, config=config
                )
                solver = BrainSolverAdapter(brain)
            except Exception as e:
                print(f"⚠️ Failed to create brain '{config.brain}': {e}")
                print("⚠️ Falling back to default random solver.")
                from autogameplayer.solvers.composite import RandomSolver

                solver = RandomSolver(supported_buttons=supported_buttons)

        if dry_run:
            print("✅ Dry Run Complete: All dependencies (Solver, Env, Controller) initialized successfully.")
            return

        # Handle Knowledge Ingestion
        from autogameplayer.utils.llm import get_llm_client
        llm_client = get_llm_client()
        if config.profile and config.profile.known_locations:
            from autogameplayer.core.knowledge import KnowledgeBase
            kb = KnowledgeBase(llm_client)
            for loc in config.profile.known_locations:
                if loc.endswith(".md"):
                    await kb.ingest_file(Path(loc))

        # Context
        context = AgentContext(game_id=config.name)

        # Orchestrator
        orchestrator = SessionOrchestrator(
            env, solver, context, render_delay=config.render_delay
        )

        print("🎮 AI is now playing! (Press Ctrl+C to stop safely and save)")

        # Main Loop with Shutdown Hook
        shutdown_event = asyncio.Event()

        def handle_interrupt(*args):
            print("\n✨ Initiating Graceful Shutdown...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, handle_interrupt)

        # --- FEATURE: Config Watcher for Hot-Swap ---
        async def watch_config():
            last_mtime = Path(config_path).stat().st_mtime
            while not shutdown_event.is_set():
                await asyncio.sleep(2.0)
                try:
                    current_mtime = Path(config_path).stat().st_mtime
                    if current_mtime > last_mtime:
                        print("⚙️ Config changed! Rebuilding solver...")
                        new_config = load_game_config(config_path)
                        if new_config.agent_pipeline:
                            new_solver = SolverFactory.create_solver(
                                new_config.agent_pipeline, 
                                supported_buttons=supported_buttons,
                                optimizer=optimizer
                            )
                            await orchestrator.hot_swap_solver(new_solver)
                        last_mtime = current_mtime
                except Exception as e:
                    print(f"⚠️ Hot-swap failed: {e}")

        # Start Watcher
        watcher_task = asyncio.create_task(watch_config())

        while not shutdown_event.is_set():
            # Run in small increments to check shutdown_event frequently
            await orchestrator.run(steps=min(config.steps, 10))
            if shutdown_event.is_set():
                break
            await asyncio.sleep(0.1)
        
        watcher_task.cancel()

    finally:
        # Cleanup
        if "solver" in locals():
            try:
                print("💾 Flushing solver state and saving weights...")
                total_reward = orchestrator.total_reward if "orchestrator" in locals() else 0.0
                await solver.on_episode_end(total_reward)
            except Exception as e:
                print(f"⚠️ Failed to save solver state: {e}")

        # Final Database Maintenance
        try:
            from autogameplayer.utils.database import prune_replay_buffer
            db_path = settings.models_dir / "long_term_memory.db"
            prune_replay_buffer(db_path)
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
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and dependencies without running"
    )
    parser.add_argument(
        "--validate-config", action="store_true", help="Run Pydantic validation and exit"
    )
    args = parser.parse_args()

    if args.validate_config:
        try:
            load_game_config(args.config)
            print(f"✅ Config {args.config} is valid.")
            return
        except Exception as e:
            print(f"❌ Config Validation Failed: {e}")
            sys.exit(1)

    try:
        # Filter asyncio logs to reduce shutdown noise
        asyncio.run(run_autogame(args.config, dry_run=args.dry_run))
    except KeyboardInterrupt, asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ AI Player crashed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
