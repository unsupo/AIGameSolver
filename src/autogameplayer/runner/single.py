import sys
import sqlite3
import json
from pathlib import Path
import asyncio
import subprocess

from autogameplayer.core.config_loader import load_game_config
from autogameplayer.core.config import settings
from autogameplayer.utils.launcher import ServerLauncher
from autogameplayer.utils.process import get_base_env


def list_skills():
    """Queries the LTM database and prints learned skills in a readable format."""
    db_path = settings.models_dir / "long_term_memory.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found at {db_path}")
        return

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT name, description, reliability, score, times_run, macro_json FROM skills ORDER BY (reliability * score) DESC"
            )
            rows = cursor.fetchall()

            if not rows:
                print("ℹ️  No skills distilled into the database yet.")
                return

            print(
                f"\n✨ Learned Skills Registry ({len(rows)} skills found)\n" + "=" * 80
            )
            print(
                f"{'NAME':<20} | {'REL.':<5} | {'SCORE':<5} | {'RUNS':<4} | {'SEQUENCE SUMMARY'}"
            )
            print("-" * 80)

            for r in rows:
                name = r["name"] or "Pending..."
                reliability = f"{r['reliability']:.2f}"
                score = f"{r['score']:.2f}"
                runs = r["times_run"]

                # Create a sequence summary from macro_json
                try:
                    macro = json.loads(r["macro_json"])
                    # Standard Action model has 'button' and 'repeat'
                    # Handle both flat lists and nested structures if applicable
                    if isinstance(macro, list):
                        steps = []
                        for step in macro[:5]:  # Show first 5 steps
                            btn = step.get("button", "?")
                            rep = step.get("repeat", 1)
                            steps.append(
                                f"{btn.upper()}" + (f"x{rep}" if rep > 1 else "")
                            )
                        seq_summary = " -> ".join(steps)
                        if len(macro) > 5:
                            seq_summary += f" ... (+{len(macro) - 5} more)"
                    else:
                        seq_summary = str(r["description"])[:40]
                except json.JSONDecodeError, TypeError:
                    seq_summary = str(r["description"])[:40]

                print(
                    f"{name:<20} | {reliability:<5} | {score:<5} | {runs:<4} | {seq_summary}"
                )
            print("=" * 80 + "\n")
    except Exception as e:
        print(f"❌ Error reading skills: {e}")


async def run():
    import argparse

    parser = argparse.ArgumentParser(description="AutoGamePlayer Single Runner")
    parser.add_argument("--rom", type=str, help="Path to ROM")
    parser.add_argument("--config", type=str, help="Path to Config")
    parser.add_argument(
        "--verify", action="store_true", help="Run control verification before starting"
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip launching the Streamlit dashboard",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List all learned skills from the database and exit",
    )
    args = parser.parse_args()

    if args.list_skills:
        list_skills()
        return

    print("🚀 Initializing AutoGamePlayer POC...")

    # Use provided config or find one
    config_path = args.config
    if not config_path:
        # Better fallback: look for ANY yaml in configs/
        yaml_files = list(Path(settings.base_dir / "configs").glob("*.yaml"))
        if yaml_files:
            config_path = str(yaml_files[0])
        else:
            print("❌ Error: No configuration file found in ./configs/")
            return

    config = load_game_config(config_path)
    rom_path = args.rom or config.rom
    print(f"✅ Using Config: {config_path} | ROM: {rom_path}")

    # Setup Environment
    env = get_base_env()
    ai_proc = None
    ui_proc = None

    # 1. Start MCP Server using centralized Launcher
    launcher = ServerLauncher(rom_path, settings.server_port)
    print(f"🔌 Starting Game Server (logging to {launcher.log_path})...")
    server_proc = launcher.start(env_vars=env, config_path=config_path)

    # 2. Wait for Server Health Check
    if not await launcher.wait_until_healthy():
        print("❌ Server failed to start.")
        launcher.stop()
        return

    # 3. Stabilize
    await asyncio.sleep(3.0)

    # 4. Start Dashboard
    dashboard_log = None
    if not args.no_dashboard:
        print("\n📊 Starting Nexus Dashboard (logging to logs/dashboard.log)...")
        dashboard_log = open(settings.base_dir / "logs" / "dashboard.log", "w")
        dashboard_path = settings.base_dir / "src" / "autogameplayer" / "dashboard.py"
        ui_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_path),
                "--server.headless",
                "true",
                "--server.port",
                "8501",
            ],
            env=env,
            stdout=dashboard_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print("\n🎊 SYSTEM ONLINE. Access Dashboard at http://localhost:8501")
    else:
        print("\n🎊 SYSTEM ONLINE. Dashboard disabled via --no-dashboard.")

    # 5. Start Background Learner (MuZero)
    print("🧠 Starting MuZero Background Learner...")
    # Piping to stdout so we can see the [MuZero] epochs in the main console
    learner_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "autogameplayer.muzero.trainer",
            "--continuous",
            "--batch-size",
            "32",
        ],
        env=env,
        start_new_session=True,
    )

    # 6. Start AI Player
    print(f"🤖 Starting AI Player ({config.brain})...")
    ai_proc = subprocess.Popen(
        [sys.executable, "-m", "autogameplayer.runner.cli", "--config", config_path],
        env=env,
        start_new_session=True,
    )

    try:
        # Keep the main process alive while children run
        while server_proc.poll() is None:
            if ai_proc.poll() is not None:
                # AI Player exited unexpectedly
                print("⚠️ AI Player exited unexpectedly.")
                break
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n✨ Initiating Graceful Shutdown...")

        # 1. Stop AI Player first (Client)
        from autogameplayer.utils.process import terminate_process

        terminate_process(ai_proc, "AI Player")

        # 2. Stop Learner
        terminate_process(learner_proc, "Background Learner")

        # 3. Stop UI
        if ui_proc:
            terminate_process(ui_proc, "Dashboard")

        # 4. Stop Server last
        print("  - Stopping Game Server...")
        launcher.stop()
        if dashboard_log:
            dashboard_log.close()
        print("✅ Shutdown Complete.")


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
