import time
import subprocess
import os
import signal
import sys
import sqlite3
import argparse
from pathlib import Path

# --- DATABASE MONITORING ---


def get_last_progress_state(db_path):
    """
    Returns (last_timestamp, last_coords, last_reward_sum) from the replay_buffer.
    Used to detect if the agent is physically stuck or not finding new rewards.
    """
    if not Path(db_path).exists():
        return None, None, 0.0

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get the latest step
        cursor.execute("""
            SELECT timestamp, map_id, coords, reward 
            FROM replay_buffer 
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()

        # Get total rewards in the last 100 steps to see if we're "farming" anything new
        cursor.execute(
            "SELECT SUM(reward) FROM (SELECT reward FROM replay_buffer ORDER BY id DESC LIMIT 100)"
        )
        reward_sum = cursor.fetchone()[0] or 0.0

        conn.close()

        if row:
            coords = f"{row['map_id']}:{row['coords']}"
            return row["timestamp"], coords, reward_sum
        return None, None, 0.0
    except Exception as e:
        print(f"⚠️ Error reading DB for progress: {e}")
        return None, None, 0.0


# --- PROCESS MANAGEMENT ---


def start_play_session(config="configs/pokemon_red_agentic.yaml", no_dashboard=False):
    print("🎮 Starting Play Session...")
    cmd = ["poetry", "run", "nexus", "--config", config]
    if no_dashboard:
        cmd.append("--no-dashboard")

    # Start in a new session so we can SIGINT the whole group
    p = subprocess.Popen(cmd, start_new_session=True)
    return p


def stop_play_session(p):
    print("🛑 Stopping Play Session gracefully...")
    if p.poll() is None:
        try:
            # Send SIGINT to the entire process group (AI -> Dashboard -> Server)
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("⚠️ Play session didn't stop gracefully, forcing kill...")
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception as e:
            print(f"⚠️ Error stopping play session: {e}")


def run_training(epochs, batch_size):
    print("🧠 Starting Dream Training (Offline Re-Analysis)...")
    p = subprocess.Popen(
        [
            "poetry",
            "run",
            "muzero_train",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ]
    )
    p.wait()
    print("✅ Training Complete.")


def cleanup_agp_processes():
    """Hard cleanup of any AGP-related processes to prevent port conflicts."""
    print("🧹 Cleaning up AGP processes...")
    try:
        # 1. Port cleanup
        from autogameplayer.utils.process import kill_process_on_port

        kill_process_on_port(8000)  # Server
        kill_process_on_port(8501)  # Streamlit

        # 2. Kill by process name if necessary (optional but safer)
        subprocess.run(["pkill", "-9", "-f", "autogameplayer"], check=False)
        subprocess.run(["pkill", "-9", "-f", "streamlit"], check=False)
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")


# --- MAIN LOOP ---


def main():
    parser = argparse.ArgumentParser(
        description="AutoGamePlayer Turn-Key Loop (Progress Aware)"
    )
    parser.add_argument(
        "--max-mins", type=int, default=120, help="Hard limit for a play session"
    )
    parser.add_argument(
        "--stagnant-mins",
        type=int,
        default=10,
        help="How long to wait before declaring stagnation",
    )
    parser.add_argument(
        "--no-reward-mins",
        type=int,
        default=15,
        help="How long to wait without rewards before restart",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Offline training epochs per cycle"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Offline training batch size"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/models/long_term_memory.db",
        help="Path to LTM database",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pokemon_red_agentic.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true", help="Launch without dashboard UI"
    )
    args = parser.parse_args()

    print("🚀 AutoGamePlayer: Turn-Key Progress Aware Loop")
    print(
        f"Rules: Stop & Train if stuck for {args.stagnant_mins}m, no rewards for {args.no_reward_mins}m, or max {args.max_mins}m."
    )

    cycle = 1
    try:
        while True:
            print("\n" + "=" * 50)
            print(f"🔄 CYCLE {cycle}")
            print("=" * 50)

            # Ensure a clean slate
            cleanup_agp_processes()

            play_proc = start_play_session(args.config, args.no_dashboard)

            start_time = time.time()
            last_coords = None
            last_coord_change_time = time.time()
            last_reward_sum = 0.0
            last_reward_change_time = time.time()
            stuck_duration = 0.0
            bored_duration = 0.0

            try:
                while True:
                    # 1. Check if process died
                    if play_proc.poll() is not None:
                        print("⚠️ Play session ended unexpectedly! Restarting cycle...")
                        break

                    # 2. Check Database for Progress
                    current_ts, current_coords, current_reward_sum = (
                        get_last_progress_state(args.db)
                    )

                    now = time.time()

                    if current_coords:
                        # Detect Coordinate Change
                        if current_coords != last_coords:
                            last_coords = current_coords
                            last_coord_change_time = now

                        # Detect Reward Change
                        if current_reward_sum > last_reward_sum:
                            last_reward_sum = current_reward_sum
                            last_reward_change_time = now

                        # --- EVALUATE STUCK CONDITIONS ---
                        # In Asynchronous mode, we no longer stop the session for stagnation.
                        # Instead, we just log it. The Actor will sync weights and hopefully
                        # branch into a new timeline.

                        stuck_duration = (now - last_coord_change_time) / 60
                        if stuck_duration > args.stagnant_mins:
                            if int(now - start_time) % 60 < 5:
                                print(
                                    f"⚠️ STAGNATION WARNING: Same coordinates for {stuck_duration:.1f} minutes."
                                )

                        bored_duration = (now - last_reward_change_time) / 60
                        if bored_duration > args.no_reward_mins:
                            if int(now - start_time) % 60 < 5:
                                print(
                                    f"⚠️ PROGRESS STALL: No new rewards for {bored_duration:.1f} minutes."
                                )

                    # Case C: Hard Timeout
                    total_duration = (now - start_time) / 60
                    if total_duration > args.max_mins:
                        print(
                            f"🕒 SESSION COMPLETE: Reached maximum time of {args.max_mins} minutes."
                        )
                        break

                    # Log status every 2 minutes
                    if int(now - start_time) % 120 < 10:
                        print(
                            f"🕒 Session: {total_duration:.1f}m | Stuck: {stuck_duration:.1f}m | No-Reward: {bored_duration:.1f}m"
                        )

                    time.sleep(10)  # Poll every 10 seconds

            except KeyboardInterrupt:
                print(
                    "\n👋 Loop interrupted by user. Waiting for graceful shutdown of play session..."
                )
                # When start_new_session=True is used, the child process group DOES NOT receive
                # terminal signals. We must explicitly send SIGINT to trigger its graceful shutdown.
                try:
                    if play_proc.poll() is None:
                        os.killpg(os.getpgid(play_proc.pid), signal.SIGINT)
                    play_proc.wait(timeout=30)
                except Exception as e:
                    print(f"⚠️ Error during graceful shutdown wait: {e}")
                    os.killpg(os.getpgid(play_proc.pid), signal.SIGKILL)
                sys.exit(0)

            # End play session
            if play_proc.poll() is None:
                stop_play_session(play_proc)

            print("🌙 AI has finished this session. Moving to next cycle.")
            cycle += 1

    except KeyboardInterrupt:
        print("\n👋 Exiting Turn-Key Loop.")


if __name__ == "__main__":
    main()
