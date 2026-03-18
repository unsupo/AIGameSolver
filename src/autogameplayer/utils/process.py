import os
import subprocess
import time
import signal
from typing import Optional
from autogameplayer.core.config import settings


def get_base_env():
    """Returns a copy of the environment with PYTHONPATH correctly set for autogameplayer."""
    env = os.environ.copy()
    src_dir = str(settings.base_dir / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    # SILENCE: macOS MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
    # This warning appears when a subprocess starts and its parent or a library tries to
    # toggle logging that was never active.
    env["MALLOC_STACK_LOGGING"] = "0"

    return env


def terminate_process(
    proc: Optional[subprocess.Popen],
    name: str,
    timeout: int = 5,
    kill_group: bool = True,
):
    """Gracefully terminates a process (and its group) with escalation to kill."""
    if proc is None or proc.poll() is not None:
        return

    print(f"  - Stopping {name}...")
    try:
        # If started with start_new_session=True, we should signal the whole group
        if kill_group:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # Already gone
        else:
            proc.terminate()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  - {name} timed out, forcing...")
            if kill_group:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.wait()
    except Exception as e:
        print(f"  - Error stopping {name}: {e}")


def kill_process_on_port(port: int):
    """Kills any process currently using the specified port (Unix-like systems)."""
    try:
        # Get PID of process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                print(f"  - Port {port} in use by PID {pid}. Cleaning up...")
                subprocess.run(["kill", "-9", pid], check=False)
                time.sleep(1.0)  # Wait for port release
    except Exception:
        pass  # Silent failure if lsof is missing or no process found


class PortAllocator:
    """Manages unique port assignments to avoid conflicts."""

    def __init__(self, base_port: int = 8000):
        self.base_port = base_port
        self._used = set()

    def allocate(self, worker_id: int = 0, offset: int = 100) -> int:
        port = self.base_port + worker_id + offset
        while port in self._used:
            port += 1
        self._used.add(port)
        return port


# Global allocator instance
port_allocator = PortAllocator(base_port=settings.server_port)
