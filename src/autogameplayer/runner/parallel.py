import subprocess
import sys
import time
import os
from typing import List

from autogameplayer.core.config import settings
from autogameplayer.core.config_loader import require_rom
from autogameplayer.utils.process import get_base_env, port_allocator


class WorkerPool:
    def __init__(self, num_workers: int, rom_path: str):
        self.num_workers = num_workers
        self.rom_path = rom_path
        self.processes: List[subprocess.Popen] = []
        self.logs: List = []

    def start(self):
        print(f"🚀 Launching WorkerPool with {self.num_workers} instances...")
        env = get_base_env()

        for i in range(self.num_workers):
            port = port_allocator.allocate(offset=100, worker_id=i)
            log_dir = settings.base_dir / "logs" / "workers"
            os.makedirs(log_dir, exist_ok=True)
            log_file = open(log_dir / f"worker_{i}.log", "w")
            self.logs.append(log_file)

            print(f"  - Worker {i} starting on port {port}...")
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "autogameplayer.server.mcp_server",
                    "--rom",
                    self.rom_path,
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.processes.append(p)

        print(f"⏳ All workers starting. Check {log_dir}/worker_*.log for details.")

    def stop(self):
        print("\n🛑 Shutting down WorkerPool...")
        from autogameplayer.utils.process import terminate_process

        for i, p in enumerate(self.processes):
            terminate_process(p, f"Worker {i}")
        for log_file in self.logs:
            log_file.close()
        print("✅ All processes terminated.")


def run_pool(num_workers: int = 3):
    rom_path = require_rom()

    pool = WorkerPool(num_workers, rom_path)
    pool.start()

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pool.stop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AutoGamePlayer WorkerPool")
    parser.add_argument("--workers", type=int, default=3, help="Number of workers")
    args = parser.parse_args()
    run_pool(num_workers=args.workers)


if __name__ == "__main__":
    main()
