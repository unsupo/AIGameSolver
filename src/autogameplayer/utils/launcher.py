import subprocess
import sys
import os
import time
import httpx
import asyncio
from typing import Optional
from pathlib import Path
from autogameplayer.core.config import settings

from autogameplayer.utils.process import get_base_env

class ServerLauncher:
    """Centralized utility for launching and polling MCP server instances."""
    def __init__(self, rom_path: str, port: int, log_path: Optional[Path] = None):
        self.rom_path = rom_path
        self.port = port
        self.log_path = log_path or (settings.base_dir / "logs" / f"server_{port}.log")
        self.proc: Optional[subprocess.Popen] = None

    def start(self, env_vars: dict = None, config_path: str = None) -> subprocess.Popen:
        from autogameplayer.utils.process import kill_process_on_port
        env = env_vars or get_base_env()
        
        # 0. Cleanup any zombies using the port
        kill_process_on_port(self.port)
        
        os.makedirs(self.log_path.parent, exist_ok=True)
        self.log_file = open(self.log_path, "w")
        
        cmd = [sys.executable, "-m", "autogameplayer.server.mcp_server", "--rom", self.rom_path, "--port", str(self.port)]
        if config_path:
            cmd.extend(["--config", config_path])
            
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        return self.proc

    async def wait_until_healthy(self, timeout: int = 30) -> bool:
        # Check root instead of SSE stream to avoid hanging
        url = f"http://127.0.0.1:{self.port}/"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if the process is still alive
                if self.proc and self.proc.poll() is not None:
                    print("⚠️  Server process died early. Check logs/server.log")
                    return False

                async with httpx.AsyncClient() as client:
                    # Root / usually returns 404 in FastMCP if not mapped, but indicates the server is up
                    resp = await client.get(url, timeout=1.0)
                    if resp.status_code in [200, 404]:
                        return True
            except Exception:
                await asyncio.sleep(1)
        return False

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if hasattr(self, 'log_file'):
            self.log_file.close()
