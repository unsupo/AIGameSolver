import os
import sys
import asyncio
from pathlib import Path
from autogameplayer.core.config import settings
from autogameplayer.core.config_loader import require_rom
from autogameplayer.runner.single import run as run_single
from autogameplayer.utils.llm import OllamaBootstrap


def main():
    # 1. Setup local environment defaults for turnkey
    os.environ["AGP_LLM_PROVIDER"] = "ollama"
    os.environ["AGP_LLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
    os.environ["AGP_LLM_API_KEY"] = "ollama"

    # 2. Bootstrap the LLM hardware
    if not OllamaBootstrap.bootstrap([settings.llm_model]):
        sys.exit(1)

    # 3. Determine the best LLM config
    rom_path = require_rom()

    # Dynamically find an LLM config
    config_path = None
    yaml_files = list(Path(settings.base_dir / "configs").glob("*llm.yaml"))
    if yaml_files:
        config_path = str(yaml_files[0])
        print(f"✅ Found LLM Config: {config_path}")

    # 4. Launch the full stack
    print("🚀 Launching Turnkey LLM Stack...")
    asyncio.run(run_single(rom_path=rom_path, config_path=config_path))


if __name__ == "__main__":
    main()
