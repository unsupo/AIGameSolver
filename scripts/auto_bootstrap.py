import asyncio
from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.registry import Registry
from autogameplayer.core.config import settings
from autogameplayer.utils.launcher import ServerLauncher
from autogameplayer.utils.process import get_base_env

# Trigger registrations


async def bootstrap():
    print("🚀 Auto-Bootstrapping Pokémon Red intro...")

    rom_path = settings.roms_dir / "Pokemon - Red Version.gb"
    if not rom_path.exists():
        print(f"❌ Error: ROM not found at {rom_path}")
        return

    # 1. Start a temporary server using Launcher
    port = 8999
    launcher = ServerLauncher(str(rom_path), port)
    launcher.start(env_vars=get_base_env())

    print("⏳ Waiting for server to initialize...")
    if not await launcher.wait_until_healthy():
        print(f"❌ Error: Bootstrap server failed to start on port {port}")
        launcher.stop()
        return

    client = MCPClient(f"http://localhost:{port}/sse")
    await client.connect()

    controller = Registry.create_controller("gb")
    brain = Registry.create_brain("intro", controller)

    print("Mashing through intro (looking for Pallet Town)...")

    try:
        for i in range(1000):  # Max 1000 steps
            # Check Map ID (0xD35E)
            map_id_str = await client.call_tool("read_memory", {"address": 0xD35E})
            map_id = int(map_id_str)

            if map_id > 0:
                print(f"✅ Reached Game World! (Map ID: {map_id})")
                print(f"💾 Saving bootstrap state to Slot {settings.bootstrap_slot}...")
                await client.call_tool(
                    "manage_checkpoint",
                    {"action": "save", "slot": settings.bootstrap_slot},
                )
                print("✨ Bootstrap Complete.")
                break

            # Keep mashing
            action = brain.act(None)
            await client.call_tool(
                "send_input", {"button": action.button, "duration": 5}
            )

            if i % 20 == 0:
                print(f"  Step {i}...")

    finally:
        await client.disconnect()
        launcher.stop()


if __name__ == "__main__":
    try:
        asyncio.run(bootstrap())
    except KeyboardInterrupt:
        pass
