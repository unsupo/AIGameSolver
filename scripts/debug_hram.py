import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def debug_hram():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Read HRAM addresses
            hram_addresses = {
                "hUpdateSpritesEnabled (0xFFD6)": 0xFFD6,
                "hTextBoxFlags (0xFFD7)": 0xFFD7,
                "hVBlankFlags (0xFFD5)": 0xFFD5,
            }

            print("--- HRAM Snapshot ---")
            for name, addr in hram_addresses.items():
                val = await session.call_tool("read_memory", {"address": addr})
                print(f"{name}: {val.content[0].text}")


if __name__ == "__main__":
    asyncio.run(debug_hram())
