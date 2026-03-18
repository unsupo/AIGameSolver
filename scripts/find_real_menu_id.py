import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def find_real_menu_id():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- START ---
            print("Taking snapshot...")
            await session.call_tool("scan_memory", {"condition": "snapshot"})

            # --- PRESS B ---
            print("Pressing B...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(1.0)

            print("Scanning for decreased...")
            res = await session.call_tool("scan_memory", {"condition": "decreased"})
            print(f"Decreased: {res.content[0].text}")

            print("Scanning for EXACT 0...")
            res = await session.call_tool(
                "scan_memory", {"condition": "exact", "exact_value": 0}
            )
            print(f"Exact 0: {res.content[0].text}")


if __name__ == "__main__":
    asyncio.run(find_real_menu_id())
