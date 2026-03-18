import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def identify_menu_id():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- CLOSED ---
            print("Snapshot (Closed)...")
            await session.call_tool("scan_memory", {"condition": "snapshot"})

            # --- START MENU ---
            print("Opening START menu...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            print("Scanning for changed...")
            await session.call_tool("scan_memory", {"condition": "changed"})

            # --- NEXT MENU (e.g. Items) ---
            print("Moving to next menu item and selecting...")
            await session.call_tool("send_input", {"button": "down"})
            await asyncio.sleep(0.5)
            await session.call_tool("send_input", {"button": "a"})
            await asyncio.sleep(1.0)

            print("Scanning for changed again (within menus)...")
            res = await session.call_tool("scan_memory", {"condition": "changed"})
            print(res.content[0].text)

            # --- CLOSE ALL ---
            print("Closing all...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(1.0)
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(1.0)

            print("Scanning for EXACT 0...")
            res = await session.call_tool(
                "scan_memory", {"condition": "exact", "exact_value": 0}
            )
            print(res.content[0].text)


if __name__ == "__main__":
    asyncio.run(identify_menu_id())
