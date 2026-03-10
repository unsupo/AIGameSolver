import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def find_sprites_enabled():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- MENU OPEN STATE ---
            print("Opening menu (START)...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            
            print("Scanning for EXACT 0 (likely sprites disabled)...")
            res = await session.call_tool("scan_memory", {"condition": "snapshot"})
            res = await session.call_tool("scan_memory", {"condition": "exact", "exact_value": 0})
            print(res.content[0].text)
            
            # Read HRAM while menu is open
            h_open = await session.call_tool("read_memory", {"address": 0xFFD6})
            print(f"HRAM 0xFFD6 (Menu Open): {h_open.content[0].text}")
            
            # --- MENU CLOSED STATE ---
            print("Closing menu (B)...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(2.0)
            
            print("Scanning for EXACT 1 (likely sprites enabled)...")
            res = await session.call_tool("scan_memory", {"condition": "exact", "exact_value": 1})
            print(res.content[0].text)
            
            # Read HRAM while menu is closed
            h_closed = await session.call_tool("read_memory", {"address": 0xFFD6})
            print(f"HRAM 0xFFD6 (Menu Closed): {h_closed.content[0].text}")

if __name__ == "__main__":
    asyncio.run(find_sprites_enabled())
