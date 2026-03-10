import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def search_menu_id():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- CLOSED ---
            print("Resetting snapshot (Closed)...")
            await session.call_tool("scan_memory", {"condition": "snapshot"})
            
            # --- OPEN ---
            print("Opening menu (START)...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            
            print("Scanning for increased values...")
            res = await session.call_tool("scan_memory", {"condition": "increased"})
            print(res.content[0].text)
            
            # --- CLOSED ---
            print("Closing menu (B)...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(2.0)
            
            print("Scanning for decreased values...")
            res = await session.call_tool("scan_memory", {"condition": "decreased"})
            print(res.content[0].text)
            
            print("Scanning for EXACT 0...")
            res = await session.call_tool("scan_memory", {"condition": "exact", "exact_value": 0})
            print(res.content[0].text)

if __name__ == "__main__":
    asyncio.run(search_menu_id())
