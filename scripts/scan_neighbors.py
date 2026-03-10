import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def scan_cfcb_neighbors():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- START MENU ---
            print("Opening menu...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            res = await session.call_tool("scan_neighborhood", {"address": "0xCFCB", "radius": 5})
            print(f"Open: {res.content[0].text}")
            
            # --- CLOSE ---
            print("Closing menu...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(2.0)
            res = await session.call_tool("scan_neighborhood", {"address": "0xCFCB", "radius": 5})
            print(f"Closed: {res.content[0].text}")

if __name__ == "__main__":
    asyncio.run(scan_cfcb_neighbors())
