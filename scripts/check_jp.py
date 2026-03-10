import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def check_japanese_addr():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- MENU OPEN ---
            print("Opening menu (START)...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            
            addr = await session.call_tool("read_memory", {"address": 0xCFAF})
            print(f"0xCFAF (Menu Open): {addr.content[0].text}")
            
            # --- MENU CLOSED ---
            print("Closing menu (B)...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(2.0)
            
            addr = await session.call_tool("read_memory", {"address": 0xCFAF})
            print(f"0xCFAF (Menu Closed): {addr.content[0].text}")

if __name__ == "__main__":
    asyncio.run(check_japanese_addr())
