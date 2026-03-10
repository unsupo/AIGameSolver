import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def check_cfc4():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- CLOSED ---
            val = await session.call_tool("read_memory", {"address": 0xCFC4})
            print(f"0xCFC4 (Closed): {val.content[0].text}")
            
            # --- OPEN ---
            print("Opening menu (START)...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            val = await session.call_tool("read_memory", {"address": 0xCFC4})
            print(f"0xCFC4 (Open): {val.content[0].text}")
            
            # --- CLOSED ---
            print("Closing menu (B)...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(2.0)
            val = await session.call_tool("read_memory", {"address": 0xCFC4})
            print(f"0xCFC4 (Closed): {val.content[0].text}")

if __name__ == "__main__":
    asyncio.run(check_cfc4())
