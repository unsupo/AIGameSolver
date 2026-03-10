import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def check_menu_id():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- MENU OPEN ---
            print("Opening menu (START)...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            
            m_open = await session.call_tool("read_memory", {"address": 0xCFCB})
            h_open = await session.call_tool("read_memory", {"address": 0xFFD7})
            print(f"0xCFCB (Menu Open): {m_open.content[0].text}")
            print(f"0xFFD7 (Menu Open): {h_open.content[0].text}")
            
            # --- MENU CLOSED ---
            print("Closing menu (B)...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(2.0)
            
            m_closed = await session.call_tool("read_memory", {"address": 0xCFCB})
            h_closed = await session.call_tool("read_memory", {"address": 0xFFD7})
            print(f"0xCFCB (Menu Closed): {m_closed.content[0].text}")
            print(f"0xFFD7 (Menu Closed): {h_closed.content[0].text}")

if __name__ == "__main__":
    asyncio.run(check_menu_id())
