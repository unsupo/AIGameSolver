import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def check_dff5():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- CLOSED ---
            val = await session.call_tool("read_memory", {"address": 0xDFF5})
            print(f"0xDFF5 (Closed): {val.content[0].text}")

            # --- OPEN START ---
            print("Opening START menu...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)
            val = await session.call_tool("read_memory", {"address": 0xDFF5})
            print(f"0xDFF5 (Start Menu): {val.content[0].text}")

            # --- OPEN ITEMS ---
            print("Opening ITEMS menu...")
            await session.call_tool("send_input", {"button": "down"})
            await asyncio.sleep(0.5)
            await session.call_tool("send_input", {"button": "a"})
            await asyncio.sleep(1.0)
            val = await session.call_tool("read_memory", {"address": 0xDFF5})
            print(f"0xDFF5 (Items Menu): {val.content[0].text}")

            # --- CLOSE ALL ---
            print("Closing all...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(1.0)
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(1.0)
            val = await session.call_tool("read_memory", {"address": 0xDFF5})
            print(f"0xDFF5 (Closed): {val.content[0].text}")


if __name__ == "__main__":
    asyncio.run(check_dff5())
