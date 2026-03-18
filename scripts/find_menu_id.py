import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def find_menu_id():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Taking initial snapshot...")
            await session.call_tool("scan_memory", {"condition": "snapshot"})

            print("Opening menu (START)...")
            await session.call_tool("send_input", {"button": "start"})
            await asyncio.sleep(2.0)  # Wait for animation

            print("Scanning for increased values...")
            res = await session.call_tool("scan_memory", {"condition": "increased"})
            print(res.content[0].text)

            print("Taking new snapshot with menu open...")
            await session.call_tool("scan_memory", {"condition": "snapshot"})

            print("Closing menu (B)...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(2.0)

            print("Scanning for decreased values...")
            res = await session.call_tool("scan_memory", {"condition": "decreased"})
            print(res.content[0].text)


if __name__ == "__main__":
    asyncio.run(find_menu_id())
