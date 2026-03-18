import asyncio
import json
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def debug_route_11():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Sending 'B' to clear any menu on Route 11...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(1.0)

            res = await session.call_tool("get_game_state", {"include_ocr": False})
            state = json.loads(res.content[0].text)
            ctx = state["context"]
            print(f"menu_id_ram (0xCFC4): {ctx.get('menu_id_ram')}")
            print(f"sprites_enabled (0xFFD6): {ctx.get('sprites_enabled')}")
            print(f"is_overworld_active: {ctx.get('is_overworld_active')}")
            print(f"Status: {state['symbolic']['status']}")


if __name__ == "__main__":
    asyncio.run(debug_route_11())
