import asyncio
import json
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def poke_test():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Poking menu_id_ram (0xCFC4) to 0...")
            await session.call_tool("poke_memory", {"address": "0xCFC4", "value": 0})
            print("Poking sprites_enabled (0xFFD6) to 1...")
            await session.call_tool("poke_memory", {"address": "0xFFD6", "value": 1})
            await asyncio.sleep(0.5)

            res = await session.call_tool("get_game_state", {"include_ocr": False})
            state = json.loads(res.content[0].text)
            ctx = state["context"]
            print(f"menu_id_ram: {ctx.get('menu_id_ram')}")
            print(f"sprites_enabled: {ctx.get('sprites_enabled')}")
            print(f"is_overworld_active: {ctx.get('is_overworld_active')}")


if __name__ == "__main__":
    asyncio.run(poke_test())
