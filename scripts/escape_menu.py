import asyncio
import json
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def escape_menu():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Sending 'B' multiple times...")
            for _ in range(5):
                await session.call_tool("send_input", {"button": "b"})
                await asyncio.sleep(0.5)
            
            res = await session.call_tool("get_game_state", {"include_ocr": False})
            state = json.loads(res.content[0].text)
            print(f"menu_id_ram (0xCFC4): {state['context'].get('menu_id_ram')}")
            print(f"sprites_enabled (0xFFD6): {state['context'].get('sprites_enabled')}")
            # print(f"Status: {state['symbolic']['status']}") # Fix potential KeyError

if __name__ == "__main__":
    asyncio.run(escape_menu())
