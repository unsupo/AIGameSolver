import asyncio
import json
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def check_ocr():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Fetching game state with OCR...")
            res = await session.call_tool("get_game_state", {"include_ocr": True})
            state = json.loads(res.content[0].text)
            ctx = state.get('context', {})
            print(f"OCR Text: {state.get('ocr_text', 'NONE')}")
            print(f"Interface Mode: {ctx.get('interface_mode', 'UNKNOWN')}")
            print(f"Symbolic Status: {ctx.get('symbolic', {}).get('status', 'UNKNOWN')}")
            print(f"Map ID: {ctx.get('map_id')}")

if __name__ == "__main__":
    asyncio.run(check_ocr())
