import asyncio
import json
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def spam_a():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Spamming A to clear any dialogue...")
            for _ in range(20):
                await session.call_tool("send_input", {"button": "a"})
                await asyncio.sleep(0.2)

            await asyncio.sleep(1.0)
            res = await session.call_tool("get_game_state", {"include_ocr": False})
            state = json.loads(res.content[0].text)
            ctx = state.get("context", {})
            print(f"sprites_enabled (0xFFD6): {ctx.get('sprites_enabled')}")
            print(f"Status: {ctx.get('symbolic', {}).get('status', 'UNKNOWN')}")


if __name__ == "__main__":
    asyncio.run(spam_a())
