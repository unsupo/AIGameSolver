import asyncio
import json
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession


async def check_state():
    url = "http://127.0.0.1:8000/sse"
    # Wait for server to be ready
    for _ in range(10):
        try:
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.call_tool(
                        "get_game_state", {"include_ocr": False}
                    )
                    state = json.loads(res.content[0].text)
                    print(json.dumps(state["context"], indent=2))
                    print(f"Status: {state['symbolic']['status']}")
                    return
        except Exception as e:
            print(f"Waiting for server... {e}")
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(check_state())
