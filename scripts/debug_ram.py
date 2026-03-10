import asyncio
from autogameplayer.core.mcp_client import sse_client
from mcp.client.session import ClientSession

async def test_input_changes():
    url = "http://127.0.0.1:8000/sse"
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Take snapshot
            print("Taking snapshot...")
            await session.call_tool("scan_memory", {"condition": "snapshot"})
            
            # Send B
            print("Sending 'B'...")
            await session.call_tool("send_input", {"button": "b"})
            await asyncio.sleep(1.0)
            
            # Scan for changes
            print("Scanning for changes...")
            res = await session.call_tool("scan_memory", {"condition": "changed"})
            print(res.content[0].text)
            
            # Check the specific addresses again
            addresses = {
                "sprites_enabled (0xCFC4)": "0xCFC4",
                "menu_id_ram (0xCFCB)": "0xCFCB",
                "max_menu_item (0xCC28)": "0xCC28"
            }
            print("\n--- Current Values ---")
            for name, addr in addresses.items():
                val = await session.call_tool("read_memory", {"address": int(addr, 16)})
                print(f"{name}: {val.content[0].text}")

if __name__ == "__main__":
    asyncio.run(test_input_changes())
