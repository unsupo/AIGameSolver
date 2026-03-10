import asyncio
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from autogameplayer.core.memory import MemoryReader

class MCPClient(MemoryReader):
    """Handles low-level SSE/MCP communication."""
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session: ClientSession = None
        self._sse_context = None

    async def connect(self):
        self._sse_context = sse_client(self.server_url)
        streams = await self._sse_context.__aenter__()
        self.session = ClientSession(streams[0], streams[1])
        await self.session.__aenter__()
        await self.session.initialize()

    async def disconnect(self):
        try:
            if self.session:
                # Use a timeout for session exit to avoid hanging on a dead server
                try:
                    await asyncio.wait_for(self.session.__aexit__(None, None, None), timeout=1.0)
                except Exception: pass
            if self._sse_context:
                try:
                    await asyncio.wait_for(self._sse_context.__aexit__(None, None, None), timeout=1.0)
                except Exception: pass
        except Exception:
            pass
        finally:
            self.session = None
            self._sse_context = None

    async def call_tool(self, name: str, arguments: dict = None):
        if not self.session:
            raise RuntimeError("MCP Client session is not connected.")
        try:
            response = await self.session.call_tool(name, arguments or {})
            return response.content[0].text
        except Exception as e:
            # Re-raise with a more helpful message if it's a connection issue
            msg = str(e)
            if "peer closed connection" in msg or "Connection closed" in msg:
                raise ConnectionError(f"MCP Server closed connection: {e}")
            raise e

    async def read_byte(self, address: int) -> int:
        res = await self.call_tool("read_memory", {"address": address})
        try:
            return int(res)
        except (ValueError, TypeError):
            return 0
