import asyncio
from datetime import timedelta
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from autogameplayer.core.memory import MemoryReader
from autogameplayer.core.config import settings


class MCPClient(MemoryReader):
    """Handles low-level SSE/MCP communication."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session: ClientSession = None
        self._sse_context = None
        self._next_seq = 0

    async def connect(self):
        retries = 5
        last_error = None
        for i in range(retries):
            try:
                # sse_client supports timeout (connect) and sse_read_timeout (long-polling)
                self._sse_context = sse_client(
                    self.server_url, sse_read_timeout=settings.mcp_timeout
                )
                streams = await self._sse_context.__aenter__()

                # ClientSession uses read_timeout_seconds (timedelta) for tool calls
                self.session = ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=timedelta(seconds=settings.mcp_timeout),
                )
                await self.session.__aenter__()
                await self.session.initialize()
                return
            except Exception as e:
                last_error = e
                # Clean up partially initialized state on failure
                await self.disconnect()

                if i < retries - 1:
                    print(
                        f"⚠️  MCP Connection attempt {i + 1} failed: {e}. Retrying in 2s..."
                    )
                    await asyncio.sleep(2)
                else:
                    break

        raise ConnectionError(
            f"Failed to connect to MCP Server after {retries} attempts: {last_error}"
        )

    async def disconnect(self):
        try:
            if self.session:
                # Use a timeout for session exit to avoid hanging on a dead server
                try:
                    await asyncio.wait_for(
                        self.session.__aexit__(None, None, None), timeout=1.0
                    )
                except Exception:
                    pass
            if self._sse_context:
                try:
                    await asyncio.wait_for(
                        self._sse_context.__aexit__(None, None, None), timeout=1.0
                    )
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self.session = None
            self._sse_context = None

    async def call_tool(self, name: str, arguments: dict = None):
        if not self.session:
            raise RuntimeError("MCP Client session is not connected.")
        
        args = arguments or {}
        # Inject sequence number for idempotency
        args["_seq"] = self._next_seq
        self._next_seq += 1

        try:
            response = await self.session.call_tool(name, args)
            text = response.content[0].text
            
            # Detect server-side Pydantic validation errors returned as strings
            if "validation error for" in text and ("unexpected_keyword_argument" in text or "missing" in text or "Field required" in text):
                print(f"❌ CRITICAL: Server returned Pydantic validation error for tool '{name}':")
                print(f"--- SERVER ERROR START ---\n{text}\n--- SERVER ERROR END ---")
                raise ValueError(f"Server-side validation failed for tool '{name}'.")
                
            return text
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
        except ValueError, TypeError:
            return 0
