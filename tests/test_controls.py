import asyncio
import numpy as np
import sys
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from autogameplayer.core.models import GameState

async def verify_system():
    url = "http://localhost:8000/sse"
    print("🧪 Verifying Control-to-Vision Bridge...")
    
    try:
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # Retry logic for visual change
                for attempt in range(3):
                    # 1. Baseline State
                    # Wait for a few frames to ensure we're not catching a black screen or logo
                    await asyncio.sleep(1.0)
                    response_0 = await session.call_tool("get_game_state", arguments={})
                    state_0 = GameState.model_validate_json(response_0.content[0].text)
                    v0 = np.array(state_0.vision_vector)
                    
                    # 2. Action
                    print(f"⌨️  Sending 'START' command (Attempt {attempt+1}/3)...")
                    # Send a long press
                    await session.call_tool("send_input", arguments={"button": "start", "duration": 30})
                    # Wait for frames to propagate
                    await asyncio.sleep(2.0) 
                    
                    # 3. Post-Action State
                    response_1 = await session.call_tool("get_game_state", arguments={})
                    state_1 = GameState.model_validate_json(response_1.content[0].text)
                    v1 = np.array(state_1.vision_vector)
                    
                    # 4. Measure Delta
                    delta = np.linalg.norm(v0 - v1)
                    if delta > 0.01: # Lowered threshold for DINOv2 small stability
                        print(f"✅ Success! Visual change detected (Delta: {delta:.4f})")
                        return True
                    
                    print(f"⚠️  No visual response yet (Delta: {delta:.4f}), retrying...")
                    # Force a tick just in case the server loop is slow
                    await session.call_tool("tick", {"frames": 60})

                print("❌ Failure: No visual response to input after 3 attempts.")
                return False

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_system())
    if not success:
        sys.exit(1)
