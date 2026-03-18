import pytest
from autogameplayer.core.mcp_client import MCPClient
from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.solvers.composite import RandomSolver
from autogameplayer.core.context import AgentContext
from autogameplayer.core.orchestrator import SessionOrchestrator
from autogameplayer.core.config import settings


@pytest.mark.asyncio
async def test_headless_smoke_run():
    """
    Smoke test: Connect to a running MCP server and execute 10 steps.
    Expects a server to be running on settings.server_port.
    """
    server_url = f"http://{settings.server_host}:{settings.server_port}/sse"
    client = MCPClient(server_url)
    
    try:
        await client.connect()
    except Exception as e:
        pytest.skip(f"MCP Server not reachable: {e}")

    # 1. Setup Env
    env = EmulatorEnvironment(client)
    
    # 2. Setup Solver
    buttons = ["up", "down", "left", "right", "a", "b", "start", "select"]
    solver = RandomSolver(supported_buttons=buttons)
    
    # 3. Setup Context
    context = AgentContext(game_id="smoke_test")
    
    # 4. Run Orchestrator for 10 steps
    orchestrator = SessionOrchestrator(env, solver, context, render_delay=0)
    
    total_reward = await orchestrator.run(steps=10)
    
    assert context.step_index >= 10
    assert isinstance(total_reward, float)
    
    await client.disconnect()
