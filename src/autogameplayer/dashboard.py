import asyncio
import json
import logging
import time
import sqlite3
from collections import deque

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from autogameplayer.core.config import settings
from autogameplayer.core.models import GameState
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_capabilities(port):
    """Sync helper to fetch core capabilities once."""
    url = f"http://{settings.server_host}:{port}/sse"
    async def _fetch():
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resp = await session.call_tool("get_capabilities")
                return json.loads(resp.content[0].text)
    try:
        return asyncio.run(_fetch())
    except Exception:
        return None

async def call_mcp_tool(port, tool_name, arguments=None):
    """Helper to call a single tool and return result."""
    url = f"http://{settings.server_host}:{port}/sse"
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resp = await session.call_tool(tool_name, arguments or {})
                return resp.content[0].text
    except Exception as e:
        return json.dumps({"error": str(e)})

def render_spatial_grid(db_path, current_map_id, current_x, current_y, map_name="Current Area"):
    """Renders a 2D Heatmap of the explored SLAM grid with impassable scores."""
    if not db_path.exists():
        return None
        
    try:
        # Normalize map_id to int if possible for DB query
        try:
            db_map_id = int(current_map_id)
        except (ValueError, TypeError):
            db_map_id = current_map_id

        with sqlite3.connect(str(db_path), timeout=10) as conn:
            # Query x, y, impassable_score, is_warp.
            if db_map_id == -1 or db_map_id is None:
                return None
            
            query = "SELECT x, y, impassable_score, is_warp FROM explored_locations WHERE map_id = ?"
            df = pd.read_sql_query(query, conn, params=(db_map_id,))
            
            if df.empty:
                return None
                
            # Create a pivot table for the heatmap using the float score
            grid = df.pivot(index='y', columns='x', values='impassable_score').fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=grid.values,
                x=grid.columns,
                y=grid.index,
                colorscale=[[0, 'rgb(30,30,30)'], [0.5, 'rgb(255,165,0)'], [1, 'rgb(255,0,0)']],
                showscale=True,
                colorbar=dict(title="Blocked", tickvals=[0, 0.5, 1], ticktext=["0.0", "0.5", "1.0"]),
                xgap=1, ygap=1,
                zmin=0, zmax=1,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Score: %{z}<extra></extra>"
            ))

            # Overlay Warp Tiles (Purple)
            warps = df[df['is_warp'] == 1]
            if not warps.empty:
                fig.add_trace(go.Scatter(
                    x=warps['x'], y=warps['y'],
                    mode='markers',
                    marker=dict(size=10, color='purple', symbol='diamond', line=dict(width=1, color='white')),
                    name='Warp Point',
                    hovertemplate="Warp Point at (%{x}, %{y})<extra></extra>"
                ))
            
            # Overlay Current Player Position
            if current_x >= 0 and current_y >= 0:
                fig.add_trace(go.Scatter(
                    x=[current_x], y=[current_y],
                    mode='markers',
                    marker=dict(size=14, color='white', symbol='x', line=dict(width=2, color='black')),
                    name='Player',
                    hovertemplate="AI Player at (%{x}, %{y})<extra></extra>"
                ))
            
            # Ensure the plot range covers at least a few tiles even if only 1 visited
            x_min, x_max = min(grid.columns), max(grid.columns)
            y_min, y_max = min(grid.index), max(grid.index)
            
            if x_min == x_max:
                x_min -= 2
                x_max += 2
            if y_min == y_max:
                y_min -= 2
                y_max += 2

            fig.update_layout(
                title=f"🗺️ {map_name} (Map #{current_map_id})",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10),
                height=450,
                xaxis=dict(showgrid=False, zeroline=False, title="X Tile", range=[x_min-0.5, x_max+0.5]),
                yaxis=dict(showgrid=False, zeroline=False, autorange='reversed', scaleanchor="x", scaleratio=1, title="Y Tile", range=[y_max+0.5, y_min-0.5]),
                showlegend=False
            )
            return fig
    except Exception as e:
        logger.error(f"Error rendering grid for map {current_map_id}: {e}")
        return None

def main():
    st.set_page_config(page_title="AGP Nexus Dashboard", layout="wide")
    display_map_id = 0
    
    if 'supported_buttons' not in st.session_state:
        st.session_state.supported_buttons = ["up", "down", "left", "right", "a", "b", "start", "select"]
    
    if 'reasoning_steps' not in st.session_state:
        st.session_state.reasoning_steps = deque(maxlen=5)
    
    if 'last_coords' not in st.session_state:
        st.session_state.last_coords = (0, 0)
    
    if 'collision_alert' not in st.session_state:
        st.session_state.collision_alert = None

    st.sidebar.title("🎮 AGP Nexus")
    worker_port = st.sidebar.number_input("Worker Port", min_value=8000, max_value=8100, value=settings.server_port, key="sidebar_port_input")
    if st.sidebar.button("🔄 Sync Capabilities", key="sidebar_sync_btn"):
        caps = fetch_capabilities(worker_port)
        if caps:
            st.session_state.supported_buttons = caps.get('supported_buttons', [])
            st.rerun()
            
    if st.sidebar.button("🧹 Clear UI Cache"):
        st.session_state.clear()
        st.rerun()

    include_ocr = st.sidebar.checkbox("Enable OCR (Slow Path)", value=False, key="sidebar_ocr_toggle")
    follow_player = st.sidebar.checkbox("Follow Player", value=True, key="sidebar_follow_player")

    st.sidebar.header("🕹️ Controller")
    btns = st.session_state.supported_buttons
    dirs = [b for b in btns if b in ["up", "down", "left", "right"]]
    actions = [b for b in btns if b not in dirs]

    c1, c2, c3 = st.sidebar.columns(3)
    with c2: 
        if "up" in dirs and st.button("⬆️", key="btn_up"): 
            asyncio.run(call_mcp_tool(worker_port, "send_input", {"button": "up"}))
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if "left" in dirs and st.button("⬅️", key="btn_left"):
            asyncio.run(call_mcp_tool(worker_port, "send_input", {"button": "left"}))
    with c3:
        if "right" in dirs and st.button("➡️", key="btn_right"):
            asyncio.run(call_mcp_tool(worker_port, "send_input", {"button": "right"}))
    with c2:
        if "down" in dirs and st.button("⬇️", key="btn_down"):
            asyncio.run(call_mcp_tool(worker_port, "send_input", {"button": "down"}))

    st.sidebar.divider()
    cols = st.sidebar.columns(3)
    for i, btn in enumerate(actions):
        with cols[i % 3]:
            if st.button(f"[{btn.upper()}]", key=f"btn_{btn}"):
                asyncio.run(call_mcp_tool(worker_port, "send_input", {"button": btn}))

    st.sidebar.header("💬 AI Guidance")
    user_msg = st.sidebar.text_input("Instruction", placeholder="e.g. Go to the PokeCenter", key="sidebar_guidance_input")
    if st.sidebar.button("Send Guidance", key="sidebar_send_guidance_btn"):
        asyncio.run(call_mcp_tool(worker_port, "set_guidance", {"message": user_msg}))

    st.header("📺 Universal Game Nexus")
    
    tab_live, tab_kb = st.tabs(["🎮 Live Session", "🧠 Knowledge Base"])

    with tab_live:
        col_main, col_stats = st.columns([2, 1])
        
        with col_main:
            # --- VIDEO FEED ---
            ws_port = worker_port + 100
            st.components.v1.html(f"""
                <div id="game-container" style="background: #000; padding: 10px; border-radius: 8px; text-align: center; box-sizing: border-box;">
                    <canvas id="game-canvas" width="240" height="160" style="width: 100%; height: auto; max-width: 800px; max-height: 650px; image-rendering: pixelated; border: 2px solid #333; display: block; margin: auto;"></canvas>
                </div>
                <script>
                    const canvas = document.getElementById('game-canvas');
                    const ctx = canvas.getContext('2d');
                    ctx.imageSmoothingEnabled = false;
                    function connect() {{
                        const ws = new WebSocket('ws://127.0.0.1:{ws_port}');
                        ws.onmessage = (e) => {{
                            const data = JSON.parse(e.data);
                            if (data.type === 'frame') {{
                                if (data.width && data.height && (canvas.width !== data.width || canvas.height !== data.height)) {{
                                    canvas.width = data.width;
                                    canvas.height = data.height;
                                    ctx.imageSmoothingEnabled = false;
                                }}
                                const img = new Image();
                                img.onload = () => ctx.drawImage(img, 0, 0);
                                img.src = 'data:image/jpeg;base64,' + data.data;
                            }}
                        }};
                        ws.onclose = () => setTimeout(connect, 2000);
                    }}
                    connect();
                </script>
            """, height=600)
            
            # --- SPATIAL MEMORY ---
            st.divider()
            st.subheader("🗺️ Spatial Memory (SLAM)")
            grid_chart = st.empty()
            
            st.subheader("📊 Vision Latent Space")
            vector_chart = st.empty()

        with col_stats:
            # --- BRAIN STREAM ---
            st.subheader("🧠 Brain Stream")
            action_info = st.empty()
            st.write("**Recent Reasoning Steps**")
            reasoning_history_table = st.empty()
            
            st.divider()
            st.subheader("👁️ AI Perception")
            ocr_box = st.empty()
            st.subheader("📍 Live Status")
            status_box = st.empty()
            
            st.divider()
            st.subheader("📚 Knowledge Retrieval (RAG)")
            rag_box = st.empty()
            
            st.divider()
            st.subheader("🚗 Motion Delta (Odometer)")
            m_col1, m_col2 = st.columns(2)
            dx_metric = m_col1.empty()
            dy_metric = m_col2.empty()
            collision_box = st.empty()

    with tab_kb:
        st.header("Pluggable Knowledge Base")
        db_path = settings.models_dir / "long_term_memory.db"
        
        # --- ROW 1: DISCOVERY & STATS ---
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            st.subheader("📍 Discovered RAM Map")
            ram_path = settings.models_dir / "discovered_ram.json"
            if ram_path.exists():
                with open(ram_path, "r") as f:
                    st.json(json.load(f))
            else:
                st.info("No RAM addresses discovered yet.")

        with k_col2:
            st.subheader("🌍 Global Progress")
            if db_path.exists():
                try:
                    with sqlite3.connect(str(db_path), timeout=10) as conn:
                        # 1. Total Tiles Explored
                        cursor = conn.execute("SELECT COUNT(*) FROM explored_locations")
                        total_tiles = cursor.fetchone()[0]
                        st.metric("Total Unique Tiles Explored", total_tiles)
                        
                        # 2. Frequent Stuck Points
                        st.write("**Top Stuck Points (by vision hash)**")
                        cursor = conn.execute("SELECT vision_hash, count FROM stuck_hashes ORDER BY count DESC LIMIT 3")
                        stuck_rows = cursor.fetchall()
                        if stuck_rows:
                            st.table([{"Hash": r[0], "Seen Count": r[1]} for r in stuck_rows])
                        else:
                            st.info("No persistent stuck points recorded.")
                except Exception as e:
                    st.error(f"Error reading stats: {e}")
            else:
                st.info("No database found for stats.")

        st.divider()

        # --- ROW 2: SKILLS ---
        st.subheader("✨ Visual Skill Registry (Macros)")
        if db_path.exists():
            try:
                with sqlite3.connect(str(db_path), timeout=10) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("SELECT name, description, macro_json, score, reliability, times_run, is_hierarchical FROM skills ORDER BY (reliability * score) DESC")
                    rows = cursor.fetchall()
                    
                    if rows:
                        display_reg = []
                        for r in rows:
                            display_reg.append({
                                "Name": r["name"] or "Pending...",
                                "Type": "🧩 Hierarchical" if r["is_hierarchical"] else "⚡ Flat",
                                "Description": r["description"],
                                "Macro": r["macro_json"],
                                "Reliability": f"{r['reliability']:.2f}",
                                "Runs": r["times_run"],
                                "Score": f"{r['score']:.2f}"
                            })
                        st.table(display_reg)
                    else:
                        st.info("No skills distilled into the database yet.")
            except Exception as e:
                st.error(f"Error reading skills: {e}")

        st.divider()

        # --- ROW 3: RECENT EXPERIENCES ---
        st.subheader("🎞️ Recent Replays")
        if db_path.exists():
            try:
                with sqlite3.connect(str(db_path), timeout=10) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("SELECT step_index, map_id, coords, button, reward, reasoning FROM replay_buffer ORDER BY id DESC LIMIT 10")
                    replays = cursor.fetchall()
                    if replays:
                        st.table([dict(r) for r in replays])
                    else:
                        st.info("Replay buffer is empty.")
            except Exception as e:
                st.error(f"Error reading replays: {e}")
        else:
            st.info("Database not initialized yet.")

    # Fetch initial map list for sidebar selection
    map_list = {0: "Intro/Naming"}
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path), timeout=10) as conn:
                cursor = conn.execute("SELECT DISTINCT map_id FROM explored_locations")
                for row in cursor:
                    mid = row[0]
                    if mid not in map_list:
                        map_list[mid] = f"Map #{mid}"
        except Exception:
            pass

    st.sidebar.divider()
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path), timeout=10) as conn:
                # Basic diagnostic: show tile count for player's current map
                # (We don't have display_map_id here yet, so we'll skip the sidebar metric for now or make it global)
                cursor = conn.execute("SELECT COUNT(*) FROM explored_locations")
                total_tiles = cursor.fetchone()[0]
                st.sidebar.write(f"🗺️ Total Tiles in DB: {total_tiles}")
        except Exception:
            pass

    view_map_id = st.sidebar.selectbox("🗺️ Select Grid View", options=list(map_list.keys()), format_func=lambda x: map_list[x], key="sidebar_map_selector")

    # --- FRAGMENTED UPDATE LOOP ---
    @st.fragment(run_every=0.5)
    def update_pass():
        # Re-fetch map list to see new discoveries (Only occasionally or inside fragment)
        try:
            with sqlite3.connect(str(db_path), timeout=10) as conn:
                cursor = conn.execute("SELECT DISTINCT map_id FROM explored_locations")
                for row in cursor:
                    mid = row[0]
                    if mid not in map_list:
                        map_list[mid] = f"Map #{mid}"
        except Exception: pass

        try:
            state_json = asyncio.run(call_mcp_tool(worker_port, "get_game_state", {"include_ocr": include_ocr}))
            
            if "error" in state_json and len(state_json) < 500:
                try:
                    err_data = json.loads(state_json)
                    if "error" in err_data:
                        status_box.warning(f"Connecting to Game Server... ({err_data['error'][:30]})")
                        return
                except Exception: return

            state_data = GameState.model_validate_json(state_json)

            # --- UPDATE UI ELEMENTS ---
            if state_data.last_action:
                plan_text = f"🎯 **Plan:** {state_data.current_plan}\n\n" if state_data.current_plan else ""
                repeat_suffix = f" (x{state_data.context.get('last_repeat', 1)})" if state_data.context.get('last_repeat', 1) > 1 else ""
                reasoning = state_data.last_reasoning or "Thinking..."
                action_info.info(f"{plan_text}🤖 **Action:** {state_data.last_action.upper()}{repeat_suffix}")

                # Update reasoning history
                current_entry = {"Action": state_data.last_action.upper(), "Reasoning": reasoning}
                if not st.session_state.reasoning_steps or st.session_state.reasoning_steps[-1] != current_entry:
                    st.session_state.reasoning_steps.append(current_entry)

                reasoning_history_table.table(list(st.session_state.reasoning_steps)[::-1])
            else:
                action_info.write("⏳ Waiting for AI to take action...")

            if state_data.context:
                ctx = state_data.context
                ui_state = ctx.get('interface_mode', 'UNKNOWN')
                curr_map_id = ctx.get('map_id', 0)
                
                try:
                    display_map_id = int(curr_map_id)
                except (ValueError, TypeError):
                    display_map_id = curr_map_id

                curr_x, curr_y = ctx.get('x', 0), ctx.get('y', 0)
                last_x, last_y = st.session_state.last_coords
                
                dx = curr_x - last_x
                dy = curr_y - last_y
                
                # Collision Detection (Odometer)
                last_btn = state_data.last_action.lower() if state_data.last_action else ""
                collision = False
                if last_btn == "up" and dy >= 0:
                    collision = True
                elif last_btn == "down" and dy <= 0:
                    collision = True
                elif last_btn == "left" and dx >= 0:
                    collision = True
                elif last_btn == "right" and dx <= 0:
                    collision = True
                
                if last_btn not in ["up", "down", "left", "right"]:
                    collision = False
                if ui_state != "EXPLORABLE (Overworld)":
                    collision = False
                
                dx_metric.metric("Delta X", dx, delta_color="normal" if not collision else "inverse")
                dy_metric.metric("Delta Y", dy, delta_color="normal" if not collision else "inverse")
                
                if collision:
                    collision_box.error(f"🚨 Collision at ({curr_x}, {curr_y})!")
                else:
                    collision_box.empty()
                
                st.session_state.last_coords = (curr_x, curr_y)
                
                s_text = f"**Map:** #{curr_map_id} | **X:** {curr_x} | **Y:** {curr_y}\n\n"
                s_text += f"**HP:** {ctx.get('hp')} | **State:** {ui_state}"
                status_box.write(s_text)

                if state_data.ocr_text:
                    ocr_box.info(f"**OCR:** {state_data.ocr_text}")
                else:
                    ocr_box.write("🗨️ No dialogue text detected.")

                # Update RAG Retrieval Box
                if state_data.recalled_memories:
                    rag_text = "**Top Recalled Context:**\n\n"
                    for mem in state_data.recalled_memories[:3]:
                        rag_text += f"- {mem[:200]}...\n"
                    rag_box.info(rag_text)
                else:
                    rag_box.write("📖 Waiting for active knowledge retrieval...")

                # Render SLAM Grid
                grid_map_id = display_map_id if follow_player else view_map_id
                map_name = map_list.get(grid_map_id, f"Map #{grid_map_id}")
                
                player_x = curr_x if grid_map_id == display_map_id else -100
                player_y = curr_y if grid_map_id == display_map_id else -100
                
                grid_fig = render_spatial_grid(db_path, grid_map_id, player_x, player_y, map_name=map_name)
                if grid_fig:
                    grid_chart.plotly_chart(grid_fig, use_container_width=True, key="live_exploration_heatmap")
                else:
                    grid_chart.info(f"No exploration data for {map_name}.")

            if state_data.vision_vector:
                vector_chart.line_chart(state_data.vision_vector)
            else:
                vector_chart.write("📊 Vision Encoder disabled or warming up...")        
        except Exception as e:
            status_box.warning(f"Searching for Game Server... ({str(e)[:50]})")

    # Start the fragment loop
    update_pass()

if __name__ == "__main__":
    main()
