import asyncio
import json
import logging
import sqlite3
from collections import deque

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from autogameplayer.core.config import settings
from autogameplayer.core.models import GameState
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

def render_spatial_map(db_path, current_map_id):
    """Renders a Trajectory and Collision Overlay using plotly.express."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            # 1. Fetch Anchors (Trajectory)
            df_anchors = pd.read_sql_query(
                "SELECT x, y, timestamp, description FROM spatial_anchors WHERE map_id = ? ORDER BY timestamp ASC",
                conn, params=(current_map_id,)
            )
            
            # 2. Fetch Collisions from explored_locations
            df_collisions = pd.read_sql_query(
                "SELECT x, y, impassable_score FROM explored_locations WHERE map_id = ? AND impassable_score >= 0.5",
                conn, params=(current_map_id,)
            )

            if df_anchors.empty and df_collisions.empty:
                return None

            fig = go.Figure()

            # Layer 1: Path Trace (Lines between anchors)
            if len(df_anchors) > 1:
                fig.add_trace(go.Scatter(
                    x=df_anchors["x"], y=df_anchors["y"],
                    mode='lines+markers',
                    line=dict(color='cyan', width=1),
                    marker=dict(size=4, color='cyan'),
                    name="Path Trace",
                    hoverinfo='skip'
                ))

            # Layer 2: Anchors (Yellow Stars)
            if not df_anchors.empty:
                fig.add_trace(go.Scatter(
                    x=df_anchors["x"], y=df_anchors["y"],
                    mode='markers',
                    marker=dict(symbol='star', color='yellow', size=10),
                    text=df_anchors["description"],
                    name="Discovery Anchor",
                    hovertemplate="%{text}<extra></extra>"
                ))
            
            # Layer 3: Collision Heatmap (Red intensity)
            if not df_collisions.empty:
                fig.add_trace(go.Scatter(
                    x=df_collisions["x"],
                    y=df_collisions["y"],
                    mode='markers',
                    marker=dict(
                        symbol='x', 
                        color='red', 
                        size=8,
                        opacity=0.6
                    ),
                    name="Physical Blocker",
                    hovertemplate="Collision at (%{x}, %{y})<extra></extra>"
                ))
            
            fig.update_layout(
                title=f"🗺️ SLAM Frontier & Collision Zones (Map #{current_map_id})",
                xaxis=dict(autorange=True, scaleanchor="y", scaleratio=1),
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False
            )
            return fig
    except Exception as e:
        logger.error(f"Error rendering spatial map: {e}")
        return None


def render_reward_attribution(db_path):
    """Renders a pie chart showing which solvers are responsible for total reward."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            df = pd.read_sql_query(
                "SELECT solver_name, reward FROM replay_buffer WHERE reward > 0", 
                conn
            )
            if df.empty:
                return None
            
            # Aggregate rewards by solver
            attribution = df.groupby("solver_name")["reward"].sum()
            
            fig = go.Figure(data=[go.Pie(
                labels=attribution.index, 
                values=attribution.values,
                hole=.3,
                textinfo='label+percent'
            )])
            
            fig.update_layout(
                title="🎯 Reward Attribution by Solver",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False
            )
            return fig
    except Exception:
        return None


def render_latent_space(db_path):
    """Renders a PCA projection of recent hidden states."""
    if not db_path.exists():
        return None
    try:
        from sklearn.decomposition import PCA
        import numpy as np
        
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT hidden_state, reward FROM replay_buffer WHERE hidden_state IS NOT NULL ORDER BY id DESC LIMIT 200", 
                conn
            )
            
            if len(df) < 10:
                return None
                
            latents = []
            for b in df["hidden_state"]:
                latents.append(np.frombuffer(b, dtype=np.float32))
            
            X = np.stack(latents)
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            
            fig = go.Figure(data=go.Scatter(
                x=X_2d[:, 0], y=X_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df["reward"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Reward")
                ),
                text=[f"Reward: {r:.2f}" for r in df["reward"]],
                name="Hidden State"
            ))
            
            fig.update_layout(
                title="🧠 LSTM Hidden State Attractor (PCA)",
                xaxis_title="PC1",
                yaxis_title="PC2",
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            return fig
    except Exception:
        return None


def render_rnd_tracking(db_path):
    """Renders curiosity metrics: RND prediction error and intrinsic reward."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            df = pd.read_sql_query(
                "SELECT rnd_error, intrinsic_reward, timestamp FROM replay_buffer ORDER BY timestamp DESC LIMIT 200",
                conn
            )
            if df.empty:
                return None
            
            # Sort by timestamp for time-series display
            df = df.iloc[::-1]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["rnd_error"],
                mode='lines', name="RND Error (Lifelong)",
                line=dict(color='orange', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["intrinsic_reward"],
                mode='lines', name="Intrinsic Reward (Final)",
                line=dict(color='cyan', width=1, dash='dot')
            ))
            
            fig.update_layout(
                title="💡 Curiosity Trace (RND & Novelty)",
                xaxis_title="Time",
                yaxis_title="Value",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return fig
    except Exception:
        return None


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


def render_transition_graph(db_path):
    """Renders a simple bar chart of map transition frequency."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            query = """
                SELECT metadata FROM memories 
                WHERE type = 'warp' 
                ORDER BY timestamp DESC LIMIT 50
            """
            cursor = conn.execute(query)
            transitions = []
            for row in cursor:
                meta = json.loads(row[0])
                transitions.append(f"{meta['from_map']} -> {meta['to_map']}")
            
            if not transitions:
                return None
            
            counts = pd.Series(transitions).value_counts()
            fig = go.Figure(data=[go.Bar(x=counts.index, y=counts.values, marker_color='rgb(158,202,225)')])
            fig.update_layout(title="🌍 Map Transition Frequency", height=300, margin=dict(l=10, r=10, t=40, b=10))
            return fig
    except Exception:
        return None

def render_spatial_grid(
    db_path, current_map_id, current_x, current_y, map_name="Current Area"
):
    """Renders a 2D Heatmap of the explored SLAM grid."""
    if not db_path.exists():
        return None

    try:
        db_map_id = int(current_map_id)
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            query = "SELECT x, y, impassable_score, is_warp, visit_count FROM explored_locations WHERE map_id = ?"
            df = pd.read_sql_query(query, conn, params=(db_map_id,))

            if df.empty:
                return None

            grid = df.pivot(index="y", columns="x", values="visit_count").fillna(0)

            fig = go.Figure(
                data=go.Heatmap(
                    z=grid.values,
                    x=grid.columns,
                    y=grid.index,
                    colorscale=[[0, "rgb(20,20,20)"], [1, "rgb(0,100,255)"]],
                    showscale=False,
                    xgap=1,
                    ygap=1,
                    hovertemplate="X: %{x}<br>Y: %{y}<br>Visits: %{z}<extra></extra>",
                )
            )

            collisions = df[df["impassable_score"] >= 0.5]
            if not collisions.empty:
                fig.add_trace(
                    go.Scatter(
                        x=collisions["x"],
                        y=collisions["y"],
                        mode="markers",
                        marker=dict(size=8, color="red", symbol="x"),
                        name="Collision",
                    )
                )

            warps = df[df["is_warp"] == 1]
            if not warps.empty:
                fig.add_trace(
                    go.Scatter(
                        x=warps["x"],
                        y=warps["y"],
                        mode="markers",
                        marker=dict(size=10, color="purple", symbol="diamond"),
                        name="Warp",
                    )
                )

            if current_x >= 0 and current_y >= 0:
                fig.add_trace(
                    go.Scatter(
                        x=[current_x],
                        y=[current_y],
                        mode="markers",
                        marker=dict(size=14, color="white", symbol="cross"),
                        name="Player",
                    )
                )

            fig.update_layout(
                title=f"🗺️ SLAM View: {map_name} (Map #{current_map_id})",
                height=500,
                xaxis=dict(autorange=True, scaleanchor="y", scaleratio=1),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            return fig
    except Exception as e:
        logger.error(f"Grid error: {e}")
        return None


def main():
    st.set_page_config(page_title="AGP Nexus Dashboard", layout="wide")

    if "supported_buttons" not in st.session_state:
        st.session_state.supported_buttons = ["up", "down", "left", "right", "a", "b", "start", "select"]

    if "reasoning_steps" not in st.session_state:
        st.session_state.reasoning_steps = deque(maxlen=5)

    if "last_coords" not in st.session_state:
        st.session_state.last_coords = (0, 0)

    if "collision_history" not in st.session_state:
        st.session_state.collision_history = deque(maxlen=60)

    st.sidebar.title("🎮 AGP Nexus")
    worker_port = st.sidebar.number_input(
        "Worker Port",
        min_value=8000,
        max_value=8100,
        value=settings.server_port,
        key="sidebar_port_input",
    )

    if st.sidebar.button("🔄 Sync Capabilities"):
        caps = fetch_capabilities(worker_port)
        if caps:
            st.session_state.supported_buttons = caps.get("supported_buttons", [])
            st.rerun()

    include_ocr = st.sidebar.checkbox("Enable OCR (Slow Path)", value=False)
    st.sidebar.checkbox("Follow Player", value=True)

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

    st.header("📺 Universal Game Nexus")

    tab_live, tab_kb, tab_bandit, tab_curriculum = st.tabs(["🎮 Live Session", "🧠 Knowledge Base", "📈 Agent57 Bandit", "📚 Curriculum"])

    with tab_curriculum:
        st.header("Adaptive Curriculum Progress")
        st.info("Curriculum tracking active. Stage transitions are logged to the console.")
        
        stage = st.session_state.get("curriculum_stage", 0)
        st.metric("Current Stage", stage)

    with tab_bandit:
        st.header("Agent57 Meta-Controller (Bandit)")
        bandit_db = settings.models_dir / "bandit_stats.db"
        if bandit_db.exists():
            try:
                with sqlite3.connect(str(bandit_db)) as conn:
                    df_bandit = pd.read_sql_query("SELECT arm_id, reward, timestamp FROM bandit_stats ORDER BY timestamp DESC LIMIT 500", conn)
                    if not df_bandit.empty:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Arm Selection Frequency")
                            st.bar_chart(df_bandit["arm_id"].value_counts().sort_index())
                        with col2:
                            st.subheader("Avg Reward per Arm")
                            st.bar_chart(df_bandit.groupby("arm_id")["reward"].mean().sort_index())
                        st.subheader("Recent Arm Performance Trace")
                        df_bandit["timestamp"] = pd.to_datetime(df_bandit["timestamp"])
                        st.line_chart(df_bandit.set_index("timestamp")["reward"])
                    else:
                        st.info("No bandit history recorded yet.")
            except Exception as e:
                st.error(f"Error reading bandit stats: {e}")
        else:
            st.info("📉 Bandit database not found. This tab will populate once the agent starts completing training episodes.")

    with tab_live:
        col_main, col_stats = st.columns([2, 1])

        with col_main:
            # --- VIDEO FEED ---
            ws_port = worker_port + 100
            st.components.v1.html(
                f"""
                <style>
                    body {{ margin: 0; padding: 0; background: #000; overflow: hidden; display: flex; justify-content: center; align-items: center; height: 100vh; width: 100vw; }}
                    #game-container {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        width: 100%;
                        height: 100%;
                    }}
                    #game-canvas {{
                        width: 100%;
                        height: 100%;
                        max-width: 100vw;
                        max-height: 100vh;
                        object-fit: contain;
                        image-rendering: pixelated;
                        image-rendering: crisp-edges;
                        border: 2px solid #333;
                        border-radius: 4px;
                    }}
                </style>
                <div id="game-container">
                    <canvas id="game-canvas"></canvas>
                </div>
                <script>
                    const canvas = document.getElementById('game-canvas');
                    const ctx = canvas.getContext('2d');
                    function connect() {{
                        const ws = new WebSocket('ws://127.0.0.1:{ws_port}');
                        ws.onmessage = (e) => {{
                            const data = JSON.parse(e.data);
                            const img = new Image();
                            img.onload = () => {{
                                if (canvas.width !== img.width || canvas.height !== img.height) {{
                                    canvas.width = img.width;
                                    canvas.height = img.height;
                                }}
                                ctx.drawImage(img, 0, 0);
                            }};
                            img.src = 'data:image/jpeg;base64,' + data.data;
                        }};
                        ws.onclose = () => setTimeout(connect, 2000);
                    }}
                    connect();
                </script>
            """,
                height=650,
            )

            # --- SPATIAL MEMORY ---
            st.divider()
            t_col1, t_col2 = st.columns([1, 2])
            with t_col1:
                st.subheader("🌍 Map Transitions")
                transition_chart = st.empty()
            with t_col2:
                st.subheader("🗺️ SLAM View")
                grid_chart = st.empty()
            
            st.subheader("📍 SLAM Frontier & Collision Zones")
            frontier_chart = st.empty()

            st.subheader("📊 Model Performance")
            st.subheader("🧠 Latent Space Analysis (PCA)")
            latent_chart = st.empty()

            st.subheader("🎯 Reward Attribution")
            attribution_chart = st.empty()

            st.subheader("💡 Curiosity Tracking")
            rnd_chart = st.empty()

        with col_stats:
            # --- BRAIN STREAM ---
            st.subheader("🧠 Brain Stream")
            action_info = st.empty()

            st.divider()
            st.subheader("📝 Long-Horizon Reasoning")
            reasoning_log = st.empty()

            st.divider()
            st.subheader("👁️ Perception & Status")
            ocr_box = st.empty()
            status_box = st.empty()

            st.divider()
            st.subheader("🚗 Motion Delta (Odometer)")
            m_col1, m_col2 = st.columns(2)
            dx_metric = m_col1.empty()
            dy_metric = m_col2.empty()

    with tab_kb:
        st.header("Knowledge Base")
        db_path = settings.models_dir / "long_term_memory.db"
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            st.subheader("📍 Discovered RAM Map")
            ram_path = settings.models_dir / "discovered_ram.json"
            if ram_path.exists():
                with open(ram_path, "r") as f:
                    st.json(json.load(f))
        with k_col2:
            st.subheader("🌍 Global Progress")
            if db_path.exists():
                try:
                    with sqlite3.connect(str(db_path)) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM explored_locations")
                        total_tiles = cursor.fetchone()[0]
                        st.metric("Total Unique Tiles Explored", total_tiles)
                        
                        st.divider()
                        st.subheader("📦 Replay Buffer Health")
                        cursor = conn.execute("SELECT COUNT(*), AVG(reward), AVG(priority) FROM replay_buffer")
                        count, avg_reward, avg_priority = cursor.fetchone()
                        
                        st.metric("Total Steps in Buffer", count)
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Avg Reward", f"{avg_reward:.4f}")
                        c2.metric("Avg PER Priority", f"{avg_priority:.2f}")
                        
                        # Reward Distribution
                        df_rewards = pd.read_sql_query("SELECT reward FROM replay_buffer ORDER BY timestamp DESC LIMIT 1000", conn)
                        if not df_rewards.empty:
                            st.write("**Recent Reward Distribution**")
                            st.bar_chart(df_rewards["reward"].value_counts().sort_index())
                except Exception:
                    pass

    # --- FRAGMENTED UPDATE LOOP ---
    @st.fragment(run_every=0.5)
    def update_pass():
        try:
            state_json = asyncio.run(call_mcp_tool(worker_port, "get_game_state", {"include_ocr": include_ocr}))
            state_data = GameState.model_validate_json(state_json)

            if state_data.context:
                ctx = state_data.context
                curr_map_id = ctx.get("map_id", 0)
                curr_x, curr_y = ctx.get("x", 0), ctx.get("y", 0)
                last_x, last_y = st.session_state.last_coords
                dx, dy = curr_x - last_x, curr_y - last_y

                dx_metric.metric("Delta X", dx)
                dy_metric.metric("Delta Y", dy)
                st.session_state.last_coords = (curr_x, curr_y)

                status_box.write(f"**Map:** #{curr_map_id} | **X:** {curr_x} | **Y:** {curr_y}")
                ocr_box.info(f"**OCR:** {state_data.ocr_text or 'None'}")

                # Render Visualizations
                grid_map_id = curr_map_id
                
                grid_fig = render_spatial_grid(db_path, grid_map_id, curr_x, curr_y)
                if grid_fig:
                    grid_chart.plotly_chart(grid_fig, use_container_width=True)
                
                t_fig = render_transition_graph(db_path)
                if t_fig:
                    transition_chart.plotly_chart(t_fig, use_container_width=True)
                
                f_fig = render_spatial_map(db_path, grid_map_id)
                if f_fig:
                    frontier_chart.plotly_chart(f_fig, use_container_width=True)
                
                l_fig = render_latent_space(db_path)
                if l_fig:
                    latent_chart.plotly_chart(l_fig, use_container_width=True)
                
                a_fig = render_reward_attribution(db_path)
                if a_fig:
                    attribution_chart.plotly_chart(a_fig, use_container_width=True)
                
                r_fig = render_rnd_tracking(db_path)
                if r_fig:
                    rnd_chart.plotly_chart(r_fig, use_container_width=True)

            if state_data.last_action:
                action_info.info(f"🤖 **Action:** {state_data.last_action.upper()}")
                reasoning_log.info(f"💬 {state_data.last_reasoning or 'Thinking...'}")

        except Exception as e:
            status_box.warning(f"Searching for Game Server... ({str(e)[:50]})")

    update_pass()


if __name__ == "__main__":
    main()
