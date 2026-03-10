# Universal Game Nexus

An Agnostic Game Emulator MCP Server with Vision Encoding, SLAM Mapping, and Agentic Reasoning.

## 🚀 Key Features

- **Hierarchical Agentic Brain**: Uses a **Planner-Actor-Critic** pattern to synthesize high-level strategy with low-level tactical execution.
- **SLAM (Simultaneous Localization and Mapping)**: Dynamically constructs a 2D occupancy grid of the game world, tracking "Walkable" vs "Blocked" tiles with a weighted **Impassable Score**.
- **A* Pathfinder**: Translates coordinate-based goals from the LLM into frame-perfect directional sequences using the learned spatial graph.
- **Perception Engine**: Uses `facebook/dinov2-small` for high-dimensional vision vectors and `pytesseract` for real-time OCR.
- **Autonomous Macro Synthesis (Self-TAS)**:
    - **Gold Run Discovery**: Distills high-reward sequences into reusable skills.
    - **Genetic Refinement**: Evolves macros using mutation and crossover to find the fastest possible paths.
    - **Reactive Execution**: Macros stop immediately if a collision is detected via the Odometer.
- **Real-Time Interactive Dashboard**:
    - **Spatial Grid**: Live Plotly heatmap of the AI's internal map belief.
    - **Brain Stream**: Live feed of the AI's reasoning, current plan, and RAG retrieval.
    - **Motion Delta**: Odometer monitoring to visualize "Intent vs Reality" during movement.

## 🧠 AI Brains

- **Agentic Brain (`agentic`)**: The flagship brain. Features long-term memory (RAG), milestone persistence, and multi-model support (e.g., using a 12b model for planning and 4b for acting).
- **LLM Brain (`llm`)**: Direct decision-making using multimodal models (GPT-4o, Gemma 3, etc.).
- **Macro Wrapper**: Intercepts any brain's decision to check if a high-confidence macro/TAS entry point exists for the current screen hash.
- **Evolution Brain**: Neural Network evolved via Genetic Algorithms.

## 🛠️ Installation & Setup

1. **Requirements**: Python 3.14+, `uv` installed, and `Ollama` (for local LLMs).
2. **Clone and Install**:
   ```bash
   git clone https://github.com/your-repo/AutoGamePlayer.git
   cd AutoGamePlayer
   uv sync
   ```
3. **Configure**: Create a `.env` file for API keys if using cloud models (GPT-4o, Claude):
   ```env
   AGP_LLM_API_KEY=your_key
   ```

## 🏃 Usage

### Standard Launch
Place your Game Boy ROM in `data/roms/` and run:
```bash
uv run nexus --rom "data/roms/Pokemon - Red Version.gb"
```

### Headless Training
To run extended training sessions without the Streamlit UI:
```bash
uv run nexus --no-dashboard --config configs/pokemon_red_agentic.yaml
```

### Inspect Learned Skills
Dump the current SQL-based skill registry to see what the AI has learned:
```bash
uv run nexus --list-skills
```

## 🗺️ Project Structure

- `src/autogameplayer/brains/agentic/`: Core agents (Planner, Actor, Critic, Reflector).
- `src/autogameplayer/core/context.py`: RAM extraction and UI_LOCKED logic.
- `src/autogameplayer/core/optimizer.py`: Genetic evolution and sequence mining.
- `src/autogameplayer/utils/pathfinding.py`: A* search implementation.
- `src/autogameplayer/dashboard.py`: Streamlit & Plotly visualization.

## 📼 Automated TAS Recording
The system automatically captures `.json` TAS files whenever:
1. A **Milestone** is achieved (Save to Slot 1).
2. A **Warp** occurs (Map Transition).
3. A **Stagnation Rollback** is triggered (Rollback to Slot 1).

## ⚖️ License
MIT
