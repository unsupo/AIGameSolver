# Universal Game Nexus

Agnostic Game Emulator MCP Server with Vision Encoding.

## Features
- **Agnostic Core**: Uses PyBoy for Game Boy emulation.
- **Perception Module**: Uses `facebook/dinov2-small` to generate 384-dimension vision vectors.
- **MCP Tools**:
  - `get_game_state()`: Base64 screenshot, Vision Vector, OCR.
  - `send_input(button)`: Press/Release game buttons.
  - `manage_checkpoint(action, slot)`: Save/Load emulator states.

## AI Brains
- **Intro Brain**: Mashing START and A to get past logos.
- **Walk Brain**: Straight-line exploration.
- **Vision Brain**: Detects stagnation via DINOv2 vectors.
- **Evolution Brain**: Neural Network evolved via Genetic Algorithm.
- **LLM Brain**: GPT-4o powered multimodal decision making.

### Using the LLM Brain
To use the LLM brain, set your API key in a `.env` file:
```env
AGP_LLM_API_KEY=your_openai_api_key
AGP_LLM_MODEL=gpt-4o
```
Then run:
```bash
uv run nexus --rom "roms/Pokemon - Red Version.gb"
```
And make sure the config is set to `brain: "llm"`.

## Usage
1. Place a Game Boy ROM (`.gb` or `.gbc`) in the project root.
2. Run with `uv`:
   ```bash
   uv run nexus
   ```
3. The MCP server will start on SSE by default at `http://localhost:8000/sse`.

## Requirements
- Python 3.12+
- `uv` installed
# AIGameSolver
