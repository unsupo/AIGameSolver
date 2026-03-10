# Free Local LLM Setup for AutoGamePlayer

You can run a high-quality Vision LLM locally for free using **Ollama**. This allows the "LLM Brain" to play the game without any API costs or privacy concerns.

### 1. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com).

### 2. Pull a Vision Model
Open your terminal and run:
```bash
ollama pull llama3.2-vision
```
*(Llama 3.2 Vision is a fast, high-performance multimodal model that can "see" your game screen.)*

### 3. Configure AutoGamePlayer
Create or update your `.env` file in the project root:
```env
AGP_LLM_PROVIDER=ollama
AGP_LLM_MODEL=llama3.2-vision
AGP_LLM_BASE_URL=http://localhost:11434/v1
AGP_LLM_API_KEY=ollama
```

### 4. Run the Game
```bash
uv run nexus --rom "roms/Pokemon - Red Version.gb"
```
*(Ensure your config at `configs/pokemon_red_llm.yaml` is set to `brain: "llm"`. I have already created this file for you.)*

### Why this works:
Ollama provides an **OpenAI-compatible API** at `localhost:11434`. AutoGamePlayer uses this endpoint to send screenshots and get back button decisions, just like it would with GPT-4o, but entirely on your own machine.
