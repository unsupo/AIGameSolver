# Configs Directory ⚙️

This directory contains the orchestration and profile configurations for different games and agent behaviors.

### Contents
- `profiles/`: Contains YAML memory maps and curriculum definitions for specific games (e.g., `pokemon_red.yaml`).
- `agnostic_standard.yaml`: A baseline configuration for games without a specific profile.
- `pokemon_red_agentic.yaml`: The primary configuration for the agentic Pokémon Red stack.
- `pokemon_red_llm.yaml`: A vision-only LLM-driven configuration.

### Usage
Configurations defined here are passed to the `--config` flag when launching the AI or the Server.
