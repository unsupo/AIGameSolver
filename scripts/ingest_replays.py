import json
from pathlib import Path
import asyncio
from openai import AsyncOpenAI
from autogameplayer.core.config import settings
from autogameplayer.brains.agentic.memory import LongTermMemory
from autogameplayer.utils.llm import OpenAIClientWrapper

raw_client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
client = OpenAIClientWrapper(raw_client)
ltm = LongTermMemory(client)


async def summarize_sequence(steps):
    """Asks the LLM to summarize a successful sequence of actions."""
    if not steps:
        return None

    # Format the sequence for the LLM
    seq_text = []
    for i, s in enumerate(steps):
        ctx = s.get("context", {})
        action = s.get("action", {}).get("button", "none")
        ocr = s.get("ocr", "")
        seq_text.append(
            f"Step {i}: Map {ctx.get('map_id')} @ ({ctx.get('x')},{ctx.get('y')}) | Pressed {action.upper()} | OCR: '{ocr}'"
        )

    prompt = f"""
    The following is a sequence of successful steps in a Pokemon game that led to a reward.
    Analyze the sequence and provide a ONE SENTENCE strategic hint for future runs.
    
    SEQUENCE:
    {chr(10).join(seq_text)}
    
    HINT EXAMPLE: "To reach the world map from the bedroom, walk south and exit through the door."
    HINT:
    """

    try:
        response_text = await client.acreate_completion(
            model="gemma3:4b",  # Use a fast model for ingestion
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.3,
        )
        return response_text.strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return None


async def ingest_replays():
    dataset_dir = Path("datasets")
    if not dataset_dir.exists():
        print("No datasets found.")
        return

    print(f"🔍 Scanning {dataset_dir} for high-reward moments...")

    for file in dataset_dir.glob("episode_*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)

            if not data:
                continue

            # Find spikes in reward
            for i, step in enumerate(data):
                reward = step.get("reward", 0)
                if reward > 5.0:  # Significant reward threshold
                    print(
                        f"✨ Found reward spike ({reward}) in {file.name} at step {i}"
                    )

                    # Take previous 15 steps as context
                    start_idx = max(0, i - 15)
                    sequence = data[start_idx : i + 1]

                    hint = await summarize_sequence(sequence)
                    if hint:
                        map_id = step.get("context", {}).get("map_id", 0)
                        desc = f"PAST EXPERIENCE SUCCESS: {hint}"
                        print(f"💾 Ingesting hint for Map {map_id}: {desc}")
                        await ltm.add_memory(
                            desc, {"map_id": map_id, "type": "experience_summary"}
                        )

        except Exception as e:
            print(f"Error processing {file.name}: {e}")


if __name__ == "__main__":
    asyncio.run(ingest_replays())
