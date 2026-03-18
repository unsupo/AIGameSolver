import json
import asyncio
from openai import OpenAI
from collections import defaultdict
from autogameplayer.core.config import settings
from autogameplayer.brains.agentic_brain import LongTermMemory


async def consolidate_memories():
    print("🌙 AI is entering REM Sleep to optimize its memories...")

    # Use synchronous client wrapped in to_thread for consistency with LongTermMemory
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    memory = LongTermMemory(client=client)

    if not memory.memories:
        print("No memories to optimize.")
        return

    # 1. Group memories by Map ID
    map_clusters = defaultdict(list)
    for mem in memory.memories:
        map_id = mem.get("metadata", {}).get("map_id")
        m_type = mem.get("metadata", {}).get("type")
        if map_id is not None and m_type == "experience":
            map_clusters[map_id].append(mem["text"])

    # 2. Ask the LLM to clean up and optimize each map's macros
    optimized_memories = []

    for map_id, experiences in map_clusters.items():
        if len(experiences) < 2:
            continue  # Nothing to consolidate if there's only 1 memory

        print(f"\n🧠 Optimizing {len(experiences)} experiences for Map {map_id}...")

        prompt = f"""
        You are optimizing the memory database of a TAS (Tool-Assisted Speedrun) AI for Pokemon.
        Here are several raw experiences the AI recorded on Map {map_id}:
        
        {json.dumps(experiences, indent=2)}
        
        TASK:
        1. Identify the most efficient button sequence (macro) to achieve the successes.
        2. Remove any redundant, contradictory, or useless wandering.
        3. Convert the sequence into a JSON list of TAS commands: {{"button": "name", "frames": count}}.
        4. Output a SINGLE, perfectly optimized "MASTER MACRO" rule for Map {map_id}.
        
        Start your response with: "OPTIMIZED MACRO:"
        """

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )

            optimized_rule = response.choices[0].message.content.strip()
            print(f"✨ Cleaned Rule: {optimized_rule}")
            optimized_memories.append(
                {
                    "text": optimized_rule,
                    "metadata": {"map_id": map_id, "type": "optimized_macro"},
                }
            )
        except Exception as e:
            print(f"⚠️ Optimization failed for Map {map_id}: {e}")

    # 3. Wipe the old sloppy experiences and save the optimized ones
    if optimized_memories:
        # Filter out the old raw experiences that were consolidated
        # We only remove the 'experience' type for maps we actually optimized
        maps_optimized = [m["metadata"]["map_id"] for m in optimized_memories]

        original_count = len(memory.memories)
        memory.memories = [
            m
            for m in memory.memories
            if not (
                m.get("metadata", {}).get("type") == "experience"
                and m.get("metadata", {}).get("map_id") in maps_optimized
            )
        ]

        # Also remove corresponding embeddings
        # Since LongTermMemory keeps them in sync by index, we need a better way to wipe them.
        # For now, let's just reset the embeddings list and re-add EVERYTHING to be safe,
        # or just call add_memory for the new ones.

        # Simpler: Filter both lists together
        new_mems = []
        new_embs = []
        for i, m in enumerate(memory.memories):
            new_mems.append(m)
            new_embs.append(memory.embeddings[i])

        memory.memories = new_mems
        memory.embeddings = new_embs

        # Add the new optimized ones
        for opt_mem in optimized_memories:
            await memory.add_memory(opt_mem["text"], opt_mem["metadata"])

        print(
            f"\n✅ Memory Consolidation Complete. (Before: {original_count}, After: {len(memory.memories)})"
        )
    else:
        print("\nNo maps had enough raw data to justify consolidation yet.")


if __name__ == "__main__":
    asyncio.run(consolidate_memories())
