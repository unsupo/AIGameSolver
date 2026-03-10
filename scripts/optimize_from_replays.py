import os
import asyncio
from pathlib import Path
from openai import OpenAI
from autogameplayer.core.replay import ReplaySystem
from autogameplayer.core.config import settings
from autogameplayer.brains.agentic_brain import LongTermMemory

async def review_past_runs():
    print("🌙 AI is entering 'Dream State' to review past runs...")
    
    replay_sys = ReplaySystem()
    # Use standard OpenAI client to match LongTermMemory expectation
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    memory = LongTermMemory(client=client) # Loads existing disk memory
    
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("No past runs found to review.")
        return

    # Iterate through all saved episodes
    files_to_process = list(datasets_dir.glob("episode_*.json"))
    if not files_to_process:
        print("No new episodes to process.")
        return

    for episode_file in files_to_process:
        episode_id = episode_file.stem.replace("episode_", "")
        print(f"\n📂 Reviewing Run: {episode_id}")
        
        try:
            events = replay_sys.get_episode(episode_id)
        except Exception as e:
            print(f"Error loading episode {episode_id}: {e}")
            continue
        
        # Look for spikes in the reward signal
        for i, event in enumerate(events):
            reward = event.get("reward", 0.0)
            
            # If the AI achieved something highly successful (e.g., gained a level or badge)
            if reward > 5.0:
                print(f"⭐ High Reward ({reward}) detected at step {i}!")
                
                # Gather the context of the last 10 steps that led to this success
                context_window = events[max(0, i-10):i+1]
                actions_taken = [e["action"]["button"] for e in context_window if "action" in e and "button" in e["action"]]
                ocr_context = [e["ocr"] for e in context_window if e.get("ocr")]
                
                # Ask the LLM to summarize WHY this was successful
                prompt = f"""
                You are analyzing a past run of a Pokemon AI.
                The AI took the following sequence of actions: {actions_taken}
                The text on screen during this time was: {ocr_context}
                This resulted in a MASSIVE SUCCESS (Reward: {reward}).
                
                Write a single, concise strategy rule (max 2 sentences) that the AI should remember for the future based on this success.
                Start the rule with "STRATEGY:"
                """
                
                try:
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=settings.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,
                        temperature=0.3
                    )
                    
                    learned_strategy = response.choices[0].message.content.strip()
                    print(f"🧠 Extracted Knowledge: {learned_strategy}")
                    
                    # Save the learned strategy permanently to the RAG database
                    await memory.add_memory(
                        text=learned_strategy,
                        metadata={"type": "learned_strategy", "source_episode": episode_id}
                    )
                except Exception as e:
                    print(f"Error extracting strategy: {e}")
                
        # Rename the file after processing so it isn't reviewed twice
        try:
            os.rename(episode_file, episode_file.with_suffix(".json.reviewed"))
        except Exception as e:
            print(f"Error renaming file {episode_file}: {e}")

if __name__ == "__main__":
    asyncio.run(review_past_runs())
