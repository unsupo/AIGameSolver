import json
import os

def cleanup_memory():
    path = "models/long_term_memory.json"
    if not os.path.exists(path):
        print("No memory file found.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    original_count = len(data["memories"])
    
    # Filter out problematic entries
    new_memories = []
    new_embeddings = []
    
    for i, mem in enumerate(data["memories"]):
        text = mem["text"]
        # Filter out minor successes (the spam we fixed)
        if text.startswith("MINOR SUCCESS:"):
            continue
        # Filter out the truncated rule mentioned by the user
        if text.startswith("CRITICAL RULE FOR MAP 38: The AI must not repeatedly execute the action \"A\""):
            continue
        # Filter out corrupted thoughts about mashing A at (3,6)
        if "(3, 6)" in text or "maximize reward collection" in text or "Focus on repeatedly pressing 'A'" in text:
            continue
        # Filter out critical warnings involving only A/B spam on Map 38 or 0
        if "doing 'A -> A -> A'" in text or "doing 'B -> B -> B'" in text:
            if mem.get("metadata", {}).get("map_id") in [0, 38]:
                continue
        
        new_memories.append(mem)
        new_embeddings.append(data["embeddings"][i])

    data["memories"] = new_memories
    data["embeddings"] = new_embeddings

    with open(path, "w") as f:
        json.dump(data, f)

    print(f"Cleanup complete. Removed {original_count - len(new_memories)} entries.")

if __name__ == "__main__":
    cleanup_memory()
