import asyncio
from autogameplayer.core.config import settings
from autogameplayer.brains.agentic.memory import LongTermMemory
from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.utils.llm import OpenAIClientWrapper
from openai import AsyncOpenAI

async def run_dream_cycle():
    print("🌙 AI is entering 'Dreaming' phase (Consolidating Memories)...")
    
    # 1. Initialize Strategy Optimizer
    optimizer = StrategyOptimizer()
    
    # 2. Analyze raw logs for repeating success patterns
    print("📊 Analyzing past sessions for successful patterns...")
    optimizer.optimize()
    
    # 3. Memory Pruning & Counterfactual Reflection
    raw_client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    client = OpenAIClientWrapper(raw_client)
    ltm = LongTermMemory(client)
    
    print(f"🧹 Pruning redundant memories from {settings.llm_model} history...")
    initial_count = len(ltm.memories)
    
    # New efficient SQL-based pruning
    ltm.delete_redundant_memories()
    
    # Still collect failures for reflection
    failed_attempts = [m['text'] for m in ltm.memories if "CRITICAL WARNING" in m['text'] or "FAILED" in m['text']]
        
    print(f"✅ Pruning complete. Reduced {initial_count} memories to {len(ltm.memories)} unique insights.")

    # 4. Counterfactual Reflection
    if len(failed_attempts) > 5:
        print("🤔 Performing Counterfactual Reflection on past failures...")
        sample_failures = "\n".join(failed_attempts[-20:]) # Take up to 20 recent failures
        prompt = f"""
        You are an expert game AI behavior analyst.
        Review the following log of recent FAILED actions by an AI agent:
        
        {sample_failures}
        
        Identify the most common obstacle or reason for these failures. 
        Write a single, strict, imperative rule (starting with "CRITICAL RULE:") that the AI should follow to avoid this specific trap in the future.
        """
        try:
            # We need to call the completion method on our wrapper
            new_rule_text = await client.acreate_completion(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            new_rule = new_rule_text.strip()
            print(f"💡 Dream Insight: {new_rule}")
            await ltm.add_memory(new_rule, {"type": "rule", "source": "dream_reflection"})
        except Exception as e:
            print(f"⚠️ Reflection failed: {e}")
            
    # Save the pruned (and possibly expanded) memory
    ltm._save_to_disk()

if __name__ == "__main__":
    asyncio.run(run_dream_cycle())
