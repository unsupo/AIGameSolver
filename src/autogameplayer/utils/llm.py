import subprocess
import httpx
from typing import Protocol, List, Dict, Any
from autogameplayer.core.config import settings
from openai import AsyncOpenAI

class LLMClientProtocol(Protocol):
    async def acreate_embedding(self, text: str, model: str) -> List[float]: ...
    async def acreate_completion(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> str: ...

class OpenAIClientWrapper:
    """Wraps AsyncOpenAI client to conform to LLMClientProtocol."""
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def acreate_embedding(self, text: str, model: str) -> List[float]:
        response = await self.client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    async def acreate_completion(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

def get_llm_client() -> LLMClientProtocol:
    """Factory function to create and wrap the AsyncOpenAI client."""
    from openai import AsyncOpenAI
    raw_client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url
    )
    return OpenAIClientWrapper(raw_client)

class OllamaBootstrap:
    """Utility for bootstrapping Ollama models with robust connectivity and auto-launch."""
    @staticmethod
    def bootstrap(models: List[str]):
        if not settings.is_ollama:
            return
            
        import sys
        import time
        base_url = settings.llm_base_url.replace("localhost", "127.0.0.1")
        
        # 1. Try to connect with retries
        connected = False
        for attempt in range(3):
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(base_url.replace("/v1", "/api/tags"))
                    if resp.status_code == 200:
                        connected = True
                        break
            except Exception:
                if attempt == 0 and sys.platform == "darwin":
                    print("🚀 Ollama not responding. Attempting to launch Ollama.app...")
                    subprocess.run(["open", "-a", "Ollama"])
                
                print(f"⏳ Waiting for Ollama (Attempt {attempt+1}/3)...")
                time.sleep(5)
        
        if not connected:
            print(f"❌ Ollama is not running at {base_url}")
            print("💡 Please ensure the Ollama app is open in your menu bar.")
            return False

        # 2. Check and pull models
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(base_url.replace("/v1", "/api/tags"))
                installed_models = [m['name'] for m in resp.json().get('models', [])]
                installed_names = [m.split(":")[0] for m in installed_models] + installed_models
                
                for model in models:
                    if model and model not in installed_names:
                        print(f"📥 Pulling missing model from Ollama: {model}...")
                        subprocess.run(["ollama", "pull", model], check=True)
                    elif model:
                        print(f"✅ Local model '{model}' is ready.")
            return True
        except Exception as e:
            print(f"⚠️ Error checking models: {e}")
            return False
