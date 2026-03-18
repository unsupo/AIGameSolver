import subprocess
import httpx
import json
import re
from typing import Protocol, List, Dict, Any, Optional
from autogameplayer.core.config import settings
from openai import AsyncOpenAI


class LLMClientProtocol(Protocol):
    async def acreate_embedding(self, text: str, model: str) -> List[float]: ...
    async def acreate_completion(
        self, messages: List[Dict[str, Any]], model: str, **kwargs
    ) -> str: ...


def extract_json_from_llm_response(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts and parses the first JSON object found in an LLM string."""
    if not text:
        return None

    # Try simple strip first
    text = text.strip()

    # 1. Try to find content between triple backticks (Markdown JSON)
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            # Fall through to more aggressive methods if backtick content is not pure JSON
            pass

    # 2. Try to find JSON block using regex if there is markdown or noise
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0).strip()
        try:
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            # If the greedy match fails, try a non-greedy one in case there are multiple blocks
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
                try:
                    return json.loads(json_str)
                except (ValueError, json.JSONDecodeError):
                    pass

    # 3. Fallback to finding first { and last }
    try:
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        pass

    return None


class OpenAIClientWrapper:
    """Wraps AsyncOpenAI client to conform to LLMClientProtocol."""

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def acreate_embedding(self, text: str, model: str) -> List[float]:
        response = await self.client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    async def acreate_completion(
        self, messages: List[Dict[str, Any]], model: str, **kwargs
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content


def get_llm_client() -> LLMClientProtocol:
    """Factory function to create and wrap the AsyncOpenAI client."""
    from openai import AsyncOpenAI

    raw_client = AsyncOpenAI(
        api_key=settings.llm_api_key, base_url=settings.llm_base_url
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
                    print(
                        "🚀 Ollama not responding. Attempting to launch Ollama.app..."
                    )
                    subprocess.run(["open", "-a", "Ollama"])

                print(f"⏳ Waiting for Ollama (Attempt {attempt + 1}/3)...")
                time.sleep(5)

        if not connected:
            print(f"❌ Ollama is not running at {base_url}")
            print("💡 Please ensure the Ollama app is open in your menu bar.")
            return False

        # 2. Check and pull models
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(base_url.replace("/v1", "/api/tags"))
                installed_models = [m["name"] for m in resp.json().get("models", [])]
                installed_names = [
                    m.split(":")[0] for m in installed_models
                ] + installed_models

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
