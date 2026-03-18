import httpx
import base64
import time
from PIL import Image
from io import BytesIO


def test_ollama():
    print("🧪 Testing Ollama Speed directly...")

    # 1. Create a tiny test image
    img = Image.new("RGB", (100, 100), color="red")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "llama3.2-vision",
        "prompt": "What color is this image? Reply with one word.",
        "stream": False,
        "images": [img_str],
    }

    start = time.time()
    try:
        print("📡 Sending request to Ollama...")
        resp = httpx.post(url, json=payload, timeout=30.0)
        elapsed = time.time() - start
        if resp.status_code == 200:
            print(f"✅ Success! Response: {resp.json().get('response')}")
            print(f"⏱️ Time taken: {elapsed:.2f} seconds")
        else:
            print(f"❌ Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")


if __name__ == "__main__":
    test_ollama()
