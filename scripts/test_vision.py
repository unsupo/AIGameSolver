import time
from autogameplayer.vision.encoder import VisionEncoder
from PIL import Image

print("Initializing VisionEncoder...")
start_t = time.time()
encoder = VisionEncoder()
# Trigger lazy init
dummy_img = Image.new("RGB", (240, 160))
print("Encoding first image (triggering lazy init)...")
vec = encoder.encode(dummy_img)
print(f"VisionEncoder ready in {time.time() - start_t:.2f}s")
print(f"Vector shape: {vec.shape}")
