from pyboy import PyBoy
import time
import os

rom_path = "data/roms/Pokemon - Red Version.gb"
if not os.path.exists(rom_path):
    print(f"ERROR: ROM not found at {rom_path}")
    exit(1)

print(f"Initializing PyBoy with {rom_path}...")
start_t = time.time()
pyboy = PyBoy(rom_path, window="null", sound=False)
pyboy.set_emulation_speed(0)
print(f"PyBoy initialized in {time.time() - start_t:.2f}s")

print("Skipping 1500 frames...")
start_t = time.time()
for i in range(1500):
    pyboy.tick(render=False)
    if (i+1) % 100 == 0:
        print(f"  - Ticked {i+1} frames...")
print(f"Skipped 1500 frames in {time.time() - start_t:.2f}s")

pyboy.stop()
print("PyBoy closed.")
