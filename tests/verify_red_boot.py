import numpy as np
from autogameplayer.emulators.factory import create_emulator
import os

def test_boot():
    rom_path = "roms/Pokemon - Red Version.gb"
    if not os.path.exists(rom_path):
        print(f"❌ ROM not found at {rom_path}")
        return

    print(f"💿 Testing boot sequence for {rom_path}...")
    emulator = create_emulator(rom_path)
    
    # 300 frames are already skipped in constructor
    
    # Take a screenshot
    img = emulator.get_screenshot()
    img.save("debug_red_boot.png")
    
    # Check pixels
    pixels = np.array(img)
    non_zero = np.count_nonzero(pixels)
    total = pixels.size
    percent = (non_zero / total) * 100
    
    print("🖼️  Screenshot saved to 'debug_red_boot.png'")
    print(f"📊 Non-zero pixels: {non_zero} / {total} ({percent:.2f}%)")
    
    if non_zero > 0:
        print("✅ SUCCESS: The screen is NOT black!")
    else:
        # Try ticking more
        print("⏳ Screen is black, ticking 300 more frames...")
        for _ in range(300):
            emulator.tick(1)
        
        img = emulator.get_screenshot()
        pixels = np.array(img)
        non_zero = np.count_nonzero(pixels)
        print(f"📊 Non-zero pixels after extra tick: {non_zero}")
        if non_zero > 0:
            print("✅ SUCCESS: The screen is NOT black now!")
        else:
            print("❌ FAILURE: The screen is still black.")

if __name__ == "__main__":
    test_boot()
