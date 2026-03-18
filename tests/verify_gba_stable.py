import numpy as np
from autogameplayer.emulators.factory import create_emulator
import sys


def verify():
    rom = "roms/Pokemon - FireRed Version (USA, Europe) (Rev 1).gba"
    print(f"🧪 Testing GBA stability for: {rom}")
    try:
        emu = create_emulator(rom)
        print("✅ Core initialized.")

        # Test a few frames
        for i in range(5):
            emu.tick(1)
            print(f"  Frame {i + 1} ticked.")

        img = emu.get_screenshot()
        pixels = np.array(img)
        print(f"✅ Screenshot captured: {img.size} {pixels.dtype}")

        if np.mean(pixels) > 0:
            print("✅ Visual data confirmed (Non-black screen).")
        else:
            print("⚠️  Warning: Screen is black (Normal during early BIOS).")

        emu.close()
        print("✅ Core closed safely. No segfault.")

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    verify()
