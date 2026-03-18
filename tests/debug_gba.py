import numpy as np
from pyboy_advance.gba import PyBoyAdvance
from PIL import Image


def debug_boot():
    rom = "roms/Pokemon - FireRed Version (USA, Europe) (Rev 1).gba"
    bios = "gba_bios.bin"  # The one we downloaded

    print("--- Debugging GBA Boot ---")

    configs = [
        {"name": "Skip BIOS (True)", "skip_bios": True, "bios": None},
        {
            "name": "Skip BIOS (False) + Downloaded BIOS",
            "skip_bios": False,
            "bios": bios,
        },
    ]

    for cfg in configs:
        print(f"\nTesting: {cfg['name']}")
        try:
            gba = PyBoyAdvance(
                gamepak=rom, bios=cfg["bios"], skip_bios=cfg["skip_bios"]
            )

            # Tick significantly more frames
            for i in range(1000):
                gba.frame()
                if i % 200 == 0:
                    raw_buf = np.array(gba.ppu.front_buffer, dtype=np.uint16)
                    non_zero = np.count_nonzero(raw_buf)
                    print(f"  Frame {i}: Non-zero pixels: {non_zero}")

                    if non_zero > 0:
                        print(f"  ✅ SUCCESS: Pixels found at frame {i}!")
                        # Save frame
                        raw_buf = raw_buf.reshape((160, 240))
                        r = ((raw_buf >> 0) & 0x1F) << 3
                        g = ((raw_buf >> 5) & 0x1F) << 3
                        b = ((raw_buf >> 10) & 0x1F) << 3
                        rgb888 = np.stack([r, g, b], axis=-1).astype(np.uint8)
                        Image.fromarray(rgb888).save(
                            f"debug_{cfg['name'].replace(' ', '_')}.png"
                        )
                        break
            else:
                print("  ❌ Still black after 1000 frames.")

        except Exception as e:
            print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    debug_boot()
