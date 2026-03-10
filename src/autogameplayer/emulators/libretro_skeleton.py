from PIL import Image
import numpy as np
from pyboy_advance.gba import PyBoyAdvance, WindowEvent
from autogameplayer.core.interfaces import BaseEmulator
from autogameplayer.core.controllers import build_button_map, GBA_BUTTONS
from autogameplayer.core.registry import Registry

@Registry.register_emulator([".gba-alt"])
class LibRetroWrapper(BaseEmulator):
    """Skeleton/experimental GBA wrapper using pyboy-advance."""
    def __init__(self, rom_path: str):
        print(f"Initializing GBA Emulator (PyBoyAdvance) for {rom_path}...", flush=True)
        from autogameplayer.core.config import settings
        bios_path = settings.bios_dir / "gba_bios.bin"
        # We MUST use the BIOS for pyboy-advance to render correctly on boot
        self.gba = PyBoyAdvance(gamepak=rom_path, bios=str(bios_path) if bios_path.exists() else None, skip_bios=False)
        self.experimental = True
        
        # We skip 100 frames (~50s) to get past the BIOS animation and into the game logo
        print("Skipping BIOS frames (this may take ~30-60s)...", flush=True)
        for _ in range(100):
            self.gba.frame()
        print("Emulator ready.", flush=True)
        
        self.button_map = build_button_map(WindowEvent, "DPAD", GBA_BUTTONS)

    @property
    def emulator_name(self) -> str:
        return "PyBoyAdvance (Legacy GBA Bridge)"

    @property
    def supported_buttons(self) -> list[str]:
        return list(self.button_map.keys())

    @property
    def has_release(self) -> bool:
        return True

    def get_screenshot(self) -> Image.Image:
        raw_buf = np.array(self.gba.ppu.front_buffer, dtype=np.uint16).reshape((160, 240))
        r = ((raw_buf >> 0) & 0x1F) << 3
        g = ((raw_buf >> 5) & 0x1F) << 3
        b = ((raw_buf >> 10) & 0x1F) << 3
        rgb888 = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return Image.fromarray(rgb888)

    def send_input(self, button: str):
        if button in self.button_map:
            event = self.button_map[button]
            self.gba.keypad.process_window_event(event)
        else:
            raise ValueError(f"Unknown button: {button}")

    def tick(self, frames: int = 1):
        for _ in range(frames):
            self.gba.frame()

    def close(self):
        pass
