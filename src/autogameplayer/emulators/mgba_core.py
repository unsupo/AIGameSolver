import ctypes
import os
import numpy as np
from PIL import Image
from typing import List
from autogameplayer.core.interfaces import BaseEmulator
from autogameplayer.core.registry import Registry

@Registry.register_emulator([".gba"])
class MGBAEmulator(BaseEmulator):
    def __init__(self, rom_path: str):
        lib_path = os.path.abspath("scripts/gba_bridge.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native bridge not found at {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.gba_init.argtypes = [ctypes.c_char_p]
        self.lib.gba_init.restype = ctypes.c_int
        self.lib.gba_step.argtypes = []
        self.lib.gba_set_keys.argtypes = [ctypes.c_uint32]
        self.lib.gba_get_pixels.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.lib.gba_read_memory.argtypes = [ctypes.c_uint32]
        self.lib.gba_read_memory.restype = ctypes.c_uint8
        self.lib.gba_write_memory.argtypes = [ctypes.c_uint32, ctypes.c_uint8]
        
        # Initialize
        res = self.lib.gba_init(rom_path.encode('utf-8'))
        if res != 0:
            raise RuntimeError(f"GBA Bridge: Failed to init core (Error {res})")
            
        # Allocate pixel buffer
        self.width = 240
        self.height = 160
        self.pixel_buf = (ctypes.c_uint32 * (self.width * self.height))()
        
        # Fast forward past logos (60 FPS * 5 seconds = 300 frames)
        # This is now VERY fast.
        for _ in range(300):
            self.lib.gba_step()
            
        self.current_keys = 0
        self.button_map = {
            "a": 1<<0, "b": 1<<1, "select": 1<<2, "start": 1<<3,
            "right": 1<<4, "left": 1<<5, "up": 1<<6, "down": 1<<7,
            "r": 1<<8, "l": 1<<9
        }

    @property
    def emulator_name(self) -> str:
        return "mGBA (Game Boy Advance)"

    @property
    def supported_buttons(self) -> List[str]:
        return list(self.button_map.keys())

    @property
    def has_release(self) -> bool: return True

    def get_screenshot(self) -> Image.Image:
        # Tell the C bridge where our buffer is
        self.lib.gba_get_pixels(self.pixel_buf)
        # Force a single step to render into that buffer
        self.lib.gba_step()
        
        # Convert to numpy and then PIL
        arr = np.frombuffer(self.pixel_buf, dtype=np.uint32).reshape((self.height, self.width))
        
        # mGBA usually outputs 0xRRGGBB or 0xBBGGRR depending on platform
        # Let's extract components (Assuming 0xAABBGGRR for Mac)
        r = (arr & 0x000000FF).astype(np.uint8)
        g = ((arr & 0x0000FF00) >> 8).astype(np.uint8)
        b = ((arr & 0x00FF0000) >> 16).astype(np.uint8)
        
        rgb = np.stack([r, g, b], axis=-1)
        return Image.fromarray(rgb)

    def send_input(self, button: str):
        is_release = button.startswith("release_")
        btn_name = button.replace("release_", "")
        if btn_name in self.button_map:
            mask = self.button_map[btn_name]
            if is_release: self.current_keys &= ~mask
            else: self.current_keys |= mask
            self.lib.gba_set_keys(self.current_keys)

    def tick(self, frames: int = 1):
        for _ in range(frames):
            self.lib.gba_step()

    def read_memory(self, address: int) -> int:
        # Step the emulator to ensure we are reading a fresh state
        self.lib.gba_step()
        return self.lib.gba_read_memory(address)

    def write_memory(self, address: int, value: int):
        self.lib.gba_write_memory(address, value)

    def read_memory_block(self, address: int, length: int) -> bytes:
        # Step once before the block read
        self.lib.gba_step()
        return bytes([self.lib.gba_read_memory(address + i) for i in range(length)])

    def manage_checkpoint(self, action: str, slot: int):
        pass # Can be added to bridge.c later

    def close(self):
        if hasattr(self, "lib"):
            self.lib.gba_cleanup()
