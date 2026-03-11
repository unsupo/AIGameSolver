from pyboy import PyBoy
from pyboy.utils import WindowEvent
from PIL import Image
import numpy as np
from typing import List
from autogameplayer.core.interfaces import BaseEmulator
from autogameplayer.core.registry import Registry
from autogameplayer.core.controllers import build_button_map, STANDARD_BUTTONS

@Registry.register_emulator([".gb", ".gbc"])
class PyBoyWrapper(BaseEmulator):
    def __init__(self, rom_path: str, window_type: str = "null"):
        self.pyboy = PyBoy(rom_path, window=window_type, sound=False)
        self.pyboy.set_emulation_speed(0) # Max speed
        self.experimental = False
        
        # Advance 1500 frames (~25 seconds) to get past initial BIOS and into title screen
        for _ in range(1490):
            self.pyboy.tick(render=False) # Fast skip
        
        # Last 10 frames with rendering to ensure buffer is hot
        for _ in range(10):
            self.pyboy.tick(render=True)
        
        self.button_map = build_button_map(WindowEvent, "ARROW", STANDARD_BUTTONS)

    @property
    def emulator_name(self) -> str:
        return "PyBoy (Game Boy/Color)"

    @property
    def supported_buttons(self) -> List[str]:
        return ["up", "down", "left", "right", "a", "b", "start", "select"]

    @property
    def has_release(self) -> bool:
        return True

    def verify_controls(self) -> bool:
        screen_a = np.array(self.get_screenshot())
        self.send_input("start")
        self.tick(10)
        screen_b = np.array(self.get_screenshot())
        self.send_input("release_start")
        self.tick(5)
        diff = np.mean(np.abs(screen_a - screen_b))
        return diff > 0.1

    def get_screenshot(self) -> Image.Image:
        return self.pyboy.screen.image

    def send_input(self, button: str):
        if button in self.button_map:
            self.pyboy.send_input(self.button_map[button])
        else:
            raise ValueError(f"Unknown button: {button}")

    def manage_checkpoint(self, action: str, slot: int):
        import os
        from autogameplayer.core.config import settings
        save_dir = settings.saves_dir
        os.makedirs(save_dir, exist_ok=True)
        state_file = save_dir / f"state_slot_{slot}.state"
        
        if action == "save":
            with open(state_file, "wb") as f:
                self.pyboy.save_state(f)
        elif action == "load":
            with open(state_file, "rb") as f:
                self.pyboy.load_state(f)

    def tick(self, frames: int = 1):
        for _ in range(frames):
            self.pyboy.tick(render=True)

    def read_memory(self, address: int) -> int:
        # Step once to ensure RAM state reflects the most recent CPU execution
        self.pyboy.tick(render=False)
        return self.pyboy.memory[address]

    def write_memory(self, address: int, value: int):
        self.pyboy.memory[address] = value

    def read_memory_block(self, address: int, length: int) -> bytes:
        # Step once to ensure RAM state reflects the most recent CPU execution
        self.pyboy.tick(render=False)
        return bytes(self.pyboy.memory[address : address + length])

    def close(self):
        self.pyboy.stop()
