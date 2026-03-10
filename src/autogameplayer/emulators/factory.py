from autogameplayer.core.registry import Registry
import autogameplayer.emulators.pyboy_emulator
import autogameplayer.emulators.mgba_core

def create_emulator(rom_path: str):
    """Factory method using the Registry pattern."""
    return Registry.create_emulator(rom_path)
