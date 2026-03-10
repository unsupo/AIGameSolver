import argparse
import logging
import os
import asyncio
from fastmcp import FastMCP

from autogameplayer.core.config_loader import load_game_config
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.core.models import GameState
from autogameplayer.core.config import settings
from .game_session import GameSession
from .ws_server import WebSocketServer

import json
import time
from typing import Set, Optional

class MemoryDetective:
    def __init__(self, start_addr: int = 0xC000, end_addr: int = 0xE000):
        self.start_addr = start_addr
        self.end_addr = end_addr
        self.last_snapshot = bytearray()
        self.search_results: Set[int] = set(range(start_addr, end_addr))

    def take_snapshot(self, emulator):
        """Grabs the current state of the RAM."""
        self.last_snapshot = bytearray([emulator.read_memory(i) for i in range(self.start_addr, self.end_addr)])

    def filter_ram(self, emulator, condition: str, value: Optional[int] = None):
        """Filters the RAM addresses based on how they changed since the snapshot."""
        current_ram = bytearray([emulator.read_memory(i) for i in range(self.start_addr, self.end_addr)])
        new_results = set()

        if not self.last_snapshot:
            self.last_snapshot = current_ram
            return list(self.search_results)

        for i in self.search_results:
            idx = i - self.start_addr
            old_val = self.last_snapshot[idx]
            new_val = current_ram[idx]

            if condition == "decreased" and new_val < old_val:
                new_results.add(i)
            elif condition == "increased" and new_val > old_val:
                new_results.add(i)
            elif condition == "changed" and new_val != old_val:
                new_results.add(i)
            elif condition == "exact" and new_val == value:
                new_results.add(i)

        self.search_results = new_results
        self.last_snapshot = current_ram
        return list(self.search_results)

def create_server(rom_path: str, vision_encoder: VisionEncoder = None, port: int = 8000, config: None = None) -> FastMCP:
    mcp = FastMCP("Universal Game Nexus")
    session = GameSession(rom_path, vision_encoder=vision_encoder, config=config)
    detective = MemoryDetective()
    
    # Start the WebSocket server for frame streaming
    ws_server = WebSocketServer(session, port)
    ws_server.start()

    @mcp.tool()
    def scan_memory(condition: str, exact_value: int = None) -> str:
        """
        TOOL FOR AI: Use this to discover RAM addresses.
        Conditions: 'snapshot' (start tracking), 'decreased', 'increased', 'changed', 'exact'.
        """
        if condition == "snapshot":
            detective.take_snapshot(session.emulator)
            detective.search_results = set(range(detective.start_addr, detective.end_addr))
            return f"Snapshot taken. {len(detective.search_results)} addresses tracking. Do an action, then scan again."
        
        results = detective.filter_ram(session.emulator, condition, exact_value)
        
        if len(results) > 50:
            return f"Filtered down to {len(results)} possible addresses. Too many to list. Do another action and scan again."
        elif len(results) == 0:
            return "0 addresses match this condition. You filtered too much. Take a new 'snapshot' to reset."
        else:
            hex_results = [hex(addr) for addr in sorted(results)]
            return f"Found {len(results)} candidate addresses: {hex_results}"

    @mcp.tool()
    def save_discovered_address(name: str, hex_address: str, description: str) -> str:
        """Saves a discovered memory address to the dynamic context JSON."""
        discovery_path = settings.models_dir / "discovered_ram.json"
        os.makedirs(discovery_path.parent, exist_ok=True)
        
        data = {}
        if discovery_path.exists():
            try:
                with open(discovery_path, "r") as f:
                    data = json.load(f)
            except Exception: pass
            
        data[name] = {"address": hex_address, "description": description}
        
        with open(discovery_path, "w") as f:
            json.dump(data, f, indent=2)
            
        return f"Successfully mapped {name} to {hex_address}."

    @mcp.tool()
    def poke_memory(address: str, value: int) -> str:
        """
        OVERWRITE RAM: Use this to verify theories. 
        Example: If you think 0xD016 is HP, write 255 to it and see if your health fills up.
        """
        try:
            addr = int(address, 16) if isinstance(address, str) and address.startswith("0x") else int(address)
            with session.lock:
                session.emulator.write_memory(addr, value & 0xFF)
            return f"Successfully wrote {value & 0xFF} to {hex(addr)}."
        except Exception as e:
            return f"Failed to poke memory: {e}"

    @mcp.tool()
    def scan_neighborhood(address: str, radius: int = 5) -> str:
        """
        Returns the values of memory addresses surrounding a target.
        Hacker Trick: Similar variables (like X and Y) are usually stored next to each other.
        """
        try:
            addr = int(address, 16) if isinstance(address, str) and address.startswith("0x") else int(address)
            start = addr - radius
            length = (radius * 2) + 1
            
            with session.lock:
                data = session.emulator.read_memory_block(start, length)
            
            results = {}
            for i, val in enumerate(data):
                results[hex(start + i)] = val
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Failed to scan neighborhood: {e}"

    @mcp.tool()
    def get_capabilities() -> dict:
        """Returns the active core name and its supported buttons."""
        with session.lock:
            return {
                "emulator": session.emulator.emulator_name,
                "supported_buttons": session.emulator.supported_buttons,
                "rom": os.path.basename(session.rom_path)
            }

    @mcp.tool()
    def send_input(button: str = None, duration: int = 10, reasoning: str = "", repeat: int = 1, macro: list = None):
        """Sends a button press or macro sequence to the emulator."""
        return session.send_input(button, duration, reasoning, repeat, macro)

    @mcp.tool()
    def read_memory_range(address: int, length: int) -> str:
        """Reads a range of memory and returns it as a hex string."""
        with session.lock:
            data = session.emulator.read_memory_block(address, length)
            return data.hex()

    @mcp.tool()
    def get_game_state(include_ocr: bool = False) -> GameState:
        return session.get_game_state(include_ocr=include_ocr)

    @mcp.tool()
    def manage_checkpoint(action: str, slot: int):
        res = session.manage_checkpoint(action, slot)
        
        # --- TAS AUTO-RECORDING LOGIC ---
        if action == "load":
            session.last_load_slot = slot
            # Automatically start recording when loading a known-good milestone state
            if slot == 1:
                session.recording.start(session.total_ticks)
                print(f"🎬 TAS: Auto-started recording from Slot {slot}")
        elif action == "save":
            # Automatically stop and save recording when achieving a new milestone
            if slot == 1 and session.recording.is_recording:
                filename = f"milestone_tas_{int(time.time())}.json"
                start_slot = getattr(session, 'last_load_slot', 1)
                path = session.recording.stop(
                    os.path.basename(session.rom_path), 
                    filename,
                    start_slot=start_slot,
                    metadata={"reason": "Automated Milestone Achievement"}
                )
                print(f"🎬 TAS: Auto-stopped and saved recording to {path}")
        # --------------------------------
        
        return res

    @mcp.tool()
    def tick(frames: int = 1):
        with session.lock: session.emulator.tick(frames)
        return "Ticked"

    @mcp.tool()
    def read_memory(address: int) -> int:
        with session.lock:
            return session.emulator.read_memory(address)

    @mcp.tool()
    def set_guidance(message: str): return session.set_guidance(message)

    @mcp.tool()
    def get_guidance() -> str: return session.get_guidance()

    @mcp.tool()
    def set_plan(plan: str): return session.set_plan(plan)

    @mcp.tool()
    def start_recording():
        session.recording.start(session.total_ticks)
        return "Started"

    @mcp.tool()
    def stop_recording(filename: str = "user_play.json", start_slot: int = 10, metadata: dict = None):
        return session.recording.stop(os.path.basename(session.rom_path), filename, start_slot=start_slot, metadata=metadata)

    @mcp.tool()
    async def replay_recording(filename: str = "user_play.json"):
        success = await session.recording.replay(
            filename, 
            session.emulator, 
            session.lock, 
            lambda: session.total_ticks
        )
        return "Replay Completed" if success else f"Failed to find or replay {filename}"

    @mcp.tool()
    def health_check(): return {"status": "ok", "rom": os.path.basename(rom_path)}

    return mcp

def main():
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str)
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--transport", type=str, default=settings.transport)
    parser.add_argument("--port", type=int, default=settings.server_port)
    args = parser.parse_args()
    
    rom_path = args.rom or settings.rom_path
    config = load_game_config(args.config) if args.config else None
    if config and not rom_path:
        rom_path = config.rom
        
    mcp = create_server(str(rom_path), port=args.port, config=config)
    try: 
        mcp.run(transport=args.transport, port=args.port)
    except (KeyboardInterrupt, asyncio.CancelledError): 
        pass

if __name__ == "__main__":
    main()
