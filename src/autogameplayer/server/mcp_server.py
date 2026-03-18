import argparse
import logging
import os
import asyncio
from fastmcp import FastMCP

from autogameplayer.core.config_loader import load_game_config, GameConfig
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.core.config import settings
from .game_session import GameSession
from .ws_server import WebSocketServer

import json
import time
from typing import Set, Optional, Any


class MemoryDetective:
    def __init__(self, start_addr: int = None, end_addr: int = None):
        # Ranges are initialized by update_range_for_platform or provided directly
        self.start_addr = start_addr
        self.end_addr = end_addr
        self.last_snapshot = bytearray()
        self.search_results: Set[int] = set()

        if self.start_addr is not None and self.end_addr is not None:
            self.search_results = set(range(self.start_addr, self.end_addr))

    def update_range_for_platform(self, emulator_name: str):
        """Adjusts detection range based on whether we are playing GB or GBA."""
        if "gba" in emulator_name.lower():
            self.start_addr, self.end_addr = settings.gba_memory_range
        else:
            self.start_addr, self.end_addr = settings.gb_memory_range

        self.search_results = set(range(self.start_addr, self.end_addr))
        self.last_snapshot = bytearray()
        print(
            f"🔍 MemoryDetective: Range set to {hex(self.start_addr)}-{hex(self.end_addr)} for {emulator_name}"
        )

    def take_snapshot(self, emulator):
        """Grabs the current state of the RAM."""
        self.last_snapshot = bytearray(
            [emulator.read_memory(i) for i in range(self.start_addr, self.end_addr)]
        )

    def filter_ram(self, emulator, condition: str, value: Optional[int] = None):
        """Filters the RAM addresses based on how they changed since the snapshot."""
        current_ram = bytearray(
            [emulator.read_memory(i) for i in range(self.start_addr, self.end_addr)]
        )
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


class SequenceHandler:
    """Tracks sequence numbers to prevent duplicate execution of tools."""
    def __init__(self):
        self.last_seq = -1
        self.responses = {} # seq -> last_result

    def is_duplicate(self, seq: int) -> bool:
        if seq is None:
            return False
        return seq <= self.last_seq

    def update(self, seq: int, result: Any):
        if seq is not None:
            self.last_seq = max(self.last_seq, seq)
            self.responses[seq] = result
            # Prune old responses
            if len(self.responses) > 100:
                min_seq = min(self.responses.keys())
                del self.responses[min_seq]


def create_server(
    rom_path: str,
    vision_encoder: VisionEncoder = None,
    port: int = 8000,
    config: Optional[GameConfig] = None,
) -> FastMCP:
    mcp = FastMCP("Universal Game Nexus")
    session = GameSession(rom_path, vision_encoder=vision_encoder, config=config)
    detective = MemoryDetective()
    seq_handler = SequenceHandler()

    # Auto-adjust memory range based on core
    detective.update_range_for_platform(session.emulator.emulator_name)

    # Start the WebSocket server for frame streaming
    ws_server = WebSocketServer(session, port)
    ws_server.start()

    @mcp.tool()
    def scan_memory(condition: str, exact_value: int = None, _seq: int = None) -> Any:
        """
        TOOL FOR AI: Use this to discover RAM addresses.
        Conditions: 'snapshot' (start tracking), 'decreased', 'increased', 'changed', 'exact'.
        """
        if condition == "snapshot":
            detective.take_snapshot(session.emulator)
            detective.search_results = set(
                range(detective.start_addr, detective.end_addr)
            )
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
    def save_discovered_address(
        name: str, hex_address: str, description: str, _seq: int = None
    ) -> Any:
        """Saves a discovered memory address to the dynamic context JSON."""
        discovery_path = settings.models_dir / "discovered_ram.json"
        os.makedirs(discovery_path.parent, exist_ok=True)

        data = {}
        if discovery_path.exists():
            try:
                with open(discovery_path, "r") as f:
                    data = json.load(f)
            except Exception:
                pass

        data[name] = {"address": hex_address, "description": description}

        with open(discovery_path, "w") as f:
            json.dump(data, f, indent=2)

        return f"Successfully mapped {name} to {hex_address}."

    @mcp.tool()
    def poke_memory(address: str, value: int, _seq: int = None) -> Any:
        """
        OVERWRITE RAM: Use this to verify theories.
        Example: If you think 0xD016 is HP, write 255 to it and see if your health fills up.
        """
        try:
            addr = (
                int(address, 16)
                if isinstance(address, str) and address.startswith("0x")
                else int(address)
            )
            with session.lock:
                session.emulator.write_memory(addr, value & 0xFF)
            return f"Successfully wrote {value & 0xFF} to {hex(addr)}."
        except Exception as e:
            return f"Failed to poke memory: {e}"

    @mcp.tool()
    def scan_neighborhood(address: str, radius: int = 5, _seq: int = None) -> Any:
        """
        Returns the values of memory addresses surrounding a target.
        Hacker Trick: Similar variables (like X and Y) are usually stored next to each other.
        """
        try:
            addr = (
                int(address, 16)
                if isinstance(address, str) and address.startswith("0x")
                else int(address)
            )
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
    def get_capabilities(_seq: int = None) -> Any:
        """Returns the active core name and its supported buttons."""
        with session.lock:
            return {
                "emulator": session.emulator.emulator_name,
                "supported_buttons": session.emulator.supported_buttons,
                "rom": os.path.basename(session.rom_path),
            }

    @mcp.tool()
    def send_input(
        button: str = None,
        duration: int = 10,
        reasoning: str = "",
        repeat: int = 1,
        macro: list = None,
        _seq: int = None,
    ):
        """Sends a button press or macro sequence to the emulator."""
        if seq_handler.is_duplicate(_seq):
            return seq_handler.responses.get(_seq, "Duplicate Action")
            
        result = session.send_input(button, duration, reasoning, repeat, macro)
        seq_handler.update(_seq, result)
        return result

    @mcp.tool()
    def read_memory_range(address: int, length: int, _seq: int = None) -> Any:
        """Reads a range of memory and returns it as a hex string."""
        with session.lock:
            data = session.emulator.read_memory_block(address, length)
            return data.hex()

    @mcp.tool()
    def get_game_state(include_ocr: bool = False, _seq: int = None) -> Any:
        return session.get_game_state(include_ocr=include_ocr).model_dump()

    @mcp.tool()
    def manage_checkpoint(action: str, slot: int, _seq: int = None) -> Any:
        res = session.manage_checkpoint(action, slot)

        # --- TAS AUTO-RECORDING LOGIC ---
        if action == "load":
            session.last_load_slot = slot
            # Automatically start recording when loading a known-good milestone state
            if slot == settings.tas_trigger_slot:
                session.recording.start(session.total_ticks)
                print(f"🎬 TAS: Auto-started recording from Slot {slot}")
        elif action == "save":
            # Automatically stop and save recording when achieving a new milestone
            if slot == settings.tas_trigger_slot and session.recording.is_recording:
                filename = f"milestone_tas_{int(time.time())}.json"
                start_slot = getattr(
                    session, "last_load_slot", settings.tas_trigger_slot
                )
                path = session.recording.stop(
                    os.path.basename(session.rom_path),
                    filename,
                    start_slot=start_slot,
                    metadata={"reason": "Automated Milestone Achievement"},
                )
                print(f"🎬 TAS: Auto-stopped and saved recording to {path}")
        # --------------------------------

        return res

    @mcp.tool()
    def tick(frames: int = 1, _seq: int = None) -> Any:
        with session.lock:
            session.emulator.tick(frames)
        return "Ticked"

    @mcp.tool()
    def read_memory(address: int, _seq: int = None) -> Any:
        with session.lock:
            return session.emulator.read_memory(address)

    @mcp.tool()
    def set_guidance(message: str, _seq: int = None) -> Any:
        return session.set_guidance(message)

    @mcp.tool()
    def get_guidance(_seq: int = None) -> Any:
        return session.get_guidance()

    @mcp.tool()
    def set_plan(plan: str, _seq: int = None) -> Any:
        return session.set_plan(plan)

    @mcp.tool()
    def set_recalled_memories(memories: list[str], _seq: int = None) -> Any:
        """Sets the currently recalled RAG memories for dashboard display."""
        return session.set_recalled_memories(memories)

    @mcp.tool()
    def get_recalled_memories(_seq: int = None) -> Any:
        return session.get_recalled_memories()

    @mcp.tool()
    def list_waypoints(_seq: int = None) -> Any:

        """Returns all discovered spatial anchors (teleport locations) on the current map."""
        try:
            from autogameplayer.utils.database import get_db_connection
            db_path = settings.models_dir / "long_term_memory.db"
            
            # Get current map from emulator state
            state = session.get_game_state(include_ocr=False)
            curr_map = state.context.get("map_id", -1)

            with get_db_connection(db_path) as conn:
                conn.row_factory = json.dumps # Or dict
                import sqlite3
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM spatial_anchors WHERE map_id = ? ORDER BY timestamp DESC", 
                    (curr_map,)
                )
                anchors = [dict(row) for row in cursor.fetchall()]
                
                if not anchors:
                    return f"No waypoints discovered on Map #{curr_map} yet."
                
                return json.dumps(anchors, indent=2)
        except Exception as e:
            return f"Failed to list waypoints: {e}"

    @mcp.tool()
    def navigate_to(x: int, y: int, _seq: int = None):
        """Sets a spatial navigation goal. The AI will attempt to find a path using A*."""
        return session.set_navigation_goal(x, y)

    @mcp.tool()
    def start_recording(_seq: int = None):
        session.recording.start(session.total_ticks)
        return "Started"

    @mcp.tool()
    def stop_recording(
        filename: str = "user_play.json",
        start_slot: int = 10,
        metadata: dict = None,
        _seq: int = None,
    ):
        return session.recording.stop(
            os.path.basename(session.rom_path),
            filename,
            start_slot=start_slot,
            metadata=metadata,
        )

    @mcp.tool()
    async def replay_recording(filename: str = "user_play.json", _seq: int = None):
        success = await session.recording.replay(
            filename, session.emulator, session.lock, lambda: session.total_ticks
        )
        return "Replay Completed" if success else f"Failed to find or replay {filename}"

    @mcp.tool()
    def health_check(_seq: int = None):
        return {"status": "ok", "rom": os.path.basename(rom_path)}

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
    except KeyboardInterrupt, asyncio.CancelledError:
        pass


if __name__ == "__main__":
    main()
