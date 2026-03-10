import os
import json
import asyncio
from typing import List, Dict, Any
from autogameplayer.core.config import settings

class RecordingSession:
    """Manages the lifecycle of a TAS recording."""
    def __init__(self):
        self.is_recording = False
        self.data: List[Dict[str, Any]] = []
        self.start_tick = 0

    def start(self, current_tick: int):
        self.is_recording = True
        self.data = []
        self.start_tick = current_tick

    def record_event(self, current_tick: int, event_type: str, key: str = None, context: dict = None):
        if not self.is_recording:
            return
            
        entry = {
            "tick": current_tick - self.start_tick
        }
        if event_type in ["keydown", "keyup"]:
            entry["type"] = event_type
            entry["key"] = key
        else:
            entry["event"] = event_type
            entry["context"] = context
            
        self.data.append(entry)

    def stop(self, rom_name: str, filename: str = "last_recording.json", start_slot: int = 10, metadata: dict = None) -> str:
        self.is_recording = False
        filepath = settings.datasets_dir / filename
        os.makedirs(filepath.parent, exist_ok=True)
        
        payload = {
            "start_slot": start_slot,
            "rom": rom_name,
            "metadata": metadata or {},
            "events": self.data
        }
        
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
            
        return str(filepath)

    async def replay(self, filename: str, emulator, lock, get_current_tick: callable):
        filepath = settings.datasets_dir / filename
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, "r") as f:
            raw_data = json.load(f)
            events = raw_data["events"]
            start_slot = raw_data.get("start_slot", 10)
            
        with lock:
            emulator.manage_checkpoint("load", start_slot)
            
        start_tick = get_current_tick()
        for event in events:
            while (get_current_tick() - start_tick) < event["tick"]:
                await asyncio.sleep(0.005)
                
            with lock:
                if "type" in event:
                    if event["type"] == "keydown":
                        emulator.send_input(event["key"])
                    else:
                        try:
                            emulator.send_input(f"release_{event['key']}")
                        except Exception: pass
        return True
