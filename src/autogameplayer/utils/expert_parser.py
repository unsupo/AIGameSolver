import json
import torch
from typing import List, Dict, Any
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.emulators.factory import create_emulator


class ExpertParser:
    """
    Utility to convert TAS (.json) recordings into sequence of latents.
    """
    def __init__(self, vision_encoder: VisionEncoder = None):
        self.vision_encoder = vision_encoder or VisionEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    async def parse_tas(self, tas_path: str, rom_path: str, context_extractor: Any = None) -> Dict[str, torch.Tensor]:
        """
        Replays a TAS and extracts vision latents and SLAM coords.
        """
        with open(tas_path, "r") as f:
            tas_data = json.load(f)
            
        events = tas_data["events"]
        start_slot = tas_data.get("start_slot", 10)
        
        emulator = create_emulator(rom_path)
        emulator.manage_checkpoint("load", start_slot)
        
        latents = []
        coords = [] # List of (map_id, x, y)
        
        # Sort events by tick
        events.sort(key=lambda x: x["tick"])
        
        event_idx = 0
        total_ticks = events[-1]["tick"] + 60
        
        print(f"🎬 Parsing TAS: {total_ticks} ticks...")
        
        for tick in range(total_ticks):
            while event_idx < len(events) and events[event_idx]["tick"] == tick:
                # ... same input processing ...
                event = events[event_idx]
                if "type" in event:
                    if event["type"] == "keydown":
                        emulator.send_input(event["key"])
                    else:
                        try:
                            emulator.send_input(f"release_{event['key']}")
                        except Exception:
                            pass
                event_idx += 1
            
            emulator.tick(1)
            
            if tick % 5 == 0:
                img = emulator.get_screenshot()
                v = self.vision_encoder.encode(img)
                latents.append(torch.tensor(v, dtype=torch.float32))
                
                # Extract SLAM if extractor provided
                if context_extractor:
                    ctx = context_extractor.extract(emulator)
                    coords.append((ctx.get("map_id", -1), ctx.get("x", -1), ctx.get("y", -1)))
                else:
                    coords.append((-1, -1, -1))

        emulator.close()
        return {
            "latents": torch.stack(latents),
            "coords": torch.tensor(coords)
        }

    def save_expert(self, result: Dict[str, torch.Tensor], output_path: str, keyframes: List[int] = None):
        """Saves latents and coords to a .pt file."""
        data = {
            "latents": result["latents"],
            "coords": result["coords"],
            "keyframes": keyframes or []
        }
        torch.save(data, output_path)
        print(f"✅ Expert demo saved to {output_path}")
