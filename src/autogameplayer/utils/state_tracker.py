import hashlib
import json
import numpy as np
from typing import Set
from autogameplayer.core.models import GameState

class StateTracker:
    """Tracks visited states to support deduplication and exploration."""
    def __init__(self, precision: int = 2):
        self.visited_hashes: Set[str] = set()
        self.precision = precision # Rounding helps group similar-enough frames

    def get_hash(self, state: GameState) -> str:
        """Creates a stable fingerprint using a combination of vision, OCR, and context."""
        # 1. Vision Fingerprint (Rounded vector for spatial stability)
        vision_part = np.round(state.vision_vector, self.precision).tolist()
        
        # 2. OCR Fingerprint (Normalized text)
        ocr_part = (state.ocr_text or "").strip().upper()
        
        # 3. Spatial Context (Map and Coordinates)
        ctx = state.context
        spatial_part = f"{ctx.get('map_id', 0)}_{ctx.get('x', 0)}_{ctx.get('y', 0)}"
        
        # Combine into a stable signature
        fingerprint = {
            "v": vision_part,
            "o": ocr_part,
            "s": spatial_part
        }
        
        return hashlib.md5(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

    def is_new(self, state_hash: str) -> bool:
        return state_hash not in self.visited_hashes

    def mark_visited(self, state_hash: str):
        self.visited_hashes.add(state_hash)

    def reset(self):
        self.visited_hashes.clear()
