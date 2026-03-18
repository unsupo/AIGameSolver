import hashlib
import json
import numpy as np
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
from autogameplayer.core.models import GameState


class StateTracker:
    """Tracks visited states to support deduplication and exploration."""

    def __init__(self, precision: int = 2, max_size: int = 10000):
        self.visited_hashes: OrderedDictType[str, None] = OrderedDict()
        self.precision = precision  # Rounding helps group similar-enough frames
        self.max_size = max_size

    def get_hash(self, state: GameState) -> str:
        """Creates a stable fingerprint using a combination of vision, OCR, and context."""
        # 1. Vision Fingerprint (Rounded vector for spatial stability)
        # Increased precision to 3 for better visual differentiation
        vision_part = np.round(state.vision_vector, 3).tolist()

        # 2. OCR Fingerprint (Normalized text)
        ocr_part = (state.ocr_text or "").strip().upper()

        # --- FEATURE: Functional State Hashing (Dialogue Ignoring) ---
        # If we are in a dialogue, we treat the state as stable spatially
        # by excluding coordinates. This stops the "Dead End" logic from
        # thinking we are stuck at (3,6) when we are just talking.
        ctx = state.context
        is_dialogue = ctx.get("is_dialogue", False)

        effective_ocr = ocr_part
        if is_dialogue:
            # Blind to specific words, and EXCLUDE coordinates
            effective_ocr = "DIALOGUE_MODE"
            spatial_part = f"{ctx.get('map_id', 0)}_DIALOGUE"
        else:
            # Standard Spatial Context
            spatial_part = f"{ctx.get('map_id', 0)}_{ctx.get('x', 0)}_{ctx.get('y', 0)}"
        # -------------------------------------------------------------

        # Combine into a stable signature
        history_part = "|".join(state.action_history[-10:]) if state.action_history else "none"
        fingerprint = {
            "v": vision_part,
            "o": effective_ocr,
            "s": spatial_part,
            "a": history_part,
        }

        # Add hidden state if available (from MuZero plan phase)
        if state.hidden_state:
            fingerprint["h"] = np.round(state.hidden_state, 2).tolist()

        return hashlib.md5(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

    def is_new(self, state_hash: str) -> bool:
        return state_hash not in self.visited_hashes

    def mark_visited(self, state_hash: str):
        self.visited_hashes[state_hash] = None
        self.visited_hashes.move_to_end(state_hash)
        if len(self.visited_hashes) > self.max_size:
            self.visited_hashes.popitem(last=False)

    def reset(self):
        self.visited_hashes.clear()
