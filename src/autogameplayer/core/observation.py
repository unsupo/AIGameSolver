import base64
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Optional
from collections import deque

from autogameplayer.core.models import GameState, GameStage
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.core.interfaces import BaseEmulator
from autogameplayer.core.context import RAMContextExtractor
from autogameplayer.vision.annotator import ImageAnnotator
from autogameplayer.vision.detector import SemanticDetector

class ObservationPipeline:
    """Refactored pipeline adhering to SRP."""
    def __init__(self, vision_encoder: Optional[VisionEncoder] = None, context_extractor: RAMContextExtractor = None):
        self.vision_encoder = vision_encoder
        self.context_extractor = context_extractor
        self.last_screenshot = None
        self.last_vision_vector = None
        self.frame_history = deque(maxlen=3) # Store last 3 base64 frames for temporal context

    def _detect_dialogue_arrow(self, screenshot: Image.Image) -> bool:
        """Heuristic to detect the 'Dialogue Finished' flashing arrow in Pokemon."""
        img = screenshot.convert("L") # Grayscale
        w, h = img.size
        
        if w == 160 and h == 144:
            region = img.crop((128, 120, 140, 140))
            if np.min(np.array(region)) < 50:
                return True
        return False

    def _detect_intro_screen(self, screenshot: Image.Image, context: dict, vision_delta: float, ocr_text: Optional[str]) -> bool:
        """Classifies if the current screen is a title/intro screen."""
        map_id = context.get("map_id", -1)
        intro_maps = context.get("intro_map_ids", [0])
        
        is_static = vision_delta < 0.05
        is_intro_map = (map_id in intro_maps) or (map_id == -1)
        
        ocr_upper = (ocr_text or "").upper()
        intro_keywords = ["GAME FREAK", "NINTENDO", "PRESS START", "PUSH START", "NEW GAME", "TM", "©", "POKEMON"]
        has_intro_text = any(k in ocr_upper for k in intro_keywords)
        
        if is_intro_map:
            if is_static or has_intro_text:
                return True
        
        if map_id == 0 and is_static:
            return True
            
        return False

    def process(self, emulator: BaseEmulator, include_ocr: bool = False) -> GameState:
        # 1. Capture Raw Frame
        screenshot = emulator.get_screenshot()
        
        # 2. Extract Memory Context
        context = self.context_extractor.extract(emulator) if self.context_extractor else {}
        
        if hasattr(self.context_extractor, 'config') and self.context_extractor.config.heuristics:
            context["intro_map_ids"] = self.context_extractor.config.heuristics.intro_map_ids

        # Determine Interface Mode
        is_dialogue = context.get("is_dialogue", False)
        is_menu_open = context.get("is_menu_open", False)
        is_start_menu = context.get("is_start_menu", False)
        battle_state = context.get("battle_state", 0)
        
        if battle_state > 0:
            stage = GameStage.BATTLE
        elif is_dialogue or is_menu_open:
            stage = GameStage.MENU
        else:
            stage = GameStage.OVERWORLD
        
        if is_dialogue:
            context["interface_mode"] = "UI_LOCKED (Dialogue)"
            context["progression_hint"] = "Progress by pressing A or START to clear text."
        elif is_menu_open:
            menu_type = "START MENU" if is_start_menu else "Generic Menu"
            context["interface_mode"] = f"UI_LOCKED ({menu_type})"
            context["progression_hint"] = "Use D-Pad to select options and A to confirm. Press START/B to close."
        else:
            context["interface_mode"] = "EXPLORABLE (Overworld)"
            context["progression_hint"] = "Use D-Pad to move. Interact with objects using A."

        # 3. Annotate Image
        annotated_img = screenshot.copy().convert("RGB")
        ImageAnnotator.draw_crosshair(annotated_img)
        ImageAnnotator.annotate_state(annotated_img, context)
        
        # 4. Extract Vision Vector
        vision_vector = np.zeros(384)
        if self.vision_encoder:
            vision_vector = self.vision_encoder.encode(annotated_img)
            
        vision_delta = 1.0
        if self.last_vision_vector is not None:
            vision_delta = float(np.linalg.norm(self.last_vision_vector - vision_vector))
        self.last_vision_vector = vision_vector

        # 5. Extract OCR
        ocr_text = None
        has_dialogue_box = False
        has_dialogue_arrow = self._detect_dialogue_arrow(screenshot)
        
        force_ocr = is_dialogue or is_menu_open or battle_state > 0
        
        if include_ocr or force_ocr:
            try:
                import pytesseract
                ocr_text = pytesseract.image_to_string(screenshot).strip()
                if ocr_text and len(ocr_text) > 5:
                    has_dialogue_box = True
            except (ImportError, Exception):
                pass

        is_intro_screen = self._detect_intro_screen(screenshot, context, vision_delta, ocr_text)
        if is_intro_screen and stage != GameStage.BATTLE:
            stage = GameStage.TITLE

        detected_sprites = SemanticDetector.detect_sprites(screenshot)
        minimap_summary = SemanticDetector.summarize_minimap(screenshot)

        # 6. Encode for Transport
        width, height = annotated_img.size
        buffered = BytesIO()
        annotated_img.save(buffered, format="JPEG", quality=60)
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Update Frame History
        self.frame_history.append(image_data)

        return GameState(
            image_data=image_data,
            temporal_image_sequence=list(self.frame_history),
            width=width,
            height=height,
            vision_vector=vision_vector.tolist(),
            vision_delta=vision_delta,
            stage=stage,
            has_dialogue_box=has_dialogue_box,
            has_dialogue_arrow=has_dialogue_arrow,
            is_intro_screen=is_intro_screen,
            detected_sprites=detected_sprites,
            minimap_summary=minimap_summary,
            ocr_text=ocr_text,
            is_skeleton=getattr(emulator, "experimental", False),
            context=context
        )
