from PIL import Image, ImageFilter
from typing import List, Dict, Any


class SemanticDetector:
    """Handles game-agnostic semantic vision: sprite detection, text localization, and minimap analysis."""

    @staticmethod
    def detect_sprites(image: Image.Image) -> List[Dict[str, Any]]:
        """Identifies potential sprites using basic contour/edge detection (agnostic)."""
        # Convert to grayscale and apply edge detection
        gray = image.convert("L")
        gray.filter(ImageFilter.FIND_EDGES)

        # Simple heuristic: sprites are usually concentrated high-contrast areas
        # In a production system, we might use a small YOLO or YOLO-tiny here.
        # For now, we'll return an empty list or very basic bounding boxes.
        return []

    @staticmethod
    def summarize_minimap(image: Image.Image) -> str:
        """Analyzes common minimap regions (corners) to describe the immediate surroundings."""
        w, h = image.size
        # Minimaps are usually in a corner. We'll check the top-right 25% area.
        # Check for color distribution (Green = Grass, Blue = Water, Gray = Road)
        tr_corner = image.crop((w * 0.7, 0, w, h * 0.3))
        colors = tr_corner.getcolors(maxcolors=256)

        if not colors:
            return "No minimap detected."

        # Sort by frequency
        colors.sort(key=lambda x: x[0], reverse=True)
        dominant_color = colors[0][1]

        # Agnostic mapping of colors to terrain types
        if (
            dominant_color[1] > dominant_color[0]
            and dominant_color[1] > dominant_color[2]
        ):
            return "Surroundings: Primarily vegetation/greenery."
        elif (
            dominant_color[2] > dominant_color[0]
            and dominant_color[2] > dominant_color[1]
        ):
            return "Surroundings: Primarily water/blue area."
        else:
            return "Surroundings: Urban or indoor environment."
