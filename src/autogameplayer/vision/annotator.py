from PIL import Image, ImageDraw


class ImageAnnotator:
    """Handles visual overlays on game screenshots for AI perception in a game-agnostic way."""

    @staticmethod
    def draw_crosshair(image: Image.Image, color: str = "red"):
        """Draws a target crosshair in the center of the image."""
        width, height = image.size
        cx, cy = width // 2, height // 2
        draw = ImageDraw.Draw(image)
        draw.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], outline=color)
        return image

    @staticmethod
    def annotate_state(image: Image.Image, context: dict, tile_size: int = 8):
        """Draw semantic labels on the image based on local RAM grid."""
        local_grid = context.get("local_grid")
        if not local_grid:
            return image

        width, height = image.size
        draw = ImageDraw.Draw(image)

        # Grid is 5x5 around player. Player is at (8, 7) in tiles for GB.
        # We need a more agnostic way to find the player position.
        # For now, let's assume the center of the screen is the player if not specified.
        # Or better, use tile-based coords if available.

        # Heuristic: Find [P] in the grid
        rows = local_grid.split("\n")
        player_row_idx = -1
        player_col_idx = -1
        for r_idx, row in enumerate(rows):
            if "[P]" in row:
                player_row_idx = r_idx
                # tiles are separated by ][
                tiles = (
                    row.replace("][", ",").replace("[", "").replace("]", "").split(",")
                )
                for c_idx, t in enumerate(tiles):
                    if t == "P":
                        player_col_idx = c_idx
                        break
                break

        if player_row_idx == -1:
            return image  # No player found in grid

        # We assume the player tile [P] is in the middle of the screen
        screen_cx, screen_cy = width // 2, height // 2

        # Player tile top-left in pixels
        player_px = screen_cx - (tile_size // 2)
        player_py = screen_cy - (tile_size // 2)

        for dy, row in enumerate(rows):
            tiles = row.replace("][", ",").replace("[", "").replace("]", "").split(",")
            for dx, tile_type in enumerate(tiles):
                # Pixel position relative to player
                px = player_px + (dx - player_col_idx) * tile_size
                py = player_py + (dy - player_row_idx) * tile_size

                color = None
                if tile_type == "W":
                    color = "red"
                elif tile_type == "D":
                    color = "blue"
                elif tile_type == "P":
                    color = "green"

                if color:
                    draw.rectangle(
                        [px, py, px + tile_size - 1, py + tile_size - 1],
                        outline=color,
                        width=1,
                    )

        # Highlight Dialogue/Interface area
        if context.get("is_dialogue"):
            # Generic box at bottom for dialogue
            draw.rectangle(
                [2, height - 40, width - 2, height - 2], outline="cyan", width=2
            )
            draw.text((width // 2 - 30, height - 50), "INTERFACE", fill="cyan")

        # Add Map ID and Coords text overlay
        map_id = context.get("map_id")
        if map_id is not None:
            draw.text(
                (5, 5),
                f"MAP: {map_id} ({context.get('x', '?')},{context.get('y', '?')})",
                fill="yellow",
            )

        return image
