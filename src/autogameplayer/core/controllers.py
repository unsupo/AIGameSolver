from typing import List, Dict, Any
from autogameplayer.core.interfaces import Controller
from autogameplayer.core.registry import Registry

STANDARD_BUTTONS = ["up", "down", "left", "right", "a", "b", "start", "select"]
GBA_BUTTONS = STANDARD_BUTTONS + ["l", "r"]


def build_button_map(
    event_class: Any, direction_prefix: str, buttons: List[str]
) -> Dict[str, Any]:
    """Helper to build a button map from a WindowEvent class and prefix."""
    mapping = {}
    directions = ["up", "down", "left", "right"]

    for b in buttons:
        b_upper = b.upper()
        if b in directions:
            # e.g., PRESS_ARROW_UP, RELEASE_ARROW_UP
            mapping[b] = getattr(event_class, f"PRESS_{direction_prefix}_{b_upper}")
            mapping[f"release_{b}"] = getattr(
                event_class, f"RELEASE_{direction_prefix}_{b_upper}"
            )
        elif b in ["l", "r"]:
            # Special case for GBA shoulder buttons in pyboy-advance
            suffix = "LEFT" if b == "l" else "RIGHT"
            mapping[b] = getattr(
                event_class, f"PRESS_SHOULDER_{suffix}", None
            ) or getattr(event_class, f"PRESS_BUTTON_{b_upper}")
            mapping[f"release_{b}"] = getattr(
                event_class, f"RELEASE_SHOULDER_{suffix}", None
            ) or getattr(event_class, f"RELEASE_BUTTON_{b_upper}")
        else:
            # e.g., PRESS_BUTTON_A, RELEASE_BUTTON_A
            mapping[b] = getattr(event_class, f"PRESS_BUTTON_{b_upper}")
            mapping[f"release_{b}"] = getattr(event_class, f"RELEASE_BUTTON_{b_upper}")
    return mapping


@Registry.register_controller("gb")
@Registry.register_controller("standard")
class StandardController(Controller):
    @property
    def buttons(self) -> List[str]:
        return STANDARD_BUTTONS


@Registry.register_controller("gba")
class GBAController(StandardController):
    @property
    def buttons(self) -> List[str]:
        return GBA_BUTTONS


@Registry.register_controller("dynamic")
class DynamicController(Controller):
    """A controller that queries the active emulator for its button list."""

    def __init__(self):
        self._buttons = []

    def sync_with_emulator(self, emulator):
        """Dynamically pulls buttons from the active core."""
        if hasattr(emulator, "supported_buttons"):
            self._buttons = emulator.supported_buttons

    async def sync_with_client(self, mcp_client):
        """Dynamically pulls buttons from the MCP server."""
        try:
            import json

            raw_caps = await mcp_client.call_tool("get_capabilities")
            caps = json.loads(raw_caps)
            self._buttons = caps.get("supported_buttons", [])
        except Exception:
            pass

    @property
    def buttons(self) -> List[str]:
        return self._buttons
