import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from autogameplayer.core.interfaces import BaseEmulator
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.core.config import settings

class RAMContextExtractor(ABC):
    @abstractmethod
    def extract(self, emulator: BaseEmulator) -> Dict[str, Any]:
        pass

class ProfileDrivenExtractor(RAMContextExtractor):
    """Uses GameProfile RAM layout to extract context."""
    def __init__(self, config: GameConfig):
        self.config = config
        self.layout = config.profile.ram if config.profile else None

    def extract(self, emulator: BaseEmulator) -> Dict[str, Any]:
        if not self.layout:
            return {}
            
        try:
            ctx = {}
            if self.layout.map_id: ctx["map_id"] = emulator.read_memory(self.layout.map_id)
            if self.layout.x_pos: ctx["x"] = emulator.read_memory(self.layout.x_pos)
            if self.layout.y_pos: ctx["y"] = emulator.read_memory(self.layout.y_pos)
            
            # 1. Load Raw Extra Addresses
            for key, addr in self.layout.extra.items():
                ctx[key] = emulator.read_memory(addr)
                
            # 2. Derive Common Game Flags (Agnostic Logic based on common patterns)
            
            # --- Dialogue & Menu Logic ---
            is_dialogue_ram = ctx.get("is_dialogue_ram", 0) > 0
            # wUpdateSpritesEnabled (0xCFC4): 1 in Overworld, 0 in Menus/Intro
            sprites_enabled = ctx.get("sprites_enabled", 1) 
            is_overworld_active = (sprites_enabled == 1)
            
            # Additional check for Pokemon Red: if textbox or menu ID is active, 
            # we are definitely NOT in explorable overworld.
            textbox_active = ctx.get("textbox_flag", 0) > 0
            menu_id_active = ctx.get("menu_id_ram", 0) > 0
            
            if textbox_active or menu_id_active:
                is_overworld_active = False

            # max_menu_item (0xCC28) can stay '1' after a menu closes. 
            # We MUST check is_overworld_active first.
            max_menu_item = ctx.get("max_menu_item", 0)
            is_menu_open = (not is_overworld_active) and (max_menu_item > 0 or menu_id_active)
            
            # If the overworld is hidden, and it's not a standard menu, and not standard dialogue
            # we are in a cutscene or naming screen!
            is_intro_cutscene = (not is_overworld_active) and (not is_menu_open) and (not is_dialogue_ram)
            
            # Update derived UI flags
            ctx["is_dialogue"] = is_dialogue_ram or is_intro_cutscene or textbox_active
            ctx["is_menu_open"] = is_menu_open
            ctx["is_overworld_active"] = is_overworld_active
            
            # --- Engine Lock Logic ---
            joy_ignore = ctx.get("joy_ignore", 0) > 0
            joy_simulation = ctx.get("joy_simulation", 0) > 0
            script_flags = ctx.get("script_flags", 0)
            # Bits 0, 5, 6, 7 are often associated with scripted events or busy engine
            is_script_busy = (script_flags & 0xE1) != 0
            
            ctx["is_engine_locked"] = joy_ignore or joy_simulation or is_script_busy
            
            # --- Money (BCD Decoding) ---
            if "money_byte_1" in ctx and "money_byte_2" in ctx and "money_byte_3" in ctx:
                def bcd_to_int(bcd):
                    try: return int(f"{bcd:02x}")
                    except ValueError: return 0
                money = (bcd_to_int(ctx["money_byte_1"]) * 10000) + \
                        (bcd_to_int(ctx["money_byte_2"]) * 100) + \
                        bcd_to_int(ctx["money_byte_3"])
                ctx["money"] = money

            # --- Player Named Check ---
            if "first_name_char" in ctx:
                ctx["is_player_named"] = ctx["first_name_char"] not in [0x00, 0x50]
            
            # --- Facing Direction ---
            if "player_facing" in ctx:
                directions = {0: "DOWN", 4: "UP", 8: "LEFT", 12: "RIGHT"}
                ctx["facing"] = directions.get(ctx["player_facing"], "UNKNOWN")
                
            # --- HP & Battle Aggregation ---
            # FIX: Multi-byte HP Extraction
            if "slot_1_hp_high" in ctx and "slot_1_hp_low" in ctx:
                ctx["hp"] = (ctx["slot_1_hp_high"] * 256) + ctx["slot_1_hp_low"]
            elif self.layout.hp: 
                ctx["hp"] = emulator.read_memory(self.layout.hp)

            if "battle_state" in ctx:
                if ctx["battle_state"] > 0:
                    if "enemy_hp_high" in ctx and "enemy_hp_low" in ctx:
                        ctx["enemy_hp"] = (ctx["enemy_hp_high"] * 256) + ctx["enemy_hp_low"]
                    if "player_active_hp_high" in ctx and "player_active_hp_low" in ctx:
                        ctx["player_active_hp"] = (ctx["player_active_hp_high"] * 256) + ctx["player_active_hp_low"]
            
            if "party_count" in ctx and ctx["party_count"] > 0:
                if "slot_1_hp_high" in ctx and "slot_1_hp_low" in ctx:
                    ctx["slot_1_hp"] = (ctx["slot_1_hp_high"] * 256) + ctx["slot_1_hp_low"]
                if "slot_1_max_hp_high" in ctx and "slot_1_max_hp_low" in ctx:
                    ctx["slot_1_max_hp"] = (ctx["slot_1_max_hp_high"] * 256) + ctx["slot_1_max_hp_low"]
                if "slot_1_level" in ctx:
                    ctx["level"] = ctx["slot_1_level"]

            # --- Progression ---
            if "badges_byte" in ctx:
                ctx["badges_count"] = bin(ctx["badges_byte"]).count('1')
            if "pokedex_flag" in ctx:
                ctx["has_pokedex"] = (ctx["pokedex_flag"] & 0x20) != 0

            # --- 3. Symbolic State Abstraction (Agnostic Mapping) ---
            # This creates a high-level representation for the LLM.
            symbolic = {
                "location": f"Map #{ctx.get('map_id', '?')}",
                "coordinates": f"({ctx.get('x', '?')}, {ctx.get('y', '?')})",
                "status": "EXPLORABLE (Overworld)" if ctx.get("is_overworld_active") else "LOCKED (Menu/Dialogue)",
                "facing": ctx.get("facing", "UNKNOWN")
            }
            
            # Map Tile Types if available
            if "tile_under_player" in ctx:
                tile_val = ctx["tile_under_player"]
                # Common GB RPG patterns
                tile_map = {0x52: "TALL GRASS", 0x01: "INDOOR FLOOR", 0x20: "STAIRS", 0x15: "LEDGE", 0x08: "WALL"}
                symbolic["current_tile"] = tile_map.get(tile_val, f"GROUND (0x{tile_val:02x})")
            
            # Add symbolic state to the context for agents
            ctx["symbolic"] = symbolic

            return ctx
        except Exception:
            # print(f"⚠️ ProfileDrivenExtractor error: {e}")
            return {}

class DynamicContextExtractor(RAMContextExtractor):
    """Combines profile-driven extraction with dynamically discovered RAM addresses."""
    def __init__(self, config: GameConfig):
        self.config = config
        self.profile_extractor = ProfileDrivenExtractor(config)
        self.discovery_path = settings.models_dir / "discovered_ram.json"
        self.discovered_map = {}
        self.last_map_check = 0

    def _load_discovery(self):
        # Cache for performance, check disk every 5 seconds
        if time.time() - self.last_map_check < 5:
            return
            
        self.last_map_check = time.time()
        if self.discovery_path.exists():
            try:
                with open(self.discovery_path, "r") as f:
                    self.discovered_map = json.load(f)
            except Exception: pass

    def extract(self, emulator: BaseEmulator) -> Dict[str, Any]:
        # 1. Start with profile-driven extraction
        ctx = self.profile_extractor.extract(emulator)
        
        # 2. Add dynamically discovered variables
        self._load_discovery()
        for name, info in self.discovered_map.items():
            try:
                # Support both simple address strings and dicts with 'address' key
                addr_str = info["address"] if isinstance(info, dict) else info
                addr = int(addr_str, 16)
                ctx[name] = emulator.read_memory(addr)
            except (ValueError, KeyError, TypeError):
                pass
                
        return ctx

def get_extractor(config: GameConfig) -> RAMContextExtractor:
    return DynamicContextExtractor(config)
