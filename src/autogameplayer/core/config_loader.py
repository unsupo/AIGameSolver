import yaml
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from autogameplayer.core.curriculum import Checkpoint


class RewardConfig(BaseModel):
    type: str
    category: str = "extrinsic"  # "intrinsic" or "extrinsic"
    params: Dict[str, Any] = {}

    model_config = {"extra": "allow"}


class SolverConfig(BaseModel):
    type: str
    params: Dict[str, Any] = {}
    solvers: Optional[Dict[str, "SolverConfig"]] = None  # For recursive ensembles/routers
    base_solver: Optional["SolverConfig"] = None  # For decorators
    fallback_solver: Optional["SolverConfig"] = None  # For routers

    model_config = {"extra": "allow"}


class RAMLayout(BaseModel):
    """Memory map configuration for a specific game."""

    map_id: Optional[int] = None
    x_pos: Optional[int] = None
    y_pos: Optional[int] = None
    hp: Optional[int] = None
    party_size: Optional[int] = None
    level: Optional[int] = None
    badges: Optional[int] = None

    # Allow unknown fields (like 'level', 'hp', 'badges') to be parsed into the model
    model_config = {"extra": "allow"}


class DialogueArrowSignal(BaseModel):
    enabled: bool = False
    resolution_w: int = 160
    resolution_h: int = 144
    left: int = 128
    top: int = 120
    right: int = 140
    bottom: int = 140
    color_threshold: int = 50

    model_config = {"extra": "allow"}


class UISignals(BaseModel):
    dialogue_arrow: DialogueArrowSignal = DialogueArrowSignal()
    intro_keywords: List[str] = [
        "GAME FREAK",
        "NINTENDO",
        "PRESS START",
        "PUSH START",
        "NEW GAME",
        "TM",
        "©",
        "POKEMON",
    ]

    model_config = {"extra": "allow"}


class TerminalConditionConfig(BaseModel):
    max_steps: Optional[int] = None
    target_ram_address: Optional[str] = None
    target_ram_value: Optional[int] = None
    target_ocr_regex: Optional[str] = None

    model_config = {"extra": "allow"}


class GameProfile(BaseModel):
    ram: RAMLayout = RAMLayout()
    ui_signals: UISignals = UISignals()
    terminal_conditions: List[TerminalConditionConfig] = []
    action_type: str = "discrete"
    curriculum: List[Checkpoint] = []
    intro_guidance: str = ""
    known_locations: List[str] = []
    maps: Dict[int, str] = {}
    controller_type: str = "gb"  # e.g. "gb", "gba", "nes"
    memory_map_hints: str = ""  # Textual description of RAM addresses for the LLM

    model_config = {"extra": "allow"}


class GameHeuristics(BaseModel):
    """Game-specific heuristics for AI logic."""

    intro_map_ids: List[int] = [0]
    overworld_map_id: int = 1
    auto_pilot_until_map: int = 0

    model_config = {"extra": "allow"}


class GameConfig(BaseModel):
    name: str
    description: str = "a video game"
    emulator: str
    rom: str
    controller: str
    vision: str = "facebook/dinov2-small"
    llm_model: Optional[str] = None
    planner_model: Optional[str] = None
    reflector_model: Optional[str] = None
    rewards: List[RewardConfig]
    brain: str = "random"
    agent_pipeline: Optional[SolverConfig] = None
    steps: int = 1000
    render_delay: float = 0.05
    heuristics: GameHeuristics = GameHeuristics()
    profile_path: Optional[str] = None
    profile: Optional[GameProfile] = None

    # --- NEW: Reward Scheduling ---
    # Shift balance between intrinsic (exploration) and extrinsic (game goals) rewards over time
    reward_schedule: Dict[str, float] = {"intrinsic": 1.0, "extrinsic": 1.0}

    # --- NEW: Information Restriction Toggle ---
    # Options: "full" (RAM + Vision) or "vision_only" (Screen + Buttons only)
    perception_level: str = "full"

    model_config = {"extra": "allow"}


SolverConfig.model_rebuild()
GameConfig.model_rebuild()


def load_game_config(path: str) -> GameConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    config = GameConfig(**data)

    # Load profile if specified
    if config.profile_path:
        full_profile_path = Path(path).parent / config.profile_path
        if not full_profile_path.exists():
            # Try relative to base_dir
            from autogameplayer.core.config import settings

            full_profile_path = settings.base_dir / config.profile_path

        if full_profile_path.exists():
            with open(full_profile_path, "r") as f:
                profile_data = yaml.safe_load(f)
                config.profile = GameProfile(**profile_data)
        else:
            print(f"⚠️ Warning: Profile not found at {config.profile_path}")

    return config


def discover_roms(
    rom_dir: str = None, extensions: List[str] = [".gb", ".gbc", ".gba"]
) -> List[str]:
    """Finds all ROMs with specified extensions in the directory and root."""
    from autogameplayer.core.config import settings

    path = Path(rom_dir) if rom_dir else settings.roms_dir
    roms = []
    patterns = [f"*{ext}" for ext in extensions]

    if path.exists():
        for pattern in patterns:
            roms.extend(list(path.glob(pattern)))

    for pattern in patterns:
        roms.extend(list(Path(".").glob(pattern)))

    return [str(rom) for rom in roms]


def require_rom(rom_dir: str = None) -> str:
    """Finds a ROM or exits with a helpful error message."""
    roms = discover_roms(rom_dir)
    if not roms:
        from autogameplayer.core.config import settings

        print(f"❌ Error: No ROM found in {rom_dir or settings.roms_dir} or root.")
        print(
            f"💡 Please place a .gb or .gba file in the {rom_dir or settings.roms_dir} directory."
        )
        sys.exit(1)
    return roms[0]
