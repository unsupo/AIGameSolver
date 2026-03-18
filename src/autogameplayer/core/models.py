from pydantic import BaseModel
from typing import List, Optional, Any, Union, Dict
from enum import Enum


class GameStage(str, Enum):
    UI = "ui"
    EXPLORATION = "exploration"
    COMBAT = "combat"
    LOADING = "loading"
    UNKNOWN = "unknown"


class GameState(BaseModel):
    image_data: str
    temporal_image_sequence: Optional[List[str]] = (
        None  # Base64 sequence of last N frames
    )
    width: int = 240
    height: int = 160
    vision_vector: List[float]
    vision_delta: float = 1.0
    stage: GameStage = GameStage.UNKNOWN
    has_dialogue_box: bool = False
    has_dialogue_arrow: bool = False
    is_intro_screen: bool = False
    detected_sprites: List[Dict[str, Any]] = []
    minimap_summary: Optional[str] = None
    ocr_text: Optional[str] = None
    is_skeleton: bool = False
    last_action: Optional[str] = None
    action_history: List[str] = []  # Last 10 actions for hashing
    last_reasoning: Optional[str] = None
    current_plan: Optional[str] = None
    recalled_memories: List[str] = []  # Top RAG snippets used for decision
    hidden_state: Optional[List[float]] = None  # Latent vector for MuZero
    context: Dict[str, Any] = {}  # RAM variables

    model_config = {"extra": "allow"}


class Observation(BaseModel):
    state: GameState
    state_hash: str  # For deduplication
    guidance: Optional[str] = None  # User instructions
    frame_stack: List[List[float]] = []  # Last N encoded vision vectors

    model_config = {"extra": "allow"}


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}

    model_config = {"extra": "allow"}


class Action(BaseModel):
    button: Optional[str] = None
    duration: int = 5
    repeat: int = 1
    until_visual_change: bool = False
    macro: Optional[List["Action"]] = None  # Recursive for sequences
    reasoning: str = ""  # Why the brain chose this
    trigger_vision: Optional[List[float]] = None  # For Macro evaluation
    target_coords: Optional[tuple[int, int]] = None  # For Pathfinding
    save_state: Optional[int] = None
    load_state: Optional[int] = None
    explore_branches: Optional[List[List["Action"]]] = None  # Tree search over states

    # Hacker Tools
    scan_memory: Optional[Union[str, Dict[str, Any]]] = None
    poke_memory: Optional[Dict[str, Any]] = None
    scan_neighborhood: Optional[str] = None
    save_discovered_address: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# To support the recursive macro definition
Action.model_rebuild()
