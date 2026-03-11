import threading
from autogameplayer.core.config import settings
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.emulators.factory import create_emulator
from autogameplayer.core.models import GameState
from autogameplayer.core.observation import ObservationPipeline
from autogameplayer.core.context import get_extractor
from autogameplayer.core.recording import RecordingSession

from autogameplayer.server.input_manager import InputManager
from autogameplayer.server.tick_manager import TickManager

class GameSession:
    """Manages the lifecycle and state of a single game emulation session."""
    def __init__(self, rom_path: str, vision_encoder: VisionEncoder = None, config: GameConfig = None):
        self.rom_path = rom_path
        self.config = config
        self.lock = threading.Lock()
        self.running = True
        
        # Injected Components
        self.vision_encoder = vision_encoder or VisionEncoder()
        self.context_extractor = get_extractor(config) if config else None
        
        if self.context_extractor is None:
            print("⚠️ WARNING: No context extractor loaded. AI will have no positional awareness.")
            print("💡 Ensure 'profile_path' is set in your YAML and the profile contains RAM addresses.")
        else:
            print("✅ Context extractor loaded successfully.")
            
        self.obs_pipeline = ObservationPipeline(self.vision_encoder, self.context_extractor)
        self.recording = RecordingSession()
        
        # State
        self.guidance_message = ""
        self.current_plan = "Initializing strategic overseer..."
        self.last_load_slot = settings.tas_trigger_slot

        print(f"🎮 Initializing Emulator with ROM: {rom_path}...", flush=True)
        self.emulator = create_emulator(rom_path)
        
        # Delegated Managers
        self.tick_manager = TickManager(self.emulator, self.lock)
        self.input_manager = InputManager(
            self.emulator, 
            self.obs_pipeline, 
            self.recording, 
            self.lock, 
            get_tick_count=lambda: self.tick_manager.total_ticks
        )

    @property
    def total_ticks(self):
        return self.tick_manager.total_ticks

    @property
    def last_action(self):
        return self.input_manager.last_action

    @property
    def last_reasoning(self):
        return self.input_manager.last_reasoning

    @property
    def last_repeat(self):
        return self.input_manager.last_repeat

    def stop(self):
        self.running = False
        self.tick_manager.stop()
        self.input_manager.stop()

    def get_game_state(self, include_ocr: bool = False) -> GameState:
        with self.lock:
            state = self.obs_pipeline.process(self.emulator, include_ocr=include_ocr)
        state.last_action = self.last_action
        state.last_reasoning = self.last_reasoning
        state.current_plan = self.current_plan
        state.context["last_repeat"] = self.last_repeat
        return state

    def send_input(self, button: str, duration: int = 10, reasoning: str = "", repeat: int = 1, macro: list = None):
        """Queues a button press to be processed sequentially via InputManager."""
        return self.input_manager.queue_input(button, duration, reasoning, repeat, macro)

    def manage_checkpoint(self, action: str, slot: int):
        with self.lock:
            self.emulator.manage_checkpoint(action, slot)
        return f"Checkpoint {action}ed"

    def set_guidance(self, message: str):
        self.guidance_message = message
        return f"Guidance set: {message}"

    def get_guidance(self) -> str:
        msg = self.guidance_message
        self.guidance_message = "" 
        return msg

    def set_plan(self, plan: str):
        self.current_plan = plan
        return f"Plan updated: {plan}"
