import threading
import time
import queue
import numpy as np
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.emulators.factory import create_emulator
from autogameplayer.core.models import GameState
from autogameplayer.core.observation import ObservationPipeline
from autogameplayer.core.context import get_extractor
from autogameplayer.core.recording import RecordingSession

class GameSession:
    """Manages the lifecycle and state of a single game emulation session."""
    def __init__(self, rom_path: str, vision_encoder: VisionEncoder = None, config: GameConfig = None):
        self.rom_path = rom_path
        self.config = config
        self.lock = threading.Lock()
        self.running = True
        self.total_ticks = 0
        
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
        self.last_action = "None (Waiting for AI)"
        self.last_reasoning = "The AI system is initializing..."
        self.current_plan = "Initializing strategic overseer..."
        self.last_load_slot = 1
        
        print(f"🎮 Initializing Emulator with ROM: {rom_path}...", flush=True)
        self.emulator = create_emulator(rom_path)
        
        self.tick_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.tick_thread.start()

        self.input_queue = queue.Queue()
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

    def _input_loop(self):
        """Sequentially processes inputs from the queue."""
        while self.running:
            try:
                # Use timeout so we can check self.running periodically
                item = self.input_queue.get(timeout=1.0)
                if item is None: break
                
                button, duration, reasoning, repeat, macro = item
                
                # Execute as a macro if provided
                if macro:
                    self.last_action = "Macro Sequence"
                    self.last_reasoning = reasoning or "Executing multi-step macro..."
                    
                    for step in macro:
                        btn = step.get('button')
                        # Duration in frames, default 10
                        frames = step.get('frames', step.get('duration', 10))
                        until_visual_change = step.get('until_visual_change', False)
                        
                        # Store vision vector BEFORE for visual change detection
                        prev_vec = None
                        if until_visual_change:
                            with self.lock:
                                # Quick visual fingerprint
                                prev_state = self.obs_pipeline.process(self.emulator, include_ocr=False)
                                prev_vec = np.array(prev_state.vision_vector)

                        if btn:
                            with self.lock:
                                self.recording.record_event(self.total_ticks, "keydown", key=btn)
                                self.emulator.send_input(btn)
                            
                            time.sleep(frames / 60.0)
                            
                            if self.emulator.has_release:
                                with self.lock:
                                    try:
                                        self.recording.record_event(self.total_ticks, "keyup", key=btn)
                                        self.emulator.send_input(f"release_{btn}")
                                    except ValueError: pass
                            
                            # --- FEATURE: Reactive Macro Breaking ---
                            if until_visual_change and prev_vec is not None:
                                with self.lock:
                                    curr_state = self.obs_pipeline.process(self.emulator, include_ocr=False)
                                    curr_vec = np.array(curr_state.vision_vector)
                                    # Calculate distance/change
                                    dist = np.linalg.norm(prev_vec - curr_vec)
                                    if dist < 0.05:
                                        print(f"🛑 Macro Interrupted: No visual change detected for button {btn.upper()}.")
                                        break # Exit the macro loop early
                            # -----------------------------------------
                        else:
                            # Just a wait/delay
                            time.sleep(frames / 60.0)
                
                elif button:
                    # Update status variables when processing actually begins
                    self.last_action = button
                    self.last_repeat = repeat
                    if reasoning and str(reasoning).strip():
                        self.last_reasoning = reasoning
                    else:
                        self.last_reasoning = "Executing action sequence..."

                    # Execute the button press sequence
                    for _ in range(repeat):
                        with self.lock:
                            self.recording.record_event(self.total_ticks, "keydown", key=button)
                            self.emulator.send_input(button)

                        time.sleep(duration / 60.0)
                        
                        if self.emulator.has_release:
                            with self.lock:
                                try:
                                    self.recording.record_event(self.total_ticks, "keyup", key=button)
                                    self.emulator.send_input(f"release_{button}")
                                except ValueError: pass
                        
                        # Small gap between repeats
                        if repeat > 1: time.sleep(1/60.0)
                
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Error in input loop: {e}")

    def _run_loop(self):
        frame_time = 1.0 / 60.0
        while self.running:
            start_t = time.perf_counter()
            with self.lock:
                self.emulator.tick(1)
                self.total_ticks += 1
            elapsed = time.perf_counter() - start_t
            sleep_t = frame_time - elapsed
            if sleep_t > 0: time.sleep(sleep_t)

    def get_game_state(self, include_ocr: bool = False) -> GameState:
        with self.lock:
            state = self.obs_pipeline.process(self.emulator, include_ocr=include_ocr)
        state.last_action = self.last_action
        state.last_reasoning = self.last_reasoning
        state.current_plan = self.current_plan
        # Ensure last_repeat is in the context for the dashboard
        if not hasattr(self, 'last_repeat'): self.last_repeat = 1
        state.context["last_repeat"] = self.last_repeat
        return state

    def send_input(self, button: str, duration: int = 10, reasoning: str = "", repeat: int = 1, macro: list = None):
        """Queues a button press to be processed sequentially."""
        self.input_queue.put((button, duration, reasoning, repeat, macro))
        return f"Queued {button or 'macro'} x{repeat}"

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
