import time
import queue
import threading
import numpy as np
from autogameplayer.utils.timing import frames_to_seconds


class InputManager:
    """Handles the sequential processing of game inputs, including macros and repeats."""

    def __init__(self, emulator, obs_pipeline, recording, lock, get_tick_count=None):
        self.emulator = emulator
        self.obs_pipeline = obs_pipeline
        self.recording = recording
        self.lock = lock
        self.get_tick_count = get_tick_count or (lambda: 0)
        self.input_queue = queue.Queue()
        self.running = True

        # Status tracked for dashboard/state
        self.last_action = "None"
        self.last_reasoning = "Initializing..."
        self.last_repeat = 1

        self.thread = threading.Thread(target=self._input_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.input_queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def queue_input(
        self,
        button: str,
        duration: int = 10,
        reasoning: str = "",
        repeat: int = 1,
        macro: list = None,
    ):
        self.input_queue.put((button, duration, reasoning, repeat, macro))
        return f"Queued {button or 'macro'} x{repeat}"

    def _input_loop(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=1.0)
                if item is None:
                    break

                button, duration, reasoning, repeat, macro = item

                if macro:
                    self._execute_macro(macro, reasoning)
                elif button:
                    self._execute_button_press(button, duration, reasoning, repeat)

                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Error in input loop: {e}")

    def _execute_macro(self, macro, reasoning):
        self.last_action = "Macro Sequence"
        self.last_reasoning = reasoning or "Executing multi-step macro..."

        for step in macro:
            btn = step.get("button")
            frames = step.get("frames", step.get("duration", 10))
            until_visual_change = step.get("until_visual_change", False)

            prev_vec = None
            if until_visual_change:
                with self.lock:
                    prev_state = self.obs_pipeline.process(
                        self.emulator, include_ocr=False
                    )
                    prev_vec = np.array(prev_state.vision_vector)

            if btn:
                self._press_single_button(btn, frames)

                if until_visual_change and prev_vec is not None:
                    if not self._check_visual_change(prev_vec, btn):
                        break
            else:
                time.sleep(frames_to_seconds(frames))

    def _execute_button_press(self, button, duration, reasoning, repeat):
        self.last_action = button
        self.last_repeat = repeat
        self.last_reasoning = (
            reasoning
            if (reasoning and str(reasoning).strip())
            else "Executing action sequence..."
        )

        for _ in range(repeat):
            self._press_single_button(button, duration)
            if repeat > 1:
                time.sleep(frames_to_seconds(1))

    def _press_single_button(self, button, frames):
        tick = self.get_tick_count()
        with self.lock:
            self.recording.record_event(tick, "keydown", key=button)
            self.emulator.send_input(button)

        time.sleep(frames_to_seconds(frames))

        if self.emulator.has_release:
            with self.lock:
                try:
                    self.recording.record_event(
                        self.get_tick_count(), "keyup", key=button
                    )
                    self.emulator.send_input(f"release_{button}")
                except ValueError:
                    pass

    def _check_visual_change(self, prev_vec, btn):
        with self.lock:
            curr_state = self.obs_pipeline.process(self.emulator, include_ocr=False)
            curr_vec = np.array(curr_state.vision_vector)
            dist = np.linalg.norm(prev_vec - curr_vec)
            if dist < 0.05:
                print(
                    f"🛑 Macro Interrupted: No visual change detected for button {btn.upper()}."
                )
                return False
        return True
