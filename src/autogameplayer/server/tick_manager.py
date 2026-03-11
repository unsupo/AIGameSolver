import time
import threading
from autogameplayer.utils.timing import frames_to_seconds

class TickManager:
    """Manages the emulator's execution loop and tick synchronization."""
    def __init__(self, emulator, lock):
        self.emulator = emulator
        self.lock = lock
        self.total_ticks = 0
        self.running = True
        
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def tick(self, frames: int = 1):
        """Manual tick injection."""
        with self.lock:
            self.emulator.tick(frames)
            self.total_ticks += frames

    def _run_loop(self):
        frame_time = frames_to_seconds(1)
        while self.running:
            start_t = time.perf_counter()
            with self.lock:
                self.emulator.tick(1)
                self.total_ticks += 1
            elapsed = time.perf_counter() - start_t
            sleep_t = frame_time - elapsed
            if sleep_t > 0: time.sleep(sleep_t)
