def frames_to_seconds(frames: int, fps: int = 60) -> float:
    """Converts a number of game frames to wall-clock seconds."""
    return frames / float(fps)
