import json
from pathlib import Path
from typing import List, Dict, Any
from autogameplayer.core.models import Action, Observation

class DatasetLogger:
    """Logs episode transitions for metrics and replay datasets."""
    def __init__(self, log_dir: str = "datasets"):
        self.log_path = Path(log_dir)
        self.log_path.mkdir(exist_ok=True)
        self.current_episode: List[Dict[str, Any]] = []

    def log_step(self, obs: Observation, action: Action, reward: float):
        """Records a single step."""
        self.current_episode.append({
            "vision_hash": obs.state_hash,
            "vision_vector": obs.state.vision_vector,
            "action": action.model_dump(),
            "reward": reward,
            "ocr": obs.state.ocr_text,
            "context": obs.state.context
        })

    def save_episode(self, episode_id: str):
        """Writes the current episode to disk."""
        filename = self.log_path / f"episode_{episode_id}.json"
        with open(filename, "w") as f:
            json.dump(self.current_episode, f)
        self.current_episode = []

class ReplaySystem:
    """Loads recorded episodes for imitation learning or review."""
    def __init__(self, dataset_dir: str = "datasets"):
        self.dataset_dir = Path(dataset_dir)

    def get_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        filename = self.dataset_dir / f"episode_{episode_id}.json"
        with open(filename, "r") as f:
            return json.load(f)
