import logging
from typing import Dict, Any, Optional
import time


class TelemetryLogger:
    """
    Tracks solver composition telemetry, such as which solver (Macro > MCTS > LLM > Random)
    took an action on a given frame.
    """

    def __init__(self):
        self.logger = logging.getLogger("Telemetry")
        self.stats: Dict[str, Any] = {
            "total_steps": 0,
            "session_start_time": time.time(),
            "solver_usage": {},
        }

    def log_decision(
        self, solver_name: str, confidence: float, reasoning: Optional[str] = None
    ):
        """Logs a decision event made by a specific solver."""
        self.logger.debug(
            f"Decision by {solver_name} (conf: {confidence:.2f}): {reasoning}"
        )
        self.stats["total_steps"] += 1

        if solver_name not in self.stats["solver_usage"]:
            self.stats["solver_usage"][solver_name] = {
                "count": 0,
                "avg_confidence": 0.0,
            }

        usage = self.stats["solver_usage"][solver_name]
        usage["count"] += 1
        # Streaming average approximation
        usage["avg_confidence"] = (
            usage["avg_confidence"]
            + (confidence - usage["avg_confidence"]) / usage["count"]
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Returns the summary of telemetry stats for the session."""
        self.stats["session_duration"] = time.time() - self.stats["session_start_time"]
        return self.stats


telemetry = TelemetryLogger()
