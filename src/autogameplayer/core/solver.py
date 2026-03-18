from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from abc import ABC, abstractmethod

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.context import AgentContext


@dataclass
class ActionProposal:
    """
    Standardized output from any decision-making solver.
    Allows ranking and routing logic in ensembles.
    """

    action: Union[
        str, int, Action
    ]  # e.g. "a", "b", "up", "start" or a full Action object
    confidence: float  # 0.0 to 1.0 (1.0 = highly certain / TAS playback)
    reasoning: Optional[str] = None
    source: Optional[str] = None  # Which solver or submodule produced this
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class BaseSolver(ABC):
    """
    A Pluggable Solver that decides the next action based on the state and context.
    """

    @abstractmethod
    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        """
        Evaluate the state and return an ActionProposal.
        """
        pass

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool, outcome: dict = None):
        """
        Callback for online learning or weight updates.
        outcome: contains 'is_stuck', 'next_obs', etc.
        """
        pass

    async def on_episode_end(self, total_reward: float):
        """
        Hook for end-of-episode consolidation or bandit updates.
        """
        pass

    async def reset(self):
        """
        Called when an episode boundaries triggers or hard reset is needed.
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Returns metadata about this solver (e.g. models supported, speed, type).
        """
        return {"name": self.__class__.__name__}

    def transfer_state(self, old_solver: BaseSolver):
        """
        Allows state (LSTM hidden states, etc.) to persist across solver hot-swaps.
        """
        pass
