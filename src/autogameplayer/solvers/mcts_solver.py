import torch
import numpy as np
from typing import Any, List, Dict

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry
from autogameplayer.core.config import settings
from autogameplayer.vision.encoder import VisionEncoder

from autogameplayer.muzero.networks import MuZeroModel
from autogameplayer.muzero.mcts import MCTS
from autogameplayer.muzero.imagination import MentalSimulator


@SolverRegistry.register("mcts")
class MCTSSolver(BaseSolver):
    """
    MuZero-inspired MCTS solver using Representation, Dynamics, and Prediction networks.
    """

    def __init__(
        self,
        supported_buttons: List[str],
        input_dim=None,
        hidden_dim=256,
        model_name="muzero_weights.pth",
        num_simulations=50,
    ):
        self.buttons = supported_buttons
        self.action_dim = len(self.buttons)
        self.current_personality = 0

        if input_dim is None:
            input_dim = VisionEncoder.get_dim(settings.vision_model)

        # Core MuZero model components
        self.model = MuZeroModel(
            input_dim=input_dim, hidden_dim=hidden_dim, action_dim=self.action_dim
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Persistent state
        self.current_hidden_state = None
        self.last_map_id = None

        # Engines
        self.mcts = MCTS(self.model, num_simulations=num_simulations)
        self.dreamer = MentalSimulator(self.model)

        self.save_path = settings.models_dir / model_name
        self.shared_weights_path = settings.models_dir / "muzero_weights_shared.pth"
        self.steps_since_sync = 0
        self.sync_interval = 100

        self._load_weights()

    def _load_weights(self):
        if self.shared_weights_path.exists():
            try:
                self.model.load_state_dict(
                    torch.load(self.shared_weights_path, map_location=self.device)
                )
                return
            except Exception:
                pass
        if self.save_path.exists():
            try:
                self.model.load_state_dict(
                    torch.load(self.save_path, map_location=self.device)
                )
            except Exception:
                pass

    def _sync_weights_if_needed(self):
        self.steps_since_sync += 1
        if self.steps_since_sync >= self.sync_interval:
            if self.shared_weights_path.exists():
                try:
                    self.model.load_state_dict(
                        torch.load(self.shared_weights_path, map_location=self.device)
                    )
                except Exception:
                    pass
            self.steps_since_sync = 0

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        if not obs.state.vision_vector:
            return ActionProposal(
                action=Action(button=None, reasoning="No vision."),
                confidence=0.0,
                reasoning="No vision vector available."
            )

        current_map = obs.state.context.get("map_id", -1)
        if self.last_map_id is not None and current_map != self.last_map_id:
            self.current_hidden_state = None
        self.last_map_id = current_map

        self._sync_weights_if_needed()

        obs_tensor = torch.tensor(obs.state.vision_vector, dtype=torch.float32).to(
            self.device
        )

        if self.current_hidden_state is None:
            with torch.no_grad():
                self.current_hidden_state, _, _ = self.model.initial_inference(
                    obs_tensor.unsqueeze(0), personality_id=self.current_personality
                )

        # 1. Search
        action_idx, search_probs = self.mcts.search(
            initial_state=self.current_hidden_state
        )

        obs.state.context["search_statistics"] = search_probs.tolist()
        obs.state.hidden_state = self.current_hidden_state.cpu().numpy()[0].tolist()

        # 2. Dynamics Unroll
        with torch.no_grad():
            action_onehot = (
                torch.nn.functional.one_hot(
                    torch.tensor([action_idx]), num_classes=self.action_dim
                )
                .float()
                .to(self.device)
            )
            if action_onehot.dim() == 1:
                action_onehot = action_onehot.unsqueeze(0)
            self.current_hidden_state, _, _, _ = self.model.dynamics(
                self.current_hidden_state, action_onehot
            )

        button = self.buttons[action_idx]
        confidence = float(np.max(search_probs))

        return ActionProposal(
            action=Action(button=button, reasoning=f"MCTS selection: {button}"),
            confidence=confidence,
            reasoning=f"MCTS predicted {button} with {confidence:.2f} confidence.",
            metadata={"mcts_probs": search_probs.tolist(), "solver_name": "MCTSSolver"},
        )

    async def reset(self):
        self.current_hidden_state = None
        self.last_map_id = None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "device": str(self.device),
            "simulations": self.mcts.num_simulations,
        }
