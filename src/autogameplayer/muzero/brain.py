import torch
import numpy as np
from typing import Optional, Any, List
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.registry import Registry
from autogameplayer.core.config import settings
from autogameplayer.muzero.networks import MuZeroModel
from autogameplayer.muzero.mcts import MCTS
from autogameplayer.muzero.imagination import MentalSimulator
from autogameplayer.vision.encoder import VisionEncoder


@Registry.register_brain("muzero")
class MuZeroBrain(Brain):
    """
    MuZero-inspired brain using Representation, Dynamics, and Prediction networks.
    Uses Monte Carlo Tree Search (MCTS) for simulated exploration and deep tactical play.
    """

    def __init__(
        self,
        controller: Controller,
        input_dim=None,
        hidden_dim=256,
        model_name="muzero_weights.pth",
        num_simulations=50,
    ):
        self.controller = controller
        input_dim = input_dim or VisionEncoder.get_dim(settings.vision_model)
        self.buttons = self.controller.buttons
        self.action_dim = len(self.buttons)
        self.current_personality = 0

        # Core MuZero model components
        self.model = MuZeroModel(
            input_dim=input_dim, hidden_dim=hidden_dim, action_dim=self.action_dim
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Persistent state for Temporal Latent Consistency (Recurrence)
        self.current_hidden_state = None
        self.last_map_id = None

        # MCTS engine for deep search
        self.mcts = MCTS(self.model, num_simulations=num_simulations)
        self.last_search_result = None

        # Mental Simulator for internal 'dreams'
        self.dreamer = MentalSimulator(self.model)

        self.save_path = settings.models_dir / model_name
        self.shared_weights_path = settings.models_dir / "muzero_weights_shared.pth"
        self.steps_since_sync = 0
        self.sync_interval = 100  # Reload weights every 100 steps

        self._load_weights()

    def _load_weights(self):
        """Attempts to load weights from disk if they exist."""
        # Prefer shared weights if they exist (background learner output)
        if self.shared_weights_path.exists():
            try:
                print(f"🔄 Actor: Syncing weights from {self.shared_weights_path}")
                self.model.load_state_dict(
                    torch.load(self.shared_weights_path, map_location=self.device)
                )
                return
            except Exception as e:
                print(f"⚠️ Failed to load shared weights: {e}")

        if self.save_path.exists():
            try:
                print(f"🧠 Loading MuZero weights from {self.save_path}")
                self.model.load_state_dict(
                    torch.load(self.save_path, map_location=self.device)
                )
            except Exception as e:
                print(f"⚠️ Failed to load weights: {e}")

    def _sync_weights_if_needed(self):
        """Reloads weights from the background learner periodically."""
        self.steps_since_sync += 1
        if self.steps_since_sync >= self.sync_interval:
            if self.shared_weights_path.exists():
                try:
                    # Use a non-blocking load (map_location handles device)
                    self.model.load_state_dict(
                        torch.load(self.shared_weights_path, map_location=self.device)
                    )
                    # print(f"🔄 Actor: Fresh weights loaded from {self.shared_weights_path}")
                except Exception:
                    pass  # Silently fail if file is being written to
            self.steps_since_sync = 0

    def save(self):
        """Persists the model weights to disk."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"💾 Persisted MuZero weights to {self.save_path}")

    def get_hidden_state(self, observation: Observation) -> Optional[List[float]]:
        """Produces the MuZero latent vector for a given observation."""
        if not observation.state.vision_vector:
            return None
        obs_tensor = torch.tensor(
            observation.state.vision_vector, dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            h, _, _ = self.model.initial_inference(
                obs_tensor.unsqueeze(0), personality_id=self.current_personality
            )
            return h.cpu().numpy()[0].tolist()

    async def plan(self, observation: Observation, mcp_client: Optional[Any] = None):
        """Perform MCTS search as a discrete plan phase using recurrent persistence."""
        if not observation.state.vision_vector:
            return

        # Grounding/Reset logic: detects if we've moved to a new map (checkpoint/teleport)
        current_map = observation.state.map_id
        if self.last_map_id is not None and current_map != self.last_map_id:
            print(
                f"🔄 Map changed ({self.last_map_id} -> {current_map}), resetting MuZero latent state."
            )
            self.current_hidden_state = None
        self.last_map_id = current_map

        # Periodically sync weights from background trainer
        self._sync_weights_if_needed()

        obs_tensor = torch.tensor(
            observation.state.vision_vector, dtype=torch.float32
        ).to(self.device)

        # Initialize state from representation if needed
        if self.current_hidden_state is None:
            with torch.no_grad():
                self.current_hidden_state, _, _ = self.model.initial_inference(
                    obs_tensor.unsqueeze(0), personality_id=self.current_personality
                )

        # MCTS returns (best_action_idx, search_probs)
        action_idx, search_probs = self.mcts.search(
            initial_state=self.current_hidden_state
        )
        self.last_search_result = (action_idx, search_probs)

        # Store search statistics in the observation context for memory recording
        observation.state.context["search_statistics"] = search_probs.tolist()
        
        # --- FEATURE: Policy Entropy (Confidence Metric) ---
        # Entropy = -sum(p * log(p))
        probs = np.array(search_probs)
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        observation.state.context["policy_entropy"] = float(entropy)
        # ----------------------------------------------------

        # Enrich the observation with the recurrent hidden state for StateTracker
        observation.state.hidden_state = (
            self.current_hidden_state.cpu().numpy()[0].tolist()
        )

    async def act(
        self, observation: Observation, mcp_client: Optional[Any] = None
    ) -> Action:
        """
        Executes the planned action and updates the persistent recurrent state.
        """
        # Ensure we have a vision vector
        if not observation.state.vision_vector:
            return self.fallback_action(self.controller)

        # Ensure we have a plan for the current state
        if not self.last_search_result:
            await self.plan(observation, mcp_client)

        action_idx, search_probs = self.last_search_result
        self.last_search_result = None  # Consume it

        # IMPORTANT: Update persistent latent state using the chosen action (Dynamics Unroll)
        with torch.no_grad():
            action_onehot = (
                torch.nn.functional.one_hot(
                    torch.tensor([action_idx]), num_classes=self.action_dim
                )
                .float()
                .to(self.device)
            )
            # Ensure action_onehot has batch dimension
            if action_onehot.dim() == 1:
                action_onehot = action_onehot.unsqueeze(0)
            self.current_hidden_state, _, _, _ = self.model.dynamics(
                self.current_hidden_state, action_onehot
            )

        button = self.buttons[action_idx]

        return Action(
            button=button,
            duration=5,
            reasoning=f"Recurrent MCTS visits: {np.round(search_probs, 2).tolist()}",
        )

    def imagine(self, hidden_state: torch.Tensor, actions: List[int]) -> List[float]:
        """
        Utility to project future rewards/values without stepping the emulator.
        """
        rewards = []
        current_h = hidden_state

        with torch.no_grad():
            for action_idx in actions:
                # Recurrent inference: g(h, a) -> next_h, r, map_l, ocr_l; phi(next_h) -> p, v
                next_h, reward, map_l, ocr_l, policy_logits, value = (
                    self.model.recurrent_inference(current_h, action_idx)
                )
                rewards.append(reward.item())
                current_h = next_h

        return rewards

    def learn(self, state, action, reward, next_state):
        """Placeholder for online learning or backprop."""
        pass
