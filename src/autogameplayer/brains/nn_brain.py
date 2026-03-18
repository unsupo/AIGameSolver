import numpy as np
import torch
import torch.nn.functional as F
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.registry import Registry
from autogameplayer.core.config import settings
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.muzero.networks import MuZeroModel


@Registry.register_brain("evolution")
class NeuralNetworkBrain(Brain):
    """
    Refactored NeuralNetworkBrain using MuZero's tripartite architecture.
    Modularized into Representation, Dynamics, and Prediction networks.

    This brain is designed to be evolved via Genetic Algorithms (GA) by
    flattening the parameters of all three networks into a single genome.
    """

    def __init__(
        self,
        controller: Controller,
        genome: np.ndarray = None,
        input_dim=None,
        hidden_dim=128,
        model_name: str = "nn_tripartite_weights.pth",
    ):
        self.controller = controller
        self.buttons = self.controller.buttons
        self.output_size = len(self.buttons)
        
        if input_dim is None:
            input_dim = VisionEncoder.get_dim(settings.vision_model)
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Core MuZero tripartite model
        self.model = MuZeroModel(
            input_dim=input_dim, hidden_dim=hidden_dim, action_dim=self.output_size
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.save_path = settings.models_dir / model_name
        self.initial_genome = genome
        self._initialized = False

    def _lazy_init(self):
        """Initialize weights from genome, disk, or randomly."""
        if self.initial_genome is not None:
            try:
                # MuZeroModel has a helper to load from flat numpy array
                self.model.load_genome(self.initial_genome)
                # print("🧬 Loaded brain from evolutionary genome.")
            except Exception as e:
                print(f"⚠️ Genome loading failed: {e}. Using random initialization.")
        elif self.save_path.exists():
            try:
                print(f"🧠 Loading persisted tripartite weights from {self.save_path}")
                self.model.load_state_dict(
                    torch.load(self.save_path, map_location=self.device)
                )
            except Exception:
                pass

        self._initialized = True

    @staticmethod
    def get_genome_size(controller: Controller, **kwargs) -> int:
        """Utility for trainers to know the flattened genome size of the tripartite model."""
        input_dim = kwargs.get("input_dim")
        if input_dim is None:
            input_dim = VisionEncoder.get_dim(settings.vision_model)
            
        hidden_dim = kwargs.get("hidden_dim", 128)

        # We need a temp model instance to calculate parameter count
        temp_model = MuZeroModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=len(controller.buttons),
        )
        return temp_model.get_genome_size()

    def save(self):
        """Persist weights to disk for use across sessions."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"💾 Persisted tripartite weights to {self.save_path}")

    async def act(self, observation: Observation, mcp_client=None) -> Action:
        if not observation.state.vision_vector:
            return self.fallback_action(self.controller)

        if not self._initialized:
            self._lazy_init()

        # Convert vision vector to tensor
        x = torch.tensor(observation.state.vision_vector, dtype=torch.float32).to(
            self.device
        )

        with torch.no_grad():
            # Tripartite Inference:
            # 1. Representation (f): observation -> hidden state
            # 2. Prediction (phi): hidden state -> policy logits, value
            # The Dynamics network (g) is not used during direct 'act' but is available for MCTS/Imagination.
            hidden_state, policy_logits, value = self.model.initial_inference(
                x.unsqueeze(0)
            )

            # Use policy network for action selection (argmax for evolution)
            probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            button_idx = np.argmax(probs)
            button = self.buttons[button_idx]

            return Action(button=button, duration=5)

    def learn(self, state, action, reward, next_state):
        """Placeholder for online learning or backprop."""
        # Evolution brains primarily learn through the Genetic Algorithm,
        # but this tripartite structure allows for future hybrid RL/GA approaches.
        pass
