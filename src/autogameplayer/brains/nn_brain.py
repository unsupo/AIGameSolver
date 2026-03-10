import numpy as np
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.registry import Registry
from autogameplayer.core.config import settings

@Registry.register_brain("evolution")
class NeuralNetworkBrain(Brain):
    """A brain driven by a simple neural network that can be evolved or persisted."""
    def __init__(self, controller: Controller, genome: np.ndarray = None, model_name: str = "nn_best_weights.npy"):
        self.controller = controller
        self.buttons = self.controller.buttons
        self.output_size = len(self.buttons)
        self.weights = None
        self.initial_genome = genome
        self.save_path = settings.models_dir / model_name

    def _lazy_init(self, input_size: int):
        """Initialize weights from genome, disk, or randomly."""
        if self.initial_genome is not None:
            try:
                self.weights = self.initial_genome.reshape((input_size, self.output_size))
            except ValueError:
                self.weights = np.random.randn(input_size, self.output_size)
        elif self.save_path.exists():
            try:
                print(f"🧠 Loading persisted NN weights from {self.save_path}")
                self.weights = np.load(self.save_path)
                if self.weights.shape != (input_size, self.output_size):
                    print("⚠️ Weight shape mismatch. Re-initializing.")
                    self.weights = np.random.randn(input_size, self.output_size)
            except Exception:
                self.weights = np.random.randn(input_size, self.output_size)
        else:
            # Random initialization
            self.weights = np.random.randn(input_size, self.output_size)

    def save(self):
        """Persist weights to disk for use across sessions."""
        if self.weights is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.save_path, self.weights)
            print(f"💾 Persisted NN weights to {self.save_path}")

    async def act(self, observation: Observation, mcp_client=None) -> Action:

        # Convert vision vector to numpy
        x = np.array(observation.state.vision_vector)
        
        if self.weights is None:
            self._lazy_init(len(x))
            
        # Simple linear layer: y = xW
        logits = np.dot(x, self.weights)
        
        # Softmax-ish selection
        button_idx = np.argmax(logits)
        button = self.buttons[button_idx]
        
        return Action(button=button, duration=5)

    def learn(self, state, action, reward, next_state):
        """Placeholder for online learning. GA brains primarily learn through evolution."""
        # This can be expanded with simple Hebbian learning or backprop if desired.
        if getattr(self, "_learn_warned", False) == False:
            print("ℹ️ NeuralNetworkBrain: Online learn() is a no-op. Use GeneticTrainer to evolve weights.")
            self._learn_warned = True
        pass
