from PIL import Image
from autogameplayer.core.config import settings
import numpy as np


class VisionEncoder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.vision_model
        self.device = "cpu"
        self.processor = None
        self.model = None
        self._initialized = False
        self._failed = False
        self._embedding_dim = None

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim:
            return self._embedding_dim
        return self.get_dim(self.model_name)

    @embedding_dim.setter
    def embedding_dim(self, value: int):
        self._embedding_dim = value

    def _lazy_init(self):
        if self._initialized or self._failed:
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            print(f"  - Loading vision processor: {self.model_name}", flush=True)
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            print(f"  - Loading vision model: {self.model_name}", flush=True)
            self.model = AutoModel.from_pretrained(self.model_name)

            # Extract dimension from model config
            if hasattr(self.model.config, "hidden_size"):
                self.embedding_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, "dim"):
                self.embedding_dim = self.model.config.dim
            else:
                self.embedding_dim = self.get_dim(self.model_name)

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"

            print(f"  - Moving model to {self.device}...", flush=True)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            print(f"  - Vision Encoder ready (Dim: {self.embedding_dim}).", flush=True)
        except ImportError:
            print(
                "⚠️ Vision dependencies (torch/transformers) not installed. Encoder running in passthrough mode."
            )
            self._failed = True
            self.embedding_dim = self.get_dim(self.model_name)
        except Exception as e:
            print(
                f"⚠️ Vision Encoder failed to initialize: {e}. Running in passthrough mode."
            )
            self._failed = True
            self.embedding_dim = self.get_dim(self.model_name)

    @staticmethod
    def get_dim(model_name: str) -> int:
        """Returns the embedding dimension for known models without full initialization."""
        # Hardcoded mapping for performance/offline mode
        mapping = {
            "facebook/dinov2-small": 384,
            "facebook/dinov2-base": 768,
            "facebook/dinov2-large": 1024,
            "facebook/dinov2-giant": 1536,
            "google/vit-base-patch16-224": 768,
            "openai/clip-vit-base-patch32": 512,
        }

        if model_name in mapping:
            return mapping[model_name]

        # Dynamic fallback: try to load config
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, "hidden_size"):
                return config.hidden_size
            elif hasattr(config, "dim"):
                return config.dim
        except Exception:
            pass

        return 384  # Final default to 384 if everything fails

    def encode(self, image: Image.Image):
        """Processes an image into a latent vector."""
        self._lazy_init()

        if self._failed:
            # Return an empty/zero vector if vision is disabled
            return np.zeros(self.embedding_dim, dtype=np.float32)

        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                latent = outputs.pooler_output[0]
            else:
                latent = outputs.last_hidden_state.mean(dim=1)[0]
        return latent.cpu().numpy()
