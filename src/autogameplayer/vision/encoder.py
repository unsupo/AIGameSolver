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
            
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                
            print(f"  - Moving model to {self.device}...", flush=True)
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            print("  - Vision Encoder ready.", flush=True)
        except ImportError:
            print("⚠️ Vision dependencies (torch/transformers) not installed. Encoder running in passthrough mode.")
            self._failed = True
        except Exception as e:
            print(f"⚠️ Vision Encoder failed to initialize: {e}. Running in passthrough mode.")
            self._failed = True

    def encode(self, image: Image.Image):
        """Processes an image into a latent vector."""
        self._lazy_init()
        
        if self._failed:
            # Return an empty/zero vector if vision is disabled
            return np.zeros(384, dtype=np.float32) # Default size for DINOv2 small

        import torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                latent = outputs.pooler_output[0]
            else:
                latent = outputs.last_hidden_state.mean(dim=1)[0]
        return latent.cpu().numpy()
