import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


class VisualStagnationDetector:
    def __init__(self, buffer_size: int = 15, threshold: float = 0.99):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.history = []

        # Pre-calculate a standard downsample transform for speed
        self.transform = T.Compose(
            [
                T.Resize((64, 64)),
                T.Grayscale(),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        )

    def get_embedding(self, image_data: Image.Image) -> torch.Tensor:
        """Converts raw image into a low-res semantic representation."""
        with torch.no_grad():
            return self.transform(image_data).flatten()

    def is_stagnant(self, current_image: Image.Image) -> bool:
        """Checks if the current view is visually similar to recent history."""
        current_emb = self.get_embedding(current_image)

        if not self.history:
            self.history.append(current_emb)
            return False

        # Calculate Cosine Similarity against the mean of recent history
        history_tensor = torch.stack(self.history)
        similarity = (
            F.cosine_similarity(current_emb.unsqueeze(0), history_tensor).mean().item()
        )

        # Update Buffer
        self.history.append(current_emb)
        if len(self.history) > self.buffer_size:
            self.history.pop(0)

        # Trigger if similarity is too high (0.99 = virtually identical)
        return similarity > self.threshold

    def reset(self):
        self.history = []
