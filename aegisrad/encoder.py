import torch
from aegisrad.models import VisionEncoder


class BioViLEncoder:
    def __init__(self, state_dict=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Create model architecture
        self.model = VisionEncoder()

        # Load weights from shared state_dict (avoids redundant torch.load)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        # Convert to half-precision to save ~45 MB of RAM
        self.model.half()
        self.model.eval()
        self.model.to(self.device)

        print(f"[AegisRad] BioViL-T encoder loaded (fp16, device: {self.device}).")

    def encode(self, image):
        """Accept preprocessed numpy array or torch tensor.
        Returns (features, pooled) as torch tensors on self.device."""
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        image = image.to(device=self.device, dtype=torch.float16)

        with torch.no_grad():
            features, pooled = self.model(image)
        return features, pooled
