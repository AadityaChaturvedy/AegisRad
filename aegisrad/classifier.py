import torch
from aegisrad.models import ClinicalHead
from config import CONDITIONS


class Classifier:
    def __init__(self, state_dict=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = ClinicalHead(num_classes=len(CONDITIONS))

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.half()
        self.model.eval()
        self.model.to(self.device)

        print(f"[AegisRad] Classification head loaded (fp16, device: {self.device}).")

    def predict(self, pooled):
        """Accept pooled features as torch tensor [B, 2048].
        Returns dict of condition -> probability."""
        if not isinstance(pooled, torch.Tensor):
            pooled = torch.from_numpy(pooled)
        pooled = pooled.to(device=self.device, dtype=torch.float16)

        with torch.no_grad():
            logits = self.model(pooled)
            probs = torch.sigmoid(logits)

        probs_np = probs.float().cpu().numpy().squeeze()

        return {
            cond: float(round(prob, 4))
            for cond, prob in zip(CONDITIONS, probs_np)
        }
