import numpy as np
import torch
from aegisrad.models import ReflexiveProjector


class Projector:
    def __init__(self, state_dict=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = ReflexiveProjector()

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.half()
        self.model.eval()
        self.model.to(self.device)

        print(f"[AegisRad] ReflexiveProjector loaded (fp16, device: {self.device}).")

    def project(self, query_embeddings):
        """Accept query embeddings as torch tensor [B, 32, 2048].
        Returns projected features as numpy array for the LLM stage."""
        if not isinstance(query_embeddings, torch.Tensor):
            query_embeddings = torch.from_numpy(query_embeddings)
        query_embeddings = query_embeddings.to(device=self.device,
                                               dtype=torch.float16)

        # Mean-pool: [B, 32, 2048] → [B, 2048]
        pooled = query_embeddings.mean(dim=1)

        with torch.no_grad():
            projected = self.model(pooled)  # [B, 32, 2048]

        # Convert to numpy only at the boundary (LLM expects numpy)
        return projected.float().cpu().numpy()
