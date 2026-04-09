import torch
from aegisrad.models import QFormer


class RRAQAdapter:
    def __init__(self, state_dict=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = QFormer()

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.half()
        self.model.eval()
        self.model.to(self.device)

        print(f"[AegisRad] RRA-Q Adapter loaded (fp16, device: {self.device}).")

    def extract_queries(self, visual_features):
        """Accept visual features as torch tensor [B, 2048, 7, 7].
        Returns query embeddings as torch tensor [B, 32, 2048]."""
        if not isinstance(visual_features, torch.Tensor):
            visual_features = torch.from_numpy(visual_features)
        visual_features = visual_features.to(device=self.device,
                                             dtype=torch.float16)

        with torch.no_grad():
            queries = self.model(visual_features)
        return queries
