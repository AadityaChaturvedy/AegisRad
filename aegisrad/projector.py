import numpy as np


class Projector:
    def __init__(self):
        # QFormer already outputs 2048-dim embeddings
        # matching Llama's embedding space directly.
        # No projection needed.
        print("[AegisRad] Projector: passthrough (QFormer → 2048).")

    def project(self, query_embeddings: np.ndarray) -> np.ndarray:
        return query_embeddings   # [1, 32, 2048] passthrough