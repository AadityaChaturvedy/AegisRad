from config import CONFIDENCE_THRESHOLD, CLASS_THRESHOLDS


class ThresholdFilter:
    def __init__(self, thresholds: dict = CLASS_THRESHOLDS, fallback: float = CONFIDENCE_THRESHOLD):
        self.thresholds = thresholds
        self.fallback   = fallback

    def filter(self, condition_probs: dict) -> dict:
        """Filter condition probabilities using class-specific thresholds for optimal F1."""
        flagged = {}
        
        for cond, prob in condition_probs.items():
            if cond == "No Finding": continue
            
            # Use class-specific threshold if available, else fallback
            t = self.thresholds.get(cond, self.fallback)
            if prob >= t:
                flagged[cond] = prob

        # If nothing above threshold, check if "No Finding" dominates
        if not flagged:
            no_finding_prob = condition_probs.get("No Finding", 0)
            no_finding_thresh = self.thresholds.get("No Finding", self.fallback)
            
            if no_finding_prob >= no_finding_thresh:
                return {"No Finding": no_finding_prob}

            # Fallback: surface the topnd non-trivial conditions if still nothing
            # (using a lower 'safety' threshold of 0.15)
            ranked = sorted(
                ((c, p) for c, p in condition_probs.items()
                 if c != "No Finding" and p > 0.15),
                key=lambda x: x[1], reverse=True
            )
            if ranked:
                return dict(ranked[:3])

        return flagged