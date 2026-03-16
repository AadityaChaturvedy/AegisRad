from config import CONFIDENCE_THRESHOLD


class ThresholdFilter:
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def filter(self, condition_probs: dict) -> dict:
        flagged = {
            cond: prob
            for cond, prob in condition_probs.items()
            if prob >= self.threshold and cond != "No Finding"
        }
        # If nothing flagged check if No Finding is dominant
        if not flagged:
            no_finding = condition_probs.get("No Finding", 0)
            if no_finding >= self.threshold:
                return {"No Finding": no_finding}
        return flagged