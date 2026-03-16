import re

SEVERITY_MAP = {
    1: "NORMAL",
    2: "MILD",
    3: "MODERATE",
    4: "SEVERE",
    5: "CRITICAL"
}

URGENCY_ICONS = {
    "ROUTINE":  "🟢",
    "MODERATE": "🟡",
    "URGENT":   "🟠",
    "CRITICAL": "🔴"
}


class ReportFormatter:
    def parse(self, raw_text: str) -> dict:
        def extract(label):
            pattern = rf"{label}:\s*(.+?)(?=\n[A-Z]+:|$)"
            match   = re.search(pattern, raw_text, re.DOTALL)
            return match.group(1).strip() if match else "Not determined"

        severity_raw = extract("SEVERITY")
        try:
            severity_num = int(re.search(r"\d", severity_raw).group())
            severity_num = max(1, min(5, severity_num))
        except Exception:
            severity_num = 3

        urgency_raw   = extract("URGENCY").upper()
        urgency_clean = next(
            (u for u in ["CRITICAL", "URGENT", "MODERATE", "ROUTINE"]
             if u in urgency_raw),
            "MODERATE"
        )

        return {
            "findings":       extract("FINDINGS"),
            "impression":     extract("IMPRESSION"),
            "severity_score": severity_num,
            "severity_label": SEVERITY_MAP.get(severity_num, "MODERATE"),
            "urgency":        urgency_clean,
            "urgency_icon":   URGENCY_ICONS.get(urgency_clean, "🟡"),
            "recommendation": extract("RECOMMENDATION")
        }