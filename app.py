import gradio as gr
from pipeline import AegisRadPipeline
from config import LANGUAGES

pipeline = AegisRadPipeline()


def analyze(image, language):
    if image is None:
        return ("No image provided",) * 7

    image.save("/tmp/aegisrad_input.jpg")
    r = pipeline.run("/tmp/aegisrad_input.jpg", language=language)

    flagged_text = "\n".join(
        f"{cond}: {prob:.1%}"
        for cond, prob in sorted(
            r["flagged"].items(),
            key=lambda x: x[1],
            reverse=True
        )
    ) or "No significant findings detected"

    return (
        r["findings"],
        r["impression"],
        f"{r['severity_score']} / 5  —  {r['severity_label']}",
        f"{r['urgency_icon']}  {r['urgency']}",
        r["recommendation"],
        flagged_text,
        f"{r['latency_s']}s"
    )


with gr.Blocks(title="AegisRad Clinical Triage") as app:

    gr.Markdown("# AegisRad — Adaptive Radiological Intelligence")
    gr.Markdown(
        "### Edge Clinical Triage System  |  NVIDIA Jetson Nano  |  "
        "BioViL-T · RRA-Q · Gemma-2B · NLLB-200"
    )
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Chest X-Ray (JPG)"
            )
            language_select = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="English",
                label="Report Language"
            )
            analyze_btn = gr.Button(
                "Run Triage Analysis",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            flagged_out = gr.Textbox(
                label="Flagged Conditions (above 50% confidence)",
                lines=5
            )
            severity = gr.Textbox(label="Severity Score")
            urgency  = gr.Textbox(label="Urgency Flag")
            latency  = gr.Textbox(label="Inference Time")

    gr.Markdown("---")

    with gr.Row():
        findings   = gr.Textbox(label="Findings",   lines=3)
        impression = gr.Textbox(label="Impression", lines=3)

    with gr.Row():
        recommendation = gr.Textbox(
            label="Recommendation",
            lines=2
        )

    analyze_btn.click(
        fn=analyze,
        inputs=[image_input, language_select],
        outputs=[
            findings, impression, severity, urgency,
            recommendation, flagged_out, latency
        ]
    )

app.launch(server_name="0.0.0.0", server_port=7860)