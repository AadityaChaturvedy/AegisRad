# AegisRad | Jetson Nano Deployment

Welcome to the Jetson Nano deployment package for AegisRad. This version features the **Modern Bento UI** and the **Multimodal VLM** generation engine.

## 📦 Setup Instructions

1. **Hardware Preparation**:
   - Ensure you are using the **4GB Jetson Nano** model.
   - **Recommended**: Enable a 4GB Swap file to prevent OOM errors during LLM inference.
   
2. **Environment**:
   ```bash
   pip install -r requirements_nano.txt
   ```

3. **Models**:
   Ensure the following weights are in the `../Code Nano/models/` directory (or update `inference_nano.py` paths):
   - `best/components.pt` (Clinical & Projector weights)
   - `gemma-2-2b-it-Q4_K_M.gguf` (The LLM brain)

4. **Run the Dashboard**:
   ```bash
   python3 app.py
   ```
   The dashboard will be available at `http://localhost:5000` or the Jetson's IP address on your network.

## 📊 Features
- **Bento Grid Layout**: High-fidelity dark mode dashboard.
- **Privacy-First**: No probabilities or data-leakage markers shown.
- **Generative Reports**: Real-time professional radiology reports from Gemma-2B.
- **Clinic-Ready**: Follows Findings -> Impression -> Recommendation structure.
