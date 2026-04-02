import os
import sys
import torch
import time
from pipeline import AegisRadPipeline

# Test image path provided by user
TEST_IMAGE_DIR = "/Volumes/SSD/Jetson_AegisRad/Code Nano/PresentationImages"

def verify_pipeline():
    print("🚀 AegisRad Nano Performance Verification")
    print("-" * 40)
    
    # Check for images
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"❌ Test image directory not found: {TEST_IMAGE_DIR}")
        return
        
    images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.jpg')]
    if not images:
        print(f"❌ No JPG images found in {TEST_IMAGE_DIR}")
        return
        
    test_image = os.path.join(TEST_IMAGE_DIR, images[0])
    print(f"📸 Using test image: {images[0]}")

    # 1. Initialize Pipeline
    try:
        ts = time.time()
        pipeline = AegisRadPipeline(use_onnx=True)
        init_time = time.time() - ts
        print(f"✅ Pipeline initialized in {init_time:.2f}s")
        print(f"   Mode: {'ONNX' if pipeline.encoder_session else 'PyTorch'}")
        print(f"   Device: {pipeline.device}")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return

    # 2. Run Inference
    try:
        print("\nRunning full triage analysis...")
        result = pipeline.run(test_image, language="English")
        
        print("\n--- Clinical Findings ---")
        print(f"ID: {images[0]}")
        print(f"Severity: {result['severity_score']}/5 ({result['severity_label']})")
        print(f"Urgency: {result['urgency']}")
        print(f"Findings: {result['findings']}")
        print(f"Impression: {result['impression']}")
        
        print("\n--- Flagged Pathologies ---")
        for cond, prob in result['flagged'].items():
            print(f"  • {cond}: {prob:.1%}")
            
        print("\n--- Latency Breakdown ---")
        print(f"Total Latency: {result['latency_s']}s")
        for stage, duration in result['timings'].items():
            print(f"  • {stage:15}: {duration:.4f}s")
            
        print("\n✅ Verification SUCCESSFUL")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline()
