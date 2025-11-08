#!/usr/bin/env python3
"""
Example script demonstrating the new batch processing feature in SAHI.
"""

import time

import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def main():
    print("SAHI Batch Processing Example")
    print("=" * 40)

    # Create or load your model
    print("Loading model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolov8n.pt",  # This will download automatically if not present
        confidence_threshold=0.25,
        device="cuda" if "cuda" in str(time.time()) else "cpu",  # Simple check for demo
    )

    # Create a test image (you can replace this with a real image path)
    print("Creating test image...")
    test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    print("\nRunning inference with different batch sizes...")

    # Sequential processing (batch_size=1, default behavior)
    print("\n1. Sequential processing (batch_size=1):")
    start_time = time.time()
    result_sequential = get_sliced_prediction(
        image=test_image,
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        batch_size=1,  # Process one slice at a time
        verbose=0,
    )
    sequential_time = time.time() - start_time
    print(f"   Time: {sequential_time:.3f}s")
    print(f"   Predictions: {len(result_sequential.object_prediction_list)}")

    # Batch processing (batch_size=4)
    print("\n2. Batch processing (batch_size=4):")
    start_time = time.time()
    result_batch = get_sliced_prediction(
        image=test_image,
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        batch_size=4,  # Process 4 slices simultaneously
        verbose=0,
    )
    batch_time = time.time() - start_time
    print(f"   Time: {batch_time:.3f}s")
    print(f"   Predictions: {len(result_batch.object_prediction_list)}")

    # Calculate speedup
    speedup = sequential_time / batch_time if batch_time > 0 else 1
    print(f"\n📈 Speedup: {speedup:.2f}x")

    if speedup > 1.1:
        print("✅ Batch processing provided a speedup!")
    elif speedup > 0.9:
        print("ℹ️  Similar performance (results may vary with real images and GPU)")
    else:
        print("ℹ️  No speedup observed (this can happen with small examples)")

    print("\n" + "=" * 40)
    print("Key takeaways:")
    print("- Use batch_size > 1 for potential speedup")
    print("- GPU acceleration recommended for best results")
    print("- Larger images with more slices benefit most")
    print("- Backward compatible: batch_size=1 is default")

    # Show how to use with real image
    print("\nTo use with your own image:")
    print("""
result = get_sliced_prediction(
    image="path/to/your/image.jpg",  # Your image here
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    batch_size=4,  # Adjust based on your GPU memory
)
""")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required packages installed:")
        print("pip install ultralytics torch")
        print("\nFor GPU support also install appropriate CUDA version of PyTorch.")
