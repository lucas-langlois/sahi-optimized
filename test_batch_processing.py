#!/usr/bin/env python3
"""
Test script to demonstrate batch processing capabilities in SAHI.
"""

import time

import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def test_batch_vs_sequential():
    """
    Test to compare batch vs sequential processing performance.
    """
    # Init model (using YOLOv8 as example)
    try:
        # Set device preference
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="yolov8n.pt",  # Small model for testing
            confidence_threshold=0.25,
            device=device,
        )
    except Exception as e:
        print(f"Could not load YOLOv8 model: {e}")
        print("Make sure you have ultralytics installed: pip install ultralytics")
        return

    # Create a test image (or load a real one)
    # For demonstration, we'll use a random image
    test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    print("Testing SAHI batch processing capabilities...")
    print("=" * 50)

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    slice_params = {
        "slice_height": 256,
        "slice_width": 256,
        "overlap_height_ratio": 0.2,
        "overlap_width_ratio": 0.2,
        "verbose": 0,  # Reduce output for cleaner results
    }

    results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting batch_size = {batch_size}")

        start_time = time.time()

        result = get_sliced_prediction(
            image=test_image, detection_model=detection_model, batch_size=batch_size, **slice_params
        )

        end_time = time.time()
        duration = end_time - start_time

        results[batch_size] = {
            "duration": duration,
            "num_predictions": len(result.object_prediction_list),
            "slice_time": result.durations_in_seconds.get("slice", 0),
            "prediction_time": result.durations_in_seconds.get("prediction", 0),
        }

        print(f"  Total time: {duration:.3f}s")
        print(f"  Prediction time: {results[batch_size]['prediction_time']:.3f}s")
        print(f"  Detections found: {results[batch_size]['num_predictions']}")

    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    baseline = results[1]["duration"]
    print("Batch Size | Duration (s) | Speedup | Predictions")
    print("-" * 50)
    for batch_size in batch_sizes:
        duration = results[batch_size]["duration"]
        speedup = baseline / duration if duration > 0 else 0
        predictions = results[batch_size]["num_predictions"]
        print(f"{batch_size:^10} | {duration:^11.3f} | {speedup:^7.2f}x | {predictions:^11}")

    print("\nNOTE: Actual speedup depends on:")
    print("- GPU memory and compute capability")
    print("- Model complexity and image size")
    print("- Number of slices generated")
    print("- System configuration")

    print("\nBatch processing is most effective when:")
    print("- Using GPU acceleration")
    print("- Processing many small slices")
    print("- Model supports efficient batch inference")


def demonstrate_usage():
    """
    Demonstrate basic usage of the batch processing feature.
    """
    print("\nDEMONSTRATING BASIC USAGE")
    print("=" * 30)

    # Example code showing how to use batch processing
    example_code = """
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# Initialize your model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',  # or 'mmdet', 'huggingface', etc.
    model_path='yolov8n.pt',
    confidence_threshold=0.25,
    device="cuda:0",  # Use GPU for best batch performance
)

# Run sliced prediction with batch processing
result = get_sliced_prediction(
    image="path/to/your/image.jpg",
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    batch_size=4,  # Process 4 slices simultaneously
)

# Access results
predictions = result.object_prediction_list
print(f"Found {len(predictions)} objects")
"""

    print("Basic usage example:")
    print(example_code)


if __name__ == "__main__":
    # Set device preference
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run demonstration
    demonstrate_usage()

    # Ask user if they want to run performance test
    response = input("\nRun performance test? This requires downloading YOLOv8n (~6MB) (y/n): ").lower()
    if response.startswith("y"):
        test_batch_vs_sequential()
    else:
        print("\nSkipped performance test.")
        print("To run manually: python test_batch_processing.py")

    print("\nBatch processing has been successfully integrated into SAHI!")
    print("Use batch_size parameter in get_sliced_prediction() to enable it.")
