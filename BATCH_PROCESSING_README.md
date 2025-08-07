# SAHI Batch Processing Enhancement

## Overview
This modification adds batch processing capabilities to the SAHI library's `get_sliced_prediction` function, allowing multiple image slices to be processed simultaneously rather than sequentially. This can significantly improve GPU utilization and overall inference speed.

## Key Changes Made

### 1. Modified `get_sliced_prediction` function (`sahi/predict.py`)
- Added `batch_size: int = 1` parameter
- Updated function signature and docstring
- Replaced sequential slice processing with batch processing logic
- Added fallback to sequential processing if batch processing fails

### 2. Added `get_batch_prediction` function (`sahi/predict.py`)
- New function to handle batch inference for multiple images
- Supports different model types with fallback to sequential processing
- Proper error handling and performance timing

### 3. Enhanced `UltralyticsDetectionModel` (`sahi/models/ultralytics.py`)
- Added `perform_batch_inference()` method for batch inference
- Added `convert_batch_predictions()` method for batch result processing
- Enhanced `_create_object_prediction_list_from_original_predictions()` to handle multiple image shapes

### 4. Updated `predict` and `predict_fiftyone` functions
- Added `batch_size` parameter to both functions
- Updated function calls to pass through the batch_size parameter

## Usage Examples

### Basic Usage
```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# Initialize your model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='yolov8n.pt',
    confidence_threshold=0.25,
    device="cuda:0",  # GPU recommended for batch processing
)

# Run with batch processing (process 4 slices simultaneously)
result = get_sliced_prediction(
    image="path/to/your/image.jpg",
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    batch_size=4,  # New parameter!
)
```

### Using with CLI
```bash
# The predict function also supports batch processing
python -c "
from sahi.predict import predict
predict(
    model_type='ultralytics',
    model_path='yolov8n.pt', 
    source='path/to/images/',
    slice_height=512,
    slice_width=512,
    batch_size=4
)
"
```

### Backward Compatibility
```python
# This still works exactly as before (batch_size defaults to 1)
result = get_sliced_prediction(
    image="path/to/your/image.jpg",
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
)
```

## Performance Considerations

### When Batch Processing is Most Effective
1. **GPU Usage**: Batch processing shows the most benefit when using GPU acceleration
2. **Many Small Slices**: Works best when images are divided into many slices
3. **Model Support**: Currently optimized for Ultralytics models (YOLO variants)
4. **Available Memory**: Higher batch sizes require more GPU/CPU memory

### Recommended Batch Sizes
- **GPU with 8GB+ VRAM**: batch_size=8-16
- **GPU with 4-8GB VRAM**: batch_size=4-8  
- **GPU with <4GB VRAM**: batch_size=2-4
- **CPU Only**: batch_size=1-2 (limited benefit)

### Memory Usage
Batch processing increases memory usage proportionally to batch size. If you encounter out-of-memory errors, reduce the batch_size parameter.

## Model Support

### Currently Supported
- ✅ **Ultralytics Models** (YOLOv8, YOLO11, etc.): Full batch support
- ✅ **Other Models**: Automatic fallback to sequential processing

### Future Enhancements
The batch processing infrastructure can be extended to support other model types:
- MMDetection models
- HuggingFace models  
- TorchVision models
- Detectron2 models

## Technical Details

### Batch Processing Flow
1. Image is sliced as usual using existing slicing logic
2. Slices are grouped into batches of size `batch_size`
3. Each batch is processed simultaneously using model's batch inference
4. Results are converted back to individual predictions per slice
5. Standard post-processing (NMS, etc.) is applied as before

### Error Handling
- If batch processing fails for any reason, the system automatically falls back to sequential processing
- Warnings are logged when fallback occurs
- No functionality is lost - the system is robust

### Performance Monitoring
The timing information includes:
- `slice`: Time to slice the image
- `prediction`: Time for all predictions (batch or sequential)  
- `postprocess`: Time for post-processing steps

## Testing

A test script `test_batch_processing.py` is included that:
- Compares performance across different batch sizes
- Demonstrates proper usage
- Validates that results are consistent
- Provides performance benchmarking

Run it with:
```bash
python test_batch_processing.py
```

## Benefits

1. **Improved GPU Utilization**: Better use of parallel processing capabilities
2. **Faster Inference**: Significant speedup possible, especially on GPUs
3. **Backward Compatible**: Existing code continues to work unchanged
4. **Graceful Fallback**: Robust error handling ensures reliability
5. **Configurable**: Users can tune batch size based on their hardware

## Conclusion

This enhancement maintains full backward compatibility while providing significant performance improvements for users with appropriate hardware. The batch processing feature is particularly beneficial for:

- High-throughput applications
- Large image processing workflows  
- GPU-accelerated inference pipelines
- Scenarios with many small slices

The implementation is robust with proper error handling and automatic fallbacks, ensuring that the library remains reliable while providing these performance benefits.
