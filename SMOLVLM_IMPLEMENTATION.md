# SmolVLM Implementation Guide

## Overview

This document describes the implementation of SmolVLM (HuggingFaceTB/SmolVLM-2.2B-Instruct) support for the image recognition module. The implementation follows a modular architecture using separate extractor classes for different Vision Language Models.

## Architecture

### Class Hierarchy

```
VLMExtractor (Abstract Base)
├── LocalVLMExtractor (Base Class for Local Models)
│   ├── LocalQwenExtractor (Qwen2-VL-2B-Instruct)
│   └── LocalSmolVLMExtractor (SmolVLM-2.2B-Instruct)
└── RemoteVLLMExtractor (Remote vLLM Server)
```

### Design Pattern: Template Method

The implementation uses the **Template Method** design pattern:

1. **LocalVLMExtractor** provides the template with common functionality:
   - Model initialization and CUDA detection
   - Image preparation and scaling
   - JSON response parsing
   - Batch processing support
   - Error handling and logging

2. **Subclasses** implement specific methods:
   - `_load_model()`: Load model and processor with model-specific configurations
   - `_process_image()`: Process images using model-specific APIs

## Implementation Details

### LocalVLMExtractor (Base Class)

**Location:** `src/image_rec_mod/backends/vlm.py`

**Key Features:**
- Automatic CUDA detection with fallback to CPU
- Comprehensive logging for debugging
- Image scaling support via `ImageScaler`
- JSON response parsing with markdown cleanup
- Batch processing with `extract_multiple_bib_numbers()`

**Abstract Methods:**
```python
def _load_model(self) -> None:
    """Load model and processor. Set self.model and self.processor."""
    
def _process_image(self, image_path: str) -> dict:
    """Process image and return extraction result."""
```

### LocalQwenExtractor

**Model:** Qwen/Qwen2-VL-2B-Instruct

**Specific Implementation:**
- Uses `Qwen2VLForConditionalGeneration` class
- Requires `qwen_vl_utils.process_vision_info()` for image processing
- Uses file:// URI format for image paths
- Supports video inputs (unused in current implementation)

**Key Differences:**
```python
# Qwen-specific image processing
from qwen_vl_utils import process_vision_info
image_inputs, video_inputs = process_vision_info(messages)
inputs = self.processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
```

### LocalSmolVLMExtractor

**Model:** HuggingFaceTB/SmolVLM-2.2B-Instruct

**Specific Implementation:**
- Uses `AutoModelForVision2Seq` class
- Loads images directly with PIL
- Uses float16 on CUDA, float32 on CPU for optimal performance
- Standard transformers API (no special utilities needed)

**Key Differences:**
```python
# SmolVLM image processing
image = Image.open(image_path)
inputs = self.processor(
    text=text,
    images=image,
    return_tensors="pt"
)
```

## Usage

### Available Extractors

The following extractors are now available in `main.py`:

| Extractor Name | Class | Model | Device |
|----------------|-------|-------|--------|
| `qwen-inline-cpu` | LocalQwenExtractor | Qwen2-VL-2B-Instruct | CPU |
| `qwen-inline-gpu` | LocalQwenExtractor | Qwen2-VL-2B-Instruct | CUDA |
| `smol-inline-cpu` | LocalSmolVLMExtractor | SmolVLM-2.2B-Instruct | CPU |
| `smol-inline-gpu` | LocalSmolVLMExtractor | SmolVLM-2.2B-Instruct | CUDA |

### Code Examples

#### Basic Usage

```python
from image_rec_mod.main import get_extractor

# Get SmolVLM CPU extractor
extractor = get_extractor("smol-inline-cpu")

# Extract bib number from image
result = extractor.extract("path/to/runner_image.jpg")
print(result)  # {'bib': 256} or {'error': '...'}
```

#### With Image Scaling

```python
from image_rec_mod.main import get_extractor

# Create extractor with image scaling
extractor = get_extractor(
    "smol-inline-gpu",
    scale_width=1024,
    scale_height=768
)

result = extractor.extract("path/to/image.jpg")
```

#### Batch Processing

```python
extractor = get_extractor("smol-inline-cpu")

images = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]

results = extractor.extract_multiple_bib_numbers(
    images,
    scale_width=800,
    scale_height=600
)

for item in results:
    print(f"{item['image']}: {item['result']}")
```

## Testing

### Running Tests

```bash
# Test SmolVLM extractors
python test_smolvlm.py

# Compare Qwen vs SmolVLM performance
python test_smolvlm.py --compare
```

### Expected Behavior

1. **First Run:** Models will be downloaded from HuggingFace (~2.2GB for SmolVLM)
2. **Subsequent Runs:** Models are cached locally
3. **GPU Fallback:** If CUDA is unavailable, automatically falls back to CPU
4. **Logging:** Detailed logs show model loading, inference time, and errors

## Performance Considerations

### Model Comparison

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------|-------------|-------------|----------|
| Qwen2-VL-2B | ~2.2GB | Slow | Fast | High |
| SmolVLM-2.2B | ~2.2GB | Slow | Fast | High |

### Memory Requirements

- **CPU Mode:** ~4-6GB RAM
- **GPU Mode:** ~3-4GB VRAM (with float16)
- **Both models have similar resource requirements**

### Optimization Tips

1. Use GPU version when available (`smol-inline-gpu`)
2. Scale images down to reduce memory usage
3. Use batch processing for multiple images
4. Models are cached after first load (singleton pattern)

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Use CPU version or free up GPU memory:
```python
extractor = get_extractor("smol-inline-cpu")
```

#### 2. Model Download Fails

```
Error: Cannot download model from HuggingFace
```

**Solution:** Check internet connection or set HuggingFace cache directory:
```bash
export HF_HOME=/path/to/cache
```

#### 3. Import Errors

```
ImportError: No module named 'qwen_vl_utils'
```

**Solution:** This is expected for SmolVLM (it doesn't use this module). Only Qwen extractors require it.

## Extension Guide

### Adding New VLM Models

To add support for a new Vision Language Model:

1. **Create a new subclass:**

```python
class LocalNewModelExtractor(LocalVLMExtractor):
    """Extractor for NewModel VLM."""
    
    def _load_model(self):
        # Load your model
        self.model = YourModelClass.from_pretrained(
            self.model_name,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    def _process_image(self, image_path: str) -> dict:
        # Process image with your model
        image_to_process = self._prepare_image(image_path)
        # ... your processing logic ...
        return self._parse_json_response(output_text)
```

2. **Register in main.py:**

```python
from image_rec_mod.backends.vlm import LocalNewModelExtractor

if extractor_name == "newmodel-inline-cpu":
    extractor = LocalNewModelExtractor(
        "organization/model-name",
        device="cpu",
        scaler=scaler
    )
    _EXTRACTOR_CACHE[extractor_name] = extractor
    return extractor
```

## Dependencies

### Required Packages

```toml
transformers >= 4.30.0
torch >= 2.0.0
pillow >= 9.0.0
qwen-vl-utils >= 0.0.1  # Only for Qwen models
```

### Optional Dependencies

```toml
cuda >= 11.8  # For GPU support
```

## API Reference

### LocalVLMExtractor

Base class for local Vision Language Model extractors.

#### Methods

**`__init__(model_name: str, device: str = "cpu", scaler: Optional[ImageScaler] = None)`**
- Initialize the extractor with model configuration

**`extract(image_path: str) -> dict`**
- Extract bib number from a single image

**`extract_multiple_bib_numbers(image_paths: list[str], scale_width: Optional[int] = None, scale_height: Optional[int] = None) -> list[dict]`**
- Batch process multiple images

**`_load_model() -> None`** *(Abstract)*
- Load model and processor (implemented by subclasses)

**`_process_image(image_path: str) -> dict`** *(Abstract)*
- Process single image (implemented by subclasses)

**`_prepare_image(image_path: str) -> str`**
- Apply image scaling if configured

**`_parse_json_response(text: str) -> dict`**
- Parse JSON from model output with error handling

## Future Enhancements

### Planned Features

1. **Support for additional VLM models:**
   - LLaVA
   - CogVLM
   - InternVL
   - MiniCPM-V

2. **Enhanced configuration:**
   - Model-specific generation parameters
   - Custom prompts per model
   - Temperature and sampling controls

3. **Performance improvements:**
   - Model quantization (int8, int4)
   - Batch inference optimization
   - Model compilation with torch.compile()

4. **Additional functionality:**
   - Multiple bib detection per image
   - Confidence scores
   - Bounding box detection

## Contributing

When contributing VLM implementations:

1. Follow the existing class structure
2. Implement both `_load_model()` and `_process_image()`
3. Add comprehensive logging
4. Include error handling
5. Write tests
6. Update this documentation

## License

This implementation follows the same license as the parent project.

## References

- [SmolVLM Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM-2.2B-Instruct)
- [Qwen2-VL Documentation](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
