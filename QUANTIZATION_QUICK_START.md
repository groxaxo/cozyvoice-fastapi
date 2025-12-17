# Quantization Quick Start Guide

This is a condensed guide for quickly implementing quantization in the CosyVoice FastAPI server. For detailed analysis and rationale, see [QUANTIZATION_ANALYSIS.md](QUANTIZATION_ANALYSIS.md).

## TL;DR

- **Goal**: Reduce VRAM usage by 50-75%
- **Method**: BitsAndBytes 8-bit or 4-bit quantization
- **Target**: Primarily the LLM component (0.5B parameters)
- **Impact**: Minimal quality loss (<3%), slight speed reduction (5-20%)

## Prerequisites

```bash
conda activate cosyvoice3
pip install bitsandbytes>=0.41.0 accelerate>=0.20.0
```

## Quick Implementation (Environment Variable Method)

### Step 1: Add to `openai_tts_cosyvoice_server.py` (before model loading)

```python
import os
import torch
from transformers import BitsAndBytesConfig

# Configuration via environment variables
QUANTIZATION_ENABLED = os.environ.get("QUANTIZATION_ENABLED", "false").lower() == "true"
QUANTIZATION_BITS = int(os.environ.get("QUANTIZATION_BITS", "8"))

# Create quantization config
if QUANTIZATION_ENABLED:
    if QUANTIZATION_BITS == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    elif QUANTIZATION_BITS == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {QUANTIZATION_BITS}")
    
    print(f"Quantization enabled: {QUANTIZATION_BITS}-bit")
else:
    bnb_config = None
    print("Quantization disabled")
```

### Step 2: Modify CosyVoice's AutoModel (Advanced)

**Note**: This requires modifying the CosyVoice library code. If you don't want to modify the library, skip to Alternative Method below.

In `CosyVoice/cosyvoice/cli/cosyvoice.py`, modify the `AutoModel` function to accept quantization config:

```python
def AutoModel(model_dir, quantization_config=None, device_map="auto", **kwargs):
    # Existing code...
    # When loading the LLM component, pass quantization_config
    # This is highly dependent on CosyVoice's internal structure
    pass
```

### Step 3: Run with quantization

```bash
# 8-bit quantization (recommended)
export QUANTIZATION_ENABLED=true
export QUANTIZATION_BITS=8
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000

# 4-bit quantization (maximum savings)
export QUANTIZATION_ENABLED=true
export QUANTIZATION_BITS=4
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

## Alternative Method (No Library Modification)

If you can't or don't want to modify the CosyVoice library, you can apply quantization post-loading:

```python
from cosyvoice.cli.cosyvoice import AutoModel
import torch
from transformers import BitsAndBytesConfig

# Load model normally
cosyvoice_model = AutoModel(model_dir=MODEL_DIR)

# Apply quantization to specific components
if QUANTIZATION_ENABLED:
    # This is a simplified example - actual implementation depends on
    # CosyVoice's internal structure
    
    # Option 1: Quantize the entire model (may not work with all components)
    if hasattr(cosyvoice_model, 'llm'):
        from bitsandbytes import nn as bnb_nn
        # Replace Linear layers with quantized versions
        # This requires recursive module replacement
        pass
    
    # Option 2: Use torch.quantization (different approach)
    if QUANTIZATION_BITS == 8:
        cosyvoice_model = torch.quantization.quantize_dynamic(
            cosyvoice_model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

## Testing Your Implementation

```python
# Add this after model loading
import torch

if torch.cuda.is_available():
    vram_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Model VRAM usage: {vram_gb:.2f} GB")
    
    # Expected values:
    # - No quantization: ~2.0 GB
    # - 8-bit: ~1.0 GB
    # - 4-bit: ~0.5 GB
```

## Quick Test Script

```bash
# Create test_vram.py
cat > test_vram.py << 'EOF'
import sys
import torch
sys.path.append('CosyVoice')

from cosyvoice.cli.cosyvoice import AutoModel

print("Loading model...")
model = AutoModel(model_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')

if torch.cuda.is_available():
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM Usage: {vram:.2f} GB")
else:
    print("No CUDA available")

# Test inference
text = "Hello world, this is a test."
for result in model.inference_zero_shot(
    text,
    'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
    'CosyVoice/asset/zero_shot_prompt.wav',
    stream=False
):
    print(f"Generated audio shape: {result['tts_speech'].shape}")
    break
EOF

python test_vram.py
```

## Expected Results

| Configuration | VRAM | Speed | Quality |
|--------------|------|-------|---------|
| FP32 | ~2.0 GB | 1.0x | 100% |
| FP16 | ~1.0 GB | 1.0x | 100% |
| **8-bit (Recommended)** | **~0.5 GB** | **0.9x** | **99%** |
| 4-bit | ~0.25 GB | 0.85x | 97% |

## Troubleshooting

### Error: "No module named 'bitsandbytes'"

```bash
pip install bitsandbytes>=0.41.0
```

### Error: "CUDA not available"

BitsAndBytes requires NVIDIA GPU with CUDA. Check:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Model loading fails with quantization

1. Check if CosyVoice's AutoModel supports quantization parameters
2. Try the post-loading quantization approach instead
3. Fall back to native PyTorch quantization:
   ```python
   cosyvoice_model = torch.quantization.quantize_dynamic(
       cosyvoice_model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

### Audio quality is degraded

1. Use 8-bit instead of 4-bit
2. Keep vocoder in FP16/FP32
3. Apply quantization only to LLM component

### Inference is too slow

1. Ensure model is on GPU, not CPU
2. Use 8-bit instead of 4-bit
3. Disable double quantization: `bnb_4bit_use_double_quant=False`

## Next Steps

1. ✅ Install dependencies
2. ✅ Add quantization config to server
3. ✅ Test VRAM usage
4. ⬜ Compare audio quality (before/after)
5. ⬜ Benchmark inference speed
6. ⬜ Deploy to production

## Resources

- Full analysis: [QUANTIZATION_ANALYSIS.md](QUANTIZATION_ANALYSIS.md)
- BitsAndBytes docs: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- CosyVoice repo: https://github.com/FunAudioLLM/CosyVoice

## Need Help?

If you encounter issues:
1. Check the full [QUANTIZATION_ANALYSIS.md](QUANTIZATION_ANALYSIS.md) for detailed guidance
2. Verify your environment matches the prerequisites
3. Test without quantization first to ensure base setup works
4. Open an issue with error messages and environment details
