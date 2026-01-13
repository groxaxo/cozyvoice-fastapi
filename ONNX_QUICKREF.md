# ONNX Quick Reference Card

## Default Configuration

ONNX is **enabled by default** - just start the server normally:

```bash
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSYVOICE_USE_ONNX` | `true` | Enable/disable ONNX modules |
| `COSYVOICE_ONNX_REPO` | `Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX` | Hugging Face repository for ONNX models |
| `COSYVOICE_FP16` | `false` | Use FP16 precision (faster, less memory) |

## Common Configurations

### Default (ONNX + FP32)
```bash
# Already enabled - just start the server
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### ONNX + FP16 (Recommended)
```bash
export COSYVOICE_USE_ONNX=true
export COSYVOICE_FP16=true
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### Disable ONNX (Pure PyTorch)
```bash
export COSYVOICE_USE_ONNX=false
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### Custom ONNX Repository
```bash
export COSYVOICE_ONNX_REPO=your-username/custom-repo
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

## Quick Commands

### Check ONNX Status
```bash
curl http://localhost:8092/health | python -m json.tool
# Look for: "onnx": true
```

### Verify ONNX Models
```bash
ls -lh CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/*.onnx
# Should show: flow_fp32.onnx (or flow_fp16.onnx) and hift.onnx
```

### Manual Download
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
model_dir = 'CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512'
repo = 'Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX'
hf_hub_download(repo_id=repo, filename='flow_fp32.onnx', local_dir=model_dir)
hf_hub_download(repo_id=repo, filename='hift.onnx', local_dir=model_dir)
"
```

### Start with ONNX Script
```bash
chmod +x start_server_onnx.sh
./start_server_onnx.sh
```

## Troubleshooting

### Models Not Downloading
```bash
# Check internet connectivity
ping huggingface.co

# Verify huggingface_hub is installed
pip install huggingface_hub

# Download manually (see Manual Download above)
```

### Wrong Precision
```bash
# Force FP32
export COSYVOICE_FP16=false
export COSYVOICE_USE_ONNX=true

# Force FP16
export COSYVOICE_FP16=true
export COSYVOICE_USE_ONNX=true
```

### Performance Issues
```bash
# Install cuDNN for GPU acceleration
conda install -c conda-forge cudnn=8.9.7.29 -y

# Try FP16
export COSYVOICE_FP16=true

# Check GPU usage
nvidia-smi dmon -s u
```

## Performance Comparison

| Configuration | RTF | Setup | Recommended For |
|--------------|-----|-------|-----------------|
| ONNX + FP32 (Default) | ~0.350 | Easy | Most users |
| ONNX + FP16 | ~0.340 | Easy | Production |
| PyTorch (ONNX off) | 0.364 | Easy | Debugging |
| TensorRT + FP16 | 0.340 | Hard | Maximum speed |

*Lower RTF is better*

## More Information

- **Full Guide**: [ONNX_GUIDE.md](ONNX_GUIDE.md)
- **Installation**: [INSTALLATION.md](INSTALLATION.md)
- **Performance**: [PERFORMANCE.md](PERFORMANCE.md)
- **README**: [README.md](README.md)

## Support

For issues:
1. Check server logs: `tail -f server.log`
2. Verify ONNX models exist in model directory
3. Test with ONNX disabled: `export COSYVOICE_USE_ONNX=false`
4. Review [ONNX_GUIDE.md](ONNX_GUIDE.md) troubleshooting section
5. Open GitHub issue with logs and configuration
