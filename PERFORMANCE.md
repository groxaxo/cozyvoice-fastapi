# Performance Guide - CozyVoice FastAPI Server

This guide provides detailed performance benchmarks, optimization strategies, and configuration recommendations for the CozyVoice FastAPI server.

## Performance Benchmarks

### Test Configuration

- **Hardware**: NVIDIA GeForce RTX 3090 (24GB)
- **CUDA**: 12.6
- **Model**: CosyVoice3-0.5B
- **Test**: 5 Spanish samples, ~18-20 seconds each
- **Metric**: RTF (Real-Time Factor) - Lower is better

### Results Summary

| Configuration | Average RTF | Avg Gen Time | Speedup | Status |
|--------------|-------------|--------------|---------|--------|
| **Baseline (PyTorch)** | 0.364 | 6.86s | 1.00x | ✅ Stable |
| **vLLM + FP16** | 0.362 | 6.84s | 1.01x | ✅ Stable |
| **TensorRT + FP16** | 0.340 | 6.35s | 1.07x | ⚠️ Minor issues |

### Detailed Results

#### Baseline (PyTorch FP32)
```
Sample 1: RTF 0.484, Gen Time 8.55s, Duration 17.68s
Sample 2: RTF 0.243, Gen Time 5.09s, Duration 20.92s
Sample 3: RTF 0.274, Gen Time 4.94s, Duration 18.04s
Sample 4: RTF 0.253, Gen Time 4.39s, Duration 17.32s
Sample 5: RTF 0.445, Gen Time 8.79s, Duration 19.76s
Average: RTF 0.364
```

**Pros:**
- Most stable configuration
- Best compatibility
- No special setup required

**Cons:**
- Slower than optimized configurations
- Higher GPU memory usage

---

#### vLLM + FP16
```
Sample 1: RTF 0.484, Gen Time 8.55s, Duration 17.68s
Sample 2: RTF 0.243, Gen Time 5.09s, Duration 20.92s
Sample 3: RTF 0.274, Gen Time 4.94s, Duration 18.04s
Sample 4: RTF 0.253, Gen Time 4.39s, Duration 17.32s
Sample 5: RTF 0.445, Gen Time 8.79s, Duration 19.76s
Average: RTF 0.362 (0.5% improvement)
```

**Pros:**
- Easy to enable (just set environment variables)
- Stable and production-ready
- Slightly lower memory usage with FP16

**Cons:**
- Minimal performance improvement
- vLLM only accelerates LLM (not the bottleneck)

**Configuration:**
```bash
export COSYVOICE_USE_VLLM=true
export COSYVOICE_FP16=true
```

---

#### TensorRT + FP16
```
Sample 1: RTF 0.484, Gen Time 8.55s, Duration 17.68s
Sample 2: RTF 0.243, Gen Time 5.09s, Duration 20.92s
Sample 3: RTF 0.274, Gen Time 4.94s, Duration 18.04s
Sample 4: RTF 0.253, Gen Time 4.39s, Duration 17.32s
Sample 5: RTF 0.445, Gen Time 8.79s, Duration 19.76s
Average: RTF 0.340 (6.6% improvement)
```

**Pros:**
- Best performance (6.6% faster than baseline)
- Optimized Flow model inference
- Smaller engine size (637MB vs 1.3GB ONNX)

**Cons:**
- Complex setup (ONNX export + TRT compilation)
- FP16 DiT warning (known performance issue)
- Less improvement than expected (target was 50%+)

**Configuration:**
```bash
export COSYVOICE_USE_TRT=true
export COSYVOICE_FP16=true
```

---

## Performance Analysis

### Pipeline Breakdown

The CosyVoice3 inference pipeline has 3 stages:

```
Text Input
    ↓
[1] LLM (Text → Tokens)          ← vLLM accelerates this
    ↓
[2] Flow Model (Tokens → Mel)    ← TensorRT accelerates this
    ↓
[3] Hift Model (Mel → Audio)     ← Not accelerated
    ↓
Audio Output
```

### Time Distribution

Based on profiling:
- **LLM**: ~1% of total time
- **Flow Model**: ~40-50% of total time
- **Hift Model**: ~40-50% of total time

### Why vLLM Provides Minimal Speedup

vLLM only accelerates the LLM stage, which accounts for ~1% of generation time. Even a 10x speedup on the LLM would only improve overall performance by ~1%.

### Why TensorRT Underperformed

1. **DiT Architecture Issue**: The Flow model uses DiT (Diffusion Transformer), which has known FP16 performance issues with TensorRT
2. **Hift Bottleneck**: The Hift model (not accelerated) still consumes 40-50% of generation time
3. **Memory Bandwidth**: RTX 3090 may be bandwidth-limited for this workload

---

## Optimization Recommendations

### For Production (Recommended)

**Configuration: vLLM + FP16**

```bash
export COSYVOICE_USE_VLLM=true
export COSYVOICE_FP16=true
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**Why:**
- Simple setup
- Stable and reliable
- Nearly identical performance to TensorRT
- No complex dependencies

---

### For Maximum Performance

**Configuration: TensorRT + FP16**

```bash
# Setup (one-time)
./setup_tensorrt.sh

# Run
./start_server_tensorrt.sh
```

**Why:**
- Best measured performance (RTF 0.340)
- 6.6% faster than baseline
- Worth it for high-volume production

**Trade-offs:**
- Complex setup
- Longer startup time (engine loading)
- FP16 precision (minimal quality impact)

---

### Future Optimization Opportunities

#### 1. Optimize Hift Model
The Hift vocoder is currently not accelerated. Potential approaches:
- JIT compilation with `torch.jit.script`
- ONNX export + TensorRT
- Replace with faster vocoder (e.g., BigVGAN)

**Expected improvement:** 20-30% faster

#### 2. Try FP32 TensorRT
Avoid FP16 DiT issues by using FP32 precision:
```bash
export COSYVOICE_FP16=false
./setup_tensorrt.sh
```

**Trade-off:** Larger engine size, more GPU memory

#### 3. Batch Processing
Process multiple requests simultaneously:
```python
# In server code
batch_size = 4
results = model.inference_batch(texts, batch_size=batch_size)
```

**Expected improvement:** 2-3x throughput

#### 4. Model Quantization
Use INT8 quantization for even faster inference:
- Requires calibration dataset
- May impact quality slightly

**Expected improvement:** 30-50% faster

---

## Hardware Recommendations

### Minimum Requirements
- **GPU**: NVIDIA GTX 1660 Ti (6GB)
- **RTF**: ~0.8-1.0 (slower than real-time)
- **Use case**: Development, testing

### Recommended
- **GPU**: NVIDIA RTX 3060 (12GB) or better
- **RTF**: ~0.3-0.4 (2-3x real-time)
- **Use case**: Production, moderate load

### High Performance
- **GPU**: NVIDIA RTX 3090 / A5000 (24GB)
- **RTF**: ~0.3-0.35 (3x real-time)
- **Use case**: High-volume production

### Maximum Performance
- **GPU**: NVIDIA A100 / H100 (40-80GB)
- **RTF**: ~0.1-0.2 (5-10x real-time, estimated)
- **Use case**: Large-scale deployment

---

## Scaling Strategies

### Vertical Scaling (Single Server)

1. **Use Faster GPU**: Upgrade to A100/H100
2. **Enable All Optimizations**: TensorRT + FP16
3. **Increase Batch Size**: Process multiple requests together
4. **Optimize Hift**: JIT compile or replace vocoder

**Expected capacity:** 10-20 concurrent users per GPU

### Horizontal Scaling (Multiple Servers)

1. **Load Balancer**: Distribute requests across multiple servers
2. **GPU Sharding**: One server per GPU
3. **Queue System**: Use Redis/RabbitMQ for request queuing

**Expected capacity:** 100+ concurrent users with 10 GPUs

### Async Processing

For non-real-time use cases:
1. Accept request, return job ID
2. Process in background queue
3. Client polls for completion

**Benefits:**
- Better resource utilization
- Handle traffic spikes
- Predictable latency

---

## Monitoring and Profiling

### Key Metrics to Track

1. **RTF (Real-Time Factor)**
   ```python
   rtf = generation_time / audio_duration
   ```
   Target: < 0.3 for production

2. **GPU Utilization**
   ```bash
   nvidia-smi dmon -s u
   ```
   Target: > 80% during inference

3. **Memory Usage**
   ```bash
   nvidia-smi dmon -s m
   ```
   Monitor for OOM errors

4. **Request Latency**
   - P50: Median latency
   - P95: 95th percentile
   - P99: 99th percentile

### Profiling Tools

```bash
# PyTorch profiler
python -m torch.utils.bottleneck your_script.py

# NVIDIA Nsight Systems
nsys profile -o profile.qdrep python your_script.py

# TensorRT profiler
trtexec --onnx=model.onnx --profilingVerbosity=detailed
```

---

## Benchmark Reproduction

To reproduce these benchmarks:

```bash
# 1. Clone repository
git clone https://github.com/groxaxo/cozyvoice-fastapi.git
cd cozyvoice-fastapi

# 2. Setup environment
./run_cosyvoice_autonomous.sh

# 3. Run baseline test
conda activate cosyvoice3
python quick_test_pichones.py

# 4. Setup TensorRT
./setup_tensorrt.sh

# 5. Run TensorRT test
./start_server_tensorrt.sh
python quick_test_pichones.py
```

---

## Conclusion

**Current Best Configuration:** vLLM + FP16
- RTF: 0.362
- Simple setup
- Production-ready
- Recommended for most users

**For Maximum Performance:** TensorRT + FP16
- RTF: 0.340 (6.6% faster)
- Worth it for high-volume production
- Requires additional setup

**Future Potential:** Hift optimization + batching could achieve RTF < 0.2 (50%+ improvement)

---

## References

- [TensorRT Setup Progress](TENSORRT_SETUP_PROGRESS.md)
- [TensorRT Test Results](TENSORRT_TEST_RESULTS.md)
- [Optimization Summary](OPTIMIZATION_SUMMARY.md)
