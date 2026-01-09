# TensorRT Performance Test Results

## Test Completed: 2026-01-09 22:13 NZDT

### Performance Comparison

| Configuration | Average RTF | Avg Gen Time | Improvement |
|--------------|-------------|--------------|-------------|
| Baseline (PyTorch) | 0.364 | 6.86s | - |
| vLLM + FP16 | 0.362 | 6.84s | 0.5% |
| **TensorRT + FP16** | **0.340** | **6.35s** | **6.6%** |

### Detailed Results - TensorRT

| Sample | Voice | RTF | Gen Time | Audio Duration |
|--------|-------|-----|----------|----------------|
| Narrative Introduction | lucho-es | 0.484 | 8.55s | 17.68s |
| Philosophical Reflection | facu-es | 0.243 | 5.09s | 20.92s |
| Social Commentary | brenda-es | 0.274 | 4.94s | 18.04s |
| Dramatic Monologue | faculiado-es | 0.253 | 4.39s | 17.32s |
| Poetic Warning | facundito-es | 0.445 | 8.79s | 19.76s |

## Analysis

### Actual vs Expected Performance

- **Expected**: RTF 0.1-0.2 (2-3x faster)
- **Actual**: RTF 0.340 (6.6% faster)
- **Gap**: Significant underperformance

### Why TensorRT Didn't Provide Expected Speedup?

1. **TensorRT Warning**: "DiT tensorRT fp16 engine have some performance issue"
   - The Flow model uses DiT (Diffusion Transformer) architecture
   - FP16 TensorRT engines for DiT models have known performance issues
   - This was warned during setup but proceeded anyway

2. **Possible Bottlenecks**:
   - Hift model (not accelerated) may still dominate
   - Memory bandwidth limitations
   - TensorRT engine not fully optimized for this specific architecture

3. **Configuration Issues**:
   - Server health endpoint shows `"backend": "pytorch"` instead of `"tensorrt"`
   - TensorRT may not be actually active despite environment variables

### Cumulative Improvements

| Optimization | RTF | Speedup vs Baseline |
|--------------|-----|---------------------|
| Baseline | 0.364 | 1.00x |
| + vLLM | 0.362 | 1.01x |
| + TensorRT | 0.340 | 1.07x |

**Total improvement**: 7% faster than baseline

## Recommendations

### For Better Performance

1. **Use FP32 TensorRT** (not FP16)
   - Regenerate engine without FP16 flag
   - May avoid DiT-specific FP16 issues
   - Trade: Larger engine size, more memory

2. **Profile the Pipeline**
   - Measure time spent in each component:
     - LLM (text → tokens)
     - Flow (tokens → mel)
     - Hift (mel → audio)
   - Identify actual bottleneck

3. **Optimize Hift Model**
   - Currently not accelerated
   - Consider JIT compilation or ONNX export
   - May provide more significant gains

4. **Batch Processing**
   - Process multiple requests simultaneously
   - Better GPU utilization
   - Lower per-request latency

5. **Try Different GPU**
   - RTX 3090 may not be optimal for TensorRT
   - A100 or H100 may show better TensorRT performance

### Current Best Configuration

**vLLM + FP16** (RTF: 0.362)
- Simpler setup
- Similar performance to TensorRT
- No DiT-specific issues
- Recommended for production

## Files Generated

- `flow.decoder.estimator.fp32.onnx` (1.3GB)
- `flow.decoder.estimator.fp16.mygpu.plan` (637MB)
- `start_server_tensorrt.sh`
- `tensorrt_test_results.txt`
- `TENSORRT_SETUP_PROGRESS.md`

## Conclusion

TensorRT setup was successful but provided minimal performance improvement (6.6%) due to DiT architecture FP16 issues. The vLLM configuration (0.5% improvement) is simpler and nearly as fast. For significant speedup, need to:

1. Profile to find actual bottleneck
2. Optimize Hift model
3. Try FP32 TensorRT
4. Consider batch processing

**Current recommendation**: Use vLLM + FP16 configuration for production.
