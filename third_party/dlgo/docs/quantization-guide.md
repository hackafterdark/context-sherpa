# Quantization Guide

dlgo supports all standard GGML quantization formats. However, not all formats have
optimized SIMD kernels — choosing the right quantization has a **direct impact on
inference throughput**.

## Performance Tiers

### Tier 1: QxQ Integer SIMD (recommended)

These have dedicated AVX2+VNNI integer dot product kernels. The input activation vector
is quantized to Q8 format once, then all dot products run as integer SIMD
(`_mm256_dpbusd_epi32` / `_mm256_maddubs_epi16`), processing 32 values per instruction.
These also have batch prefill kernels that amortize weight unpacking across multiple
input positions.

| GGML Type | ID | Name | Bits/value | Block Size | Q8 Pairing |
|-----------|----|------|------------|------------|------------|
| Q4_0 | 2 | Basic 4-bit | 4.5 | 32 | Q8_0 |
| Q5_0 | 6 | Basic 5-bit | 5.5 | 32 | Q8_0 |
| Q8_0 | 8 | Basic 8-bit | 8.5 | 32 | Q8_0 |
| Q2_K | 10 | K-quant 2-bit | 2.6 | 256 | Q8_K |
| Q3_K | 11 | K-quant 3-bit | 3.4 | 256 | Q8_K |
| Q4_K | 12 | K-quant 4-bit | 4.5 | 256 | Q8_K |
| Q5_K | 13 | K-quant 5-bit | 5.5 | 256 | Q8_K |
| Q6_K | 14 | K-quant 6-bit | 6.6 | 256 | Q8_K |

**All common GGUF quantization names map to Tier 1:**
Q4_0, Q8_0, Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K.

### Tier 2: Fused Float SIMD (functional but slower)

These have AVX2+FMA kernels that dequantize on-the-fly to float32 and use FMA
accumulation. About 2–4x slower than the QxQ path because they operate on float32
(8 values per FMA) instead of int8 (32 values per maddubs).

| GGML Type | ID | Name |
|-----------|----|------|
| F16 | 1 | Float16 |
| Q4_1 | 3 | Basic 4-bit with min |
| Q5_1 | 7 | Basic 5-bit with min |

### Tier 3: Dequantize + Dot (avoid for large models)

Everything else goes through: dequantize entire row to float32, then SIMD dot product.
This is the slowest path and includes I-quants (IQ2_XXS through IQ4_XS), ternary
quants (TQ1_0, TQ2_0), and BF16.

## Mixed-Quantization Models

The `_S`, `_M`, `_L` suffixes (e.g., Q4_K_M) indicate mixed quantization — different
layers use different types for a quality/size tradeoff:

- **Q4_K_M**: Q4_K for most layers, Q6_K for critical layers, Q5_0 for embeddings/output
- **Q4_K_S**: Q4_K for all layers, Q5_0 for embeddings/output
- **Q3_K_L**: Q3_K for most layers, Q5_K for some, Q6_K for critical layers

All component types in these schemes (Q4_K, Q5_K, Q6_K, Q5_0) are Tier 1, so
mixed-quant models are fully optimized.

## Avoiding Quantization Overhead

The QxQ path requires quantizing the input activation vector from float32 to Q8 before
each matrix multiplication. This is O(dim) work per matmul. To avoid redundant
quantization:

1. **Fused operations quantize once, share across matrices.** `QTripleMatVecMulParallel`
   (Q/K/V attention) and `QDualMatVecMulParallel` (FFN gate+up) quantize the input
   once and dispatch all matrices in a single work pool wave.

2. **Fusion requires matching types.** If matrices have different quant types (e.g.,
   Q4_K and Q6_K), they fall back to separate quantizations. This is correct — Q4_K
   needs Q8_K format while Q4_0 needs Q8_0 format.

3. **Q8 buffers are pooled.** `sync.Pool` reuses Q8 buffers across calls to avoid
   allocation pressure.

4. **Quantization cost is negligible for generation.** For dim=2048, quantization
   takes ~100ns vs. ~140μs for a 5632-row matmul (0.07%). It only matters if done
   redundantly.

## Recommendations

| Use Case | Recommended Quant | Why |
|----------|-------------------|-----|
| Best throughput, acceptable quality | Q4_0, Q4_K_M | Smallest Tier 1 types |
| Best quality per bit | Q4_K_M, Q5_K_M | K-quants have better quality curves |
| Maximum quality | Q8_0, Q6_K | Near-lossless, still fast |
| Avoid | IQ*, TQ*, BF16 | No optimized kernels |
