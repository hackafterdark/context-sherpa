//go:build amd64 && cgo

// simd_dot.c — AVX2+FMA fused dequantize-dot-product for GGUF quantized types.
// Called from Go via CGo. These replace the hot inner loop of quantized matmul.
//
// Each function computes: sum_i( dequant(q[i]) * x[i] ) for one row, using
// AVX2 integer/float conversions and FMA accumulation.

// Enable AVX2/FMA/F16C at the function level via pragmas,
// so we don't need -mavx2 -mfma -mf16c in CFLAGS (avoids CGo flag restrictions).
#pragma GCC target("avx2,fma,f16c,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")
#pragma GCC optimize("O3")

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ── Helper: convert float16 (binary16) to float32 ──────────────
// Uses F16C intrinsic (SSE extension, always present with AVX2).
static inline float f16_to_f32(uint16_t h) {
    __m128i v = _mm_set1_epi16(h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

// ── Q4_0: 32 values per 18-byte block ──────────────────────────
// Layout: f16 scale + 16 nibble bytes
// dequant: (nibble - 8) * scale
float vec_dot_q4_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    
    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 18;
        float d = f16_to_f32(*(const uint16_t*)block);
        __m256 vd = _mm256_set1_ps(d);
        const uint8_t* qs = block + 2;
        const float* xp = x + b * 32;
        
        // Process 32 values: first 16 from low nibbles, next 16 from high nibbles
        // But they interleave: byte j has value j (low nibble) and j+16 (high nibble)
        for (int j = 0; j < 16; j += 8) {
            // Load 8 bytes, extract low and high nibbles
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_srli_epi16(bytes, 4);
            hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
            
            // Convert to int32 and subtract 8
            __m256i lo32 = _mm256_cvtepu8_epi32(lo);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi);
            __m256i eight = _mm256_set1_epi32(8);
            lo32 = _mm256_sub_epi32(lo32, eight);
            hi32 = _mm256_sub_epi32(hi32, eight);
            
            // Convert to float, multiply by scale and x
            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);
            
            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);
            
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi), xhi, acc);
        }
    }
    
    // Horizontal sum
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Batch version: process multiple rows in one CGo call ───────

// F16 row dot: n half-precision values against n float32 values.
float vec_dot_f16(const uint8_t* data, const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        uint16_t h = *(const uint16_t*)(data + i * 2);
        sum += f16_to_f32(h) * x[i];
    }
    return sum;
}

void vec_dot_f16_batch(const uint8_t* data, const float* x, int n,
                        float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_f16(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q4_0_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q4_0(data + (size_t)r * bpr, x, n);
    }
}

// ── Q8_0: 32 values per 34-byte block ──────────────────────────
// Layout: f16 scale + 32 int8 quants
// dequant: int8(q) * scale
// Optimized: 2-block unrolled loop for better ILP. Each block computes
// dot(int8_quants, x) then scales once (saves 3 vmulps per block).
float vec_dot_q8_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int b = 0;
    // Process 2 blocks at a time for better instruction-level parallelism
    for (; b <= nb - 2; b += 2) {
        // Block 0
        const uint8_t* block0 = data + b * 34;
        float d0 = f16_to_f32(*(const uint16_t*)block0);
        const int8_t* qs0 = (const int8_t*)(block0 + 2);
        const float* xp0 = x + b * 32;

        // Block 1
        const uint8_t* block1 = data + (b + 1) * 34;
        float d1 = f16_to_f32(*(const uint16_t*)block1);
        const int8_t* qs1 = (const int8_t*)(block1 + 2);
        const float* xp1 = x + (b + 1) * 32;

        // Prefetch next pair
        if (b + 3 < nb) {
            _mm_prefetch((const char*)(data + (b + 2) * 34), _MM_HINT_T0);
            _mm_prefetch((const char*)(data + (b + 3) * 34), _MM_HINT_T0);
        }

        __m256 bd0 = _mm256_setzero_ps();
        __m256 bd1 = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            __m128i bytes0 = _mm_loadl_epi64((const __m128i*)(qs0 + j));
            __m256i i32_0 = _mm256_cvtepi8_epi32(bytes0);
            __m256 fval0 = _mm256_cvtepi32_ps(i32_0);
            bd0 = _mm256_fmadd_ps(fval0, _mm256_loadu_ps(xp0 + j), bd0);

            __m128i bytes1 = _mm_loadl_epi64((const __m128i*)(qs1 + j));
            __m256i i32_1 = _mm256_cvtepi8_epi32(bytes1);
            __m256 fval1 = _mm256_cvtepi32_ps(i32_1);
            bd1 = _mm256_fmadd_ps(fval1, _mm256_loadu_ps(xp1 + j), bd1);
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), bd0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), bd1, acc1);
    }

    // Handle remaining single block
    for (; b < nb; b++) {
        const uint8_t* block = data + b * 34;
        float d = f16_to_f32(*(const uint16_t*)block);
        const int8_t* qs = (const int8_t*)(block + 2);
        const float* xp = x + b * 32;

        __m256 block_dot = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m256i i32 = _mm256_cvtepi8_epi32(bytes);
            __m256 fval = _mm256_cvtepi32_ps(i32);
            block_dot = _mm256_fmadd_ps(fval, _mm256_loadu_ps(xp + j), block_dot);
        }
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d), block_dot, acc0);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 hi128 = _mm256_extractf128_ps(acc0, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc0);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q4_1: 32 values per 20-byte block ──────────────────────────
// Layout: f16 scale + f16 min + 16 nibble bytes
// dequant: nibble * scale + min
float vec_dot_q4_1(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 20;
        float d = f16_to_f32(*(const uint16_t*)block);
        float m = f16_to_f32(*(const uint16_t*)(block + 2));
        __m256 vd = _mm256_set1_ps(d);
        __m256 vm = _mm256_set1_ps(m);
        const uint8_t* qs = block + 4;
        const float* xp = x + b * 32;

        for (int j = 0; j < 16; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_srli_epi16(bytes, 4);
            hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));

            __m256i lo32 = _mm256_cvtepu8_epi32(lo);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi);

            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);

            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);

            // val = d * q + m; dot += val * x = d*q*x + m*x
            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, flo, vm), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, fhi, vm), xhi, acc);
        }
    }

    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q5_0: 32 values per 22-byte block ──────────────────────────
// Layout: f16 scale + 4 bytes qh (5th bits) + 16 nibble bytes
// dequant: ((4bits | 5thbit<<4) - 16) * scale
float vec_dot_q5_0(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();
    __m256i sixteen = _mm256_set1_epi32(16);

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 22;
        float d = f16_to_f32(*(const uint16_t*)block);
        __m256 vd = _mm256_set1_ps(d);
        uint32_t qh;
        memcpy(&qh, block + 2, 4);
        const uint8_t* qs = block + 6;
        const float* xp = x + b * 32;

        for (int j = 0; j < 16; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo_nib = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_srli_epi16(bytes, 4);
            hi_nib = _mm_and_si128(hi_nib, _mm_set1_epi8(0x0F));

            __m256i lo32 = _mm256_cvtepu8_epi32(lo_nib);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi_nib);

            // Extract 5th bits from qh for positions j..j+7 and j+16..j+23
            uint8_t h_lo[8], h_hi[8];
            for (int k = 0; k < 8; k++) {
                h_lo[k] = ((qh >> (j + k)) & 1) ? 16 : 0;
                h_hi[k] = ((qh >> (j + k + 16)) & 1) ? 16 : 0;
            }
            __m128i hlo_128 = _mm_loadl_epi64((const __m128i*)h_lo);
            __m128i hhi_128 = _mm_loadl_epi64((const __m128i*)h_hi);
            __m256i hlo32 = _mm256_cvtepu8_epi32(hlo_128);
            __m256i hhi32 = _mm256_cvtepu8_epi32(hhi_128);

            lo32 = _mm256_or_si256(lo32, hlo32);
            hi32 = _mm256_or_si256(hi32, hhi32);
            lo32 = _mm256_sub_epi32(lo32, sixteen);
            hi32 = _mm256_sub_epi32(hi32, sixteen);

            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);
            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);

            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, flo), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, fhi), xhi, acc);
        }
    }

    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q5_1: 32 values per 24-byte block ──────────────────────────
// Layout: f16 scale + f16 min + 4 bytes qh + 16 nibble bytes
// dequant: (4bits | 5thbit<<4) * scale + min
float vec_dot_q5_1(const uint8_t* data, const float* x, int n) {
    int nb = n / 32;
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        const uint8_t* block = data + b * 24;
        float d = f16_to_f32(*(const uint16_t*)block);
        float m = f16_to_f32(*(const uint16_t*)(block + 2));
        __m256 vd = _mm256_set1_ps(d);
        __m256 vm = _mm256_set1_ps(m);
        uint32_t qh;
        memcpy(&qh, block + 4, 4);
        const uint8_t* qs = block + 8;
        const float* xp = x + b * 32;

        for (int j = 0; j < 16; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m128i lo_nib = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_srli_epi16(bytes, 4);
            hi_nib = _mm_and_si128(hi_nib, _mm_set1_epi8(0x0F));

            __m256i lo32 = _mm256_cvtepu8_epi32(lo_nib);
            __m256i hi32 = _mm256_cvtepu8_epi32(hi_nib);

            uint8_t h_lo[8], h_hi[8];
            for (int k = 0; k < 8; k++) {
                h_lo[k] = ((qh >> (j + k)) & 1) ? 16 : 0;
                h_hi[k] = ((qh >> (j + k + 16)) & 1) ? 16 : 0;
            }
            __m128i hlo_128 = _mm_loadl_epi64((const __m128i*)h_lo);
            __m128i hhi_128 = _mm_loadl_epi64((const __m128i*)h_hi);
            __m256i hlo32 = _mm256_cvtepu8_epi32(hlo_128);
            __m256i hhi32 = _mm256_cvtepu8_epi32(hhi_128);

            lo32 = _mm256_or_si256(lo32, hlo32);
            hi32 = _mm256_or_si256(hi32, hhi32);

            __m256 flo = _mm256_cvtepi32_ps(lo32);
            __m256 fhi = _mm256_cvtepi32_ps(hi32);
            __m256 xlo = _mm256_loadu_ps(xp + j);
            __m256 xhi = _mm256_loadu_ps(xp + j + 16);

            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, flo, vm), xlo, acc);
            acc = _mm256_fmadd_ps(_mm256_fmadd_ps(vd, fhi, vm), xhi, acc);
        }
    }

    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q2_K: 256 values per 84-byte block ─────────────────────────
float vec_dot_q2_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 84;
        const uint8_t* scales = bp;
        const uint8_t* qs = bp + 16;
        float d = f16_to_f32(*(const uint16_t*)(bp + 80));
        float dmin = f16_to_f32(*(const uint16_t*)(bp + 82));
        const float* xp = x + block * 256;
        
        int xi = 0;
        int is = 0;
        const uint8_t* qptr = qs;
        for (int n128 = 0; n128 < 2; n128++) {
            for (int j = 0; j < 4; j++) {
                uint32_t shift = j * 2;
                
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0xF);
                float ml = dmin * (float)(sc >> 4);
                __m256 vdl = _mm256_set1_ps(dl);
                __m256 vml = _mm256_set1_ps(ml);
                
                for (int l = 0; l < 16; l += 8) {
                    // Extract 2-bit quants
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qptr + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    // val = dl * q - ml
                    __m256 val = _mm256_fmsub_ps(vdl, fq, vml);
                    acc = _mm256_fmadd_ps(val, xv, acc);
                }
                xi += 16;
                
                sc = scales[is++];
                dl = d * (float)(sc & 0xF);
                ml = dmin * (float)(sc >> 4);
                vdl = _mm256_set1_ps(dl);
                vml = _mm256_set1_ps(ml);
                
                for (int l = 0; l < 16; l += 8) {
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qptr + 16 + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    __m256 val = _mm256_fmsub_ps(vdl, fq, vml);
                    acc = _mm256_fmadd_ps(val, xv, acc);
                }
                xi += 16;
            }
            qptr += 32;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q3_K: 256 values per 110-byte block ────────────────────────
float vec_dot_q3_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 110;
        float dAll = f16_to_f32(*(const uint16_t*)(bp + 108));
        
        const uint8_t* hm = bp;
        const uint8_t* qp = bp + 32;
        const uint8_t* scp = bp + 96;
        const float* xp = x + block * 256;
        
        // Unpack 16 x 6-bit scales
        uint32_t aux[4] = {0, 0, 0, 0};
        for (int i = 0; i < 12; i++) {
            aux[i/4] |= (uint32_t)scp[i] << ((i%4)*8);
        }
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
        aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
        aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
        
        int8_t scales[16];
        memcpy(scales, aux, 16);
        
        int xi = 0;
        int is = 0;
        uint8_t m = 1;
        
        for (int n128 = 0; n128 < 2; n128++) {
            for (int j = 0; j < 4; j++) {
                uint32_t shift = j * 2;
                float dl = dAll * (float)(scales[is] - 32);
                is++;
                __m256 vdl = _mm256_set1_ps(dl);
                
                // Process 16 values in 2 groups of 8
                for (int l = 0; l < 16; l += 8) {
                    // Load 8 bytes of qs, shift and mask to get 2-bit quants
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qp + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    
                    // Load 8 hmask bytes, test bit m
                    __m128i hbytes = _mm_loadl_epi64((const __m128i*)(hm + l));
                    __m128i mbyte = _mm_set1_epi8(m);
                    __m128i htst = _mm_cmpeq_epi8(_mm_and_si128(hbytes, mbyte), _mm_setzero_si128());
                    // htst = 0xFF where hmask bit is 0 (subtract 4), 0 where set (subtract 0)
                    __m256i h32 = _mm256_cvtepi8_epi32(htst);
                    // h32 is -1 where we need to subtract 4, 0 otherwise
                    __m256i four = _mm256_set1_epi32(4);
                    __m256i hbits = _mm256_and_si256(h32, four); // 4 or 0
                    q32 = _mm256_sub_epi32(q32, hbits);
                    
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, fq), xv, acc);
                }
                xi += 16;
                
                dl = dAll * (float)(scales[is] - 32);
                is++;
                vdl = _mm256_set1_ps(dl);
                
                for (int l = 0; l < 16; l += 8) {
                    __m128i raw = _mm_loadl_epi64((const __m128i*)(qp + 16 + l));
                    __m128i shifted = _mm_srli_epi16(raw, shift);
                    __m128i masked = _mm_and_si128(shifted, _mm_set1_epi8(3));
                    __m256i q32 = _mm256_cvtepu8_epi32(masked);
                    
                    __m128i hbytes = _mm_loadl_epi64((const __m128i*)(hm + 16 + l));
                    __m128i mbyte = _mm_set1_epi8(m);
                    __m128i htst = _mm_cmpeq_epi8(_mm_and_si128(hbytes, mbyte), _mm_setzero_si128());
                    __m256i h32 = _mm256_cvtepi8_epi32(htst);
                    __m256i four = _mm256_set1_epi32(4);
                    __m256i hbits = _mm256_and_si256(h32, four);
                    q32 = _mm256_sub_epi32(q32, hbits);
                    
                    __m256 fq = _mm256_cvtepi32_ps(q32);
                    __m256 xv = _mm256_loadu_ps(xp + xi + l);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(vdl, fq), xv, acc);
                }
                xi += 16;
                
                m <<= 1;
            }
            qp += 32;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q4_K: 256 values per 144-byte block ────────────────────────
float vec_dot_q4_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 144;
        float d = f16_to_f32(*(const uint16_t*)bp);
        float dmin = f16_to_f32(*(const uint16_t*)(bp + 2));
        
        const uint8_t* sp = bp + 4;
        float sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i] = d * (float)(sp[i] & 0x3F);
            mn[i] = dmin * (float)(sp[4+i] & 0x3F);
        }
        for (int i = 0; i < 4; i++) {
            uint8_t scHi = (sp[i] >> 6) & 0x03;
            uint8_t mnHi = (sp[4+i] >> 6) & 0x03;
            uint8_t scLo = sp[8+i] & 0x0F;
            uint8_t mnLo = (sp[8+i] >> 4) & 0x0F;
            sc[4+i] = d * (float)(scLo | (scHi << 4));
            mn[4+i] = dmin * (float)(mnLo | (mnHi << 4));
        }
        
        const uint8_t* qp = bp + 16;
        const float* xp = x + block * 256;
        int xi = 0;
        int is = 0;
        
        for (int grp = 0; grp < 4; grp++) {
            float d1 = sc[is], m1 = mn[is];
            float d2 = sc[is+1], m2 = mn[is+1];
            __m256 vd1 = _mm256_set1_ps(d1);
            __m256 vm1 = _mm256_set1_ps(m1);
            __m256 vd2 = _mm256_set1_ps(d2);
            __m256 vm2 = _mm256_set1_ps(m2);
            
            const uint8_t* qOff = qp + grp * 32;
            
            // First 32: low nibbles
            for (int l = 0; l < 32; l += 8) {
                __m128i bytes = _mm_loadl_epi64((const __m128i*)(qOff + l));
                __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(lo);
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd1, fq, vm1);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            
            // Next 32: high nibbles  
            for (int l = 0; l < 32; l += 8) {
                __m128i bytes = _mm_loadl_epi64((const __m128i*)(qOff + l));
                __m128i hi = _mm_srli_epi16(bytes, 4);
                hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(hi);
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd2, fq, vm2);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            is += 2;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q5_K: 256 values per 176-byte block ────────────────────────
float vec_dot_q5_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 176;
        float d = f16_to_f32(*(const uint16_t*)bp);
        float dmin = f16_to_f32(*(const uint16_t*)(bp + 2));
        
        const uint8_t* sp = bp + 4;
        float sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i] = d * (float)(sp[i] & 0x3F);
            mn[i] = dmin * (float)(sp[4+i] & 0x3F);
        }
        for (int i = 0; i < 4; i++) {
            uint8_t scHi = (sp[i] >> 6) & 0x03;
            uint8_t mnHi = (sp[4+i] >> 6) & 0x03;
            uint8_t scLo = sp[8+i] & 0x0F;
            uint8_t mnLo = (sp[8+i] >> 4) & 0x0F;
            sc[4+i] = d * (float)(scLo | (scHi << 4));
            mn[4+i] = dmin * (float)(mnLo | (mnHi << 4));
        }
        
        const uint8_t* qh = bp + 16;
        const uint8_t* qs = bp + 48;
        const float* xp = x + block * 256;
        int xi = 0;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        
        for (int grp = 0; grp < 4; grp++) {
            float d1 = sc[is], m1 = mn[is];
            float d2 = sc[is+1], m2 = mn[is+1];
            __m256 vd1 = _mm256_set1_ps(d1);
            __m256 vm1 = _mm256_set1_ps(m1);
            __m256 vd2 = _mm256_set1_ps(d2);
            __m256 vm2 = _mm256_set1_ps(m2);
            const uint8_t* qlOff = qs + grp * 32;
            
            // First 32: low nibbles + 5th bit
            for (int l = 0; l < 32; l += 8) {
                __m128i raw = _mm_loadl_epi64((const __m128i*)(qlOff + l));
                __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(lo);
                
                // Add 5th bit from qh
                __m128i hbytes = _mm_loadl_epi64((const __m128i*)(qh + l));
                __m128i hmask = _mm_set1_epi8(u1);
                __m128i htst = _mm_and_si128(hbytes, hmask);
                // Non-zero where bit set → need to OR 16
                __m128i hcmp = _mm_cmpeq_epi8(htst, _mm_setzero_si128());
                // hcmp = 0xFF where bit is NOT set, 0 where it IS set
                // We want 16 where set, 0 where not → invert
                __m128i hset = _mm_andnot_si128(hcmp, _mm_set1_epi8(16));
                __m256i h32 = _mm256_cvtepu8_epi32(hset);
                q32 = _mm256_or_si256(q32, h32);
                
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd1, fq, vm1);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            
            // Next 32: high nibbles + 5th bit
            for (int l = 0; l < 32; l += 8) {
                __m128i raw = _mm_loadl_epi64((const __m128i*)(qlOff + l));
                __m128i hi = _mm_srli_epi16(raw, 4);
                hi = _mm_and_si128(hi, _mm_set1_epi8(0x0F));
                __m256i q32 = _mm256_cvtepu8_epi32(hi);
                
                __m128i hbytes = _mm_loadl_epi64((const __m128i*)(qh + l));
                __m128i hmask = _mm_set1_epi8(u2);
                __m128i htst = _mm_and_si128(hbytes, hmask);
                __m128i hcmp = _mm_cmpeq_epi8(htst, _mm_setzero_si128());
                __m128i hset = _mm_andnot_si128(hcmp, _mm_set1_epi8(16));
                __m256i h32 = _mm256_cvtepu8_epi32(hset);
                q32 = _mm256_or_si256(q32, h32);
                
                __m256 fq = _mm256_cvtepi32_ps(q32);
                __m256 xv = _mm256_loadu_ps(xp + xi + l);
                __m256 val = _mm256_fmsub_ps(vd2, fq, vm2);
                acc = _mm256_fmadd_ps(val, xv, acc);
            }
            xi += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ── Q6_K: 256 values per 210-byte block ────────────────────────
float vec_dot_q6_k(const uint8_t* data, const float* x, int n) {
    int nb = n / 256;
    __m256 acc = _mm256_setzero_ps();
    
    for (int block = 0; block < nb; block++) {
        const uint8_t* bp = data + block * 210;
        float d_val = f16_to_f32(*(const uint16_t*)(bp + 208));
        const uint8_t* ql = bp;
        const uint8_t* qh_base = bp + 128;
        const int8_t* scales = (const int8_t*)(bp + 192);
        const float* xp = x + block * 256;
        __m256 vd = _mm256_set1_ps(d_val);
        __m256i thirty_two = _mm256_set1_epi32(32);
        
        for (int n128 = 0; n128 < 2; n128++) {
            const uint8_t* qlp = ql + n128 * 64;
            const uint8_t* qhp = qh_base + n128 * 32;
            int xoff = n128 * 128;
            
            // Process 8 positions at a time (each produces 4 values at l, l+32, l+64, l+96)
            for (int l = 0; l < 32; l += 8) {
                int sidx = 8 * n128 + l / 16;

                // Load 8 ql bytes at [l] and [l+32], and 8 qh bytes
                __m128i ql0_raw = _mm_loadl_epi64((const __m128i*)(qlp + l));
                __m128i ql32_raw = _mm_loadl_epi64((const __m128i*)(qlp + l + 32));
                __m128i qh_raw = _mm_loadl_epi64((const __m128i*)(qhp + l));
                
                __m128i mask4 = _mm_set1_epi8(0x0F);
                __m128i mask2 = _mm_set1_epi8(0x03);
                
                // q1 = (ql0 & 0xF) | ((qh >> 0) & 3) << 4 - 32
                __m128i q1_lo = _mm_and_si128(ql0_raw, mask4);
                __m128i q1_hi = _mm_slli_epi16(_mm_and_si128(qh_raw, mask2), 4);
                __m128i q1_8 = _mm_or_si128(q1_lo, q1_hi);
                __m256i q1_32 = _mm256_cvtepu8_epi32(q1_8);
                q1_32 = _mm256_sub_epi32(q1_32, thirty_two);
                
                // q2 = (ql32 & 0xF) | ((qh >> 2) & 3) << 4 - 32
                __m128i q2_lo = _mm_and_si128(ql32_raw, mask4);
                __m128i qh_s2 = _mm_srli_epi16(qh_raw, 2);
                __m128i q2_hi = _mm_slli_epi16(_mm_and_si128(qh_s2, mask2), 4);
                __m128i q2_8 = _mm_or_si128(q2_lo, q2_hi);
                __m256i q2_32 = _mm256_cvtepu8_epi32(q2_8);
                q2_32 = _mm256_sub_epi32(q2_32, thirty_two);
                
                // q3 = (ql0 >> 4) | ((qh >> 4) & 3) << 4 - 32
                __m128i q3_lo = _mm_and_si128(_mm_srli_epi16(ql0_raw, 4), mask4);
                __m128i qh_s4 = _mm_srli_epi16(qh_raw, 4);
                __m128i q3_hi = _mm_slli_epi16(_mm_and_si128(qh_s4, mask2), 4);
                __m128i q3_8 = _mm_or_si128(q3_lo, q3_hi);
                __m256i q3_32 = _mm256_cvtepu8_epi32(q3_8);
                q3_32 = _mm256_sub_epi32(q3_32, thirty_two);
                
                // q4 = (ql32 >> 4) | ((qh >> 6) & 3) << 4 - 32
                __m128i q4_lo = _mm_and_si128(_mm_srli_epi16(ql32_raw, 4), mask4);
                __m128i qh_s6 = _mm_srli_epi16(qh_raw, 6);
                __m128i q4_hi = _mm_slli_epi16(_mm_and_si128(qh_s6, mask2), 4);
                __m128i q4_8 = _mm_or_si128(q4_lo, q4_hi);
                __m256i q4_32 = _mm256_cvtepu8_epi32(q4_8);
                q4_32 = _mm256_sub_epi32(q4_32, thirty_two);
                
                // Scale broadcasts — all 8 values at position l..l+7 use the same scale
                // (since l/16 is constant within each group of 16)
                // First 8 values use sidx, next 8 may use sidx+1 if l crosses 16 boundary
                __m256 vs0 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx]));
                __m256 vs2 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 2]));
                __m256 vs4 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 4]));
                __m256 vs6 = _mm256_mul_ps(vd, _mm256_set1_ps((float)scales[sidx + 6]));
                
                // Load x at 4 scattered positions
                __m256 x0 = _mm256_loadu_ps(xp + xoff + l);
                __m256 x32 = _mm256_loadu_ps(xp + xoff + l + 32);
                __m256 x64 = _mm256_loadu_ps(xp + xoff + l + 64);
                __m256 x96 = _mm256_loadu_ps(xp + xoff + l + 96);
                
                // Accumulate: d * scale * q * x
                __m256 fq1 = _mm256_cvtepi32_ps(q1_32);
                __m256 fq2 = _mm256_cvtepi32_ps(q2_32);
                __m256 fq3 = _mm256_cvtepi32_ps(q3_32);
                __m256 fq4 = _mm256_cvtepi32_ps(q4_32);
                
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs0, fq1), x0, acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs2, fq2), x32, acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs4, fq3), x64, acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(vs6, fq4), x96, acc);
            }
        }
    }
    
    __m128 hi128 = _mm256_extractf128_ps(acc, 1);
    __m128 lo128 = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo128, hi128);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

// ══════════════════════════════════════════════════════════════════
// Batch versions: process nrows in one CGo call to amortize overhead
// ══════════════════════════════════════════════════════════════════

void vec_dot_q8_0_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q8_0(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q4_1_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q4_1(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q5_0_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q5_0(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q5_1_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q5_1(data + (size_t)r * bpr, x, n);
    }
}
void vec_dot_q2_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q2_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q3_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q3_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q4_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q4_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q5_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q5_k(data + (size_t)r * bpr, x, n);
    }
}

void vec_dot_q6_k_batch(const uint8_t* data, const float* x, int n,
                         float* out, int nrows, int bpr) {
    for (int r = 0; r < nrows; r++) {
        if (r + 1 < nrows)
            _mm_prefetch((const char*)(data + (size_t)(r+1) * bpr), _MM_HINT_T0);
        out[r] = vec_dot_q6_k(data + (size_t)r * bpr, x, n);
    }
}

// ── AVX2 fast exp approximation ────────────────────────────────
// Range reduction: exp(x) = 2^n * exp(r) where n=round(x/ln2), r=x-n*ln2
// Polynomial minimax for exp(r), r in [-0.5*ln2, 0.5*ln2].
// Accuracy: ~5 decimal digits (sufficient for SiLU activation).
static inline __m256 fast_exp_avx2(__m256 x) {
    // Clamp to avoid overflow/underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    __m256 log2e = _mm256_set1_ps(1.44269504088896f);
    __m256 ln2   = _mm256_set1_ps(0.6931471805599453f);

    // n = round(x / ln2)
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, log2e),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // r = x - n*ln2
    __m256 r = _mm256_fnmadd_ps(n, ln2, x);

    // Horner's: exp(r) ≈ 1 + r(1 + r(0.5 + r(1/6 + r/24)))
    __m256 p = _mm256_set1_ps(1.0f / 24.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

    // 2^n via IEEE 754 bit manipulation
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(p, pow2n);
}

// ── Fused SwiGLU: out[i] = SiLU(gate[i]) * up[i] ─────────────
// SiLU(x) = x / (1 + exp(-x))
// Uses fast AVX2 exp + Newton-Raphson reciprocal for high throughput.
void vec_swiglu(float* out, const float* gate, const float* up, int n) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_one = _mm256_set1_ps(-1.0f);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        // exp(-g)
        __m256 neg_g = _mm256_mul_ps(neg_one, g);
        __m256 exp_neg_g = fast_exp_avx2(neg_g);
        // sigmoid(g) = 1 / (1 + exp(-g)) — fast reciprocal + Newton-Raphson
        __m256 denom = _mm256_add_ps(one, exp_neg_g);
        __m256 rcp = _mm256_rcp_ps(denom);
        // One Newton-Raphson step: rcp = rcp * (2 - denom*rcp)
        __m256 two = _mm256_set1_ps(2.0f);
        rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
        // SiLU(g) * up = g * sigmoid(g) * up
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(g, rcp), u);
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < n; i++) {
        float g = gate[i];
        float eg = expf(-g);
        out[i] = g / (1.0f + eg) * up[i];
    }
}

// ── AVX2 Softmax: out[i] = exp(x[i]-max) / sum(exp(x[j]-max)) ──
// In-place softmax using fast exp approximation.
void vec_softmax(float* x, int n) {
    // Find max
    int i = 0;
    __m256 vmax = _mm256_set1_ps(-1e30f);
    for (; i <= n - 8; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    }
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
    float maxVal = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        if (x[i] > maxVal) maxVal = x[i];
    }

    // exp(x[i] - max) and sum
    __m256 voff = _mm256_set1_ps(maxVal);
    __m256 vsum = _mm256_setzero_ps();
    i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + i), voff));
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm256_castps256_ps128(vsum);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    // Normalize
    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i <= n - 8; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    }
    for (; i < n; i++) {
        x[i] /= sum;
    }
}

// ── Float32 dot product using AVX2+FMA ─────────────────────────
// Matches gollm's DotProductAVX2 assembly: processes 16 floats/iter
// with two FMA accumulators, then horizontal sum.
float vec_dot_f32(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    for (; i <= n - 8; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Batch version: compute dot products for nrows rows of a matrix against x.
// a_flat is [nrows * cols] float32, x is [cols] float32, out is [nrows] float32.
void vec_dot_f32_batch(const float* a_flat, const float* x, int cols,
                       float* out, int nrows) {
    for (int r = 0; r < nrows; r++) {
        out[r] = vec_dot_f32(a_flat + r * cols, x, cols);
    }
}

// ── SIMD scale-add (axpy): out[i] += scale * src[i] ───────────
// Used for attention weighted value accumulation.
void vec_scale_add(float* out, float scale, const float* src, int n) {
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 o = _mm256_loadu_ps(out + i);
        __m256 s = _mm256_loadu_ps(src + i);
        o = _mm256_fmadd_ps(vs, s, o);
        _mm256_storeu_ps(out + i, o);
    }
    for (; i < n; i++) {
        out[i] += scale * src[i];
    }
}
