#ifndef DLGO_VULKAN_GPU_H
#define DLGO_VULKAN_GPU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define GPU_OK                0
#define GPU_ERR_NO_VULKAN    -1
#define GPU_ERR_NO_DEVICE    -2
#define GPU_ERR_INIT_FAIL    -3
#define GPU_ERR_OOM          -4
#define GPU_ERR_SHADER       -5
#define GPU_ERR_DISPATCH     -6

// Buffer usage flags
#define GPU_BUF_STORAGE      1
#define GPU_BUF_UNIFORM      2

// Quantization type IDs (matching GGML)
#define QTYPE_F32    0
#define QTYPE_F16    1
#define QTYPE_Q4_0   2
#define QTYPE_Q4_1   3
#define QTYPE_Q5_0   6
#define QTYPE_Q5_1   7
#define QTYPE_Q8_0   8
#define QTYPE_Q2_K  10
#define QTYPE_Q3_K  11
#define QTYPE_Q4_K  12
#define QTYPE_Q5_K  13
#define QTYPE_Q6_K  14

// Shader pipeline IDs
typedef enum {
    PIPE_MATVEC_F32 = 0,
    PIPE_MATVEC_F16,
    PIPE_MATVEC_Q4_0,
    PIPE_MATVEC_Q8_0,
    PIPE_MATVEC_Q4_K,
    PIPE_MATVEC_Q5_0,
    PIPE_MATVEC_Q6_K,
    PIPE_DEQUANT_Q4_0,
    PIPE_DEQUANT_Q8_0,
    PIPE_DEQUANT_Q4_K,
    PIPE_DEQUANT_Q5_0,
    PIPE_DEQUANT_Q6_K,
    PIPE_RMSNORM,
    PIPE_SOFTMAX,
    PIPE_ROPE,
    PIPE_SWIGLU,
    PIPE_GEGLU,
    PIPE_GELU,
    PIPE_ADD,
    PIPE_ADD_SCALED,
    PIPE_SCALE,
    PIPE_MUL,
    PIPE_COPY_F32,
    PIPE_ATTENTION,
    PIPE_RMSNORM_HEADS,
    PIPE_ADD_RMSNORM,
    PIPE_COUNT
} PipelineID;

typedef uint64_t GpuBuf;

// Initialize Vulkan compute: creates instance, picks best device, creates queue
int gpu_init(void);
void gpu_shutdown(void);

// Device info
const char* gpu_device_name(void);
uint64_t gpu_vram_bytes(void);
int gpu_is_initialized(void);

// Buffer management
GpuBuf gpu_alloc(uint64_t size_bytes, int usage);
void gpu_free(GpuBuf buf);
int gpu_upload(GpuBuf dst, const void* src, uint64_t size_bytes, uint64_t offset);
int gpu_download(void* dst, GpuBuf src, uint64_t size_bytes, uint64_t offset);

// Matrix-vector multiply: out[r] = dot(weights[r,:], x) for r in [0,rows)
// weights_buf: quantized weight matrix on GPU
// x_buf: input vector on GPU (float32)
// out_buf: output vector on GPU (float32)
int gpu_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
               int rows, int cols, int qtype);

// Batch matrix-vector: out[p*rows+r] = dot(W[r,:], x[p*cols...]) for each position p
int gpu_batch_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos, int qtype);

// Element-wise operations
int gpu_rmsnorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, int n, float eps);
int gpu_rmsnorm_heads(GpuBuf data_buf, GpuBuf weight_buf, int num_heads, int head_dim, float eps);
int gpu_softmax(GpuBuf buf, int n);
int gpu_rope(GpuBuf q_buf, GpuBuf k_buf, int num_heads, int num_kv_heads,
             int head_dim, int pos, float freq_base, int neox);
int gpu_swiglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n);
int gpu_geglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n);
int gpu_gelu(GpuBuf buf, int n);
int gpu_add(GpuBuf out_buf, GpuBuf a_buf, GpuBuf b_buf, int n);
int gpu_add_rmsnorm(GpuBuf norm_out, GpuBuf sum_out,
                    GpuBuf a_buf, GpuBuf b_buf, GpuBuf weight_buf,
                    int n, float eps);
int gpu_add_bias(GpuBuf buf, GpuBuf bias_buf, int n);
int gpu_scale(GpuBuf buf, float s, int n);
int gpu_copy_f32(GpuBuf dst, GpuBuf src, int n);

// Fused multi-head attention: dispatches one workgroup per head
// q_buf: [num_heads * head_dim], k_cache/v_cache: [max_seq_len * kv_dim]
// out_buf: [num_heads * head_dim]
int gpu_attention(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                  int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                  int seq_len, float scale);

// KV cache operations
int gpu_kv_store(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                 GpuBuf k_buf, GpuBuf v_buf,
                 int pos, int kv_dim);

// Dequantize a buffer from quantized format to float32
int gpu_dequantize(GpuBuf out_f32_buf, GpuBuf quant_buf, int n, int qtype);

// Synchronize: wait for all GPU work to complete
void gpu_sync(void);

// Batch mode: record all dispatches into one command buffer, submit once
void gpu_begin_batch(void);
void gpu_end_batch(void);
void gpu_barrier(void);

#ifdef __cplusplus
}
#endif

#endif // DLGO_VULKAN_GPU_H
