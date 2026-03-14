//go:build cgo && vulkan

package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/csrc
#cgo windows CFLAGS: -IC:/VulkanSDK/1.4.341.1/Include
#cgo windows LDFLAGS: -LC:/VulkanSDK/1.4.341.1/Lib -lvulkan-1

#include "vulkan_gpu.c"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Buf is a handle to a GPU buffer.
type Buf = uint64

// Init initializes the Vulkan compute backend.
func Init() error {
	rc := C.gpu_init()
	if rc != C.GPU_OK {
		switch rc {
		case C.GPU_ERR_NO_VULKAN:
			return fmt.Errorf("gpu: vulkan runtime not found")
		case C.GPU_ERR_NO_DEVICE:
			return fmt.Errorf("gpu: no vulkan-capable GPU found")
		case C.GPU_ERR_INIT_FAIL:
			return fmt.Errorf("gpu: vulkan initialization failed")
		default:
			return fmt.Errorf("gpu: init error %d", rc)
		}
	}
	return nil
}

// Shutdown releases all GPU resources.
func Shutdown() { C.gpu_shutdown() }

// IsInitialized returns true if the GPU backend is ready.
func IsInitialized() bool { return C.gpu_is_initialized() != 0 }

// DeviceName returns the GPU device name.
func DeviceName() string { return C.GoString(C.gpu_device_name()) }

// VRAMBytes returns total device-local VRAM in bytes.
func VRAMBytes() uint64 { return uint64(C.gpu_vram_bytes()) }

// Alloc allocates a GPU buffer of the given size.
func Alloc(sizeBytes uint64) Buf {
	return uint64(C.gpu_alloc(C.uint64_t(sizeBytes), C.GPU_BUF_STORAGE))
}

// Free releases a GPU buffer.
func Free(buf Buf) { C.gpu_free(C.GpuBuf(buf)) }

// Upload copies data from CPU to GPU.
func Upload(dst Buf, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	rc := C.gpu_upload(C.GpuBuf(dst), unsafe.Pointer(&src[0]), C.uint64_t(len(src)), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: upload failed (%d)", rc)
	}
	return nil
}

// UploadF32 copies float32 data from CPU to GPU.
func UploadF32(dst Buf, src []float32) error {
	if len(src) == 0 {
		return nil
	}
	size := len(src) * 4
	rc := C.gpu_upload(C.GpuBuf(dst), unsafe.Pointer(&src[0]), C.uint64_t(size), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: upload failed (%d)", rc)
	}
	return nil
}

// Download copies data from GPU to CPU.
func Download(src Buf, dst []byte) error {
	if len(dst) == 0 {
		return nil
	}
	rc := C.gpu_download(unsafe.Pointer(&dst[0]), C.GpuBuf(src), C.uint64_t(len(dst)), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: download failed (%d)", rc)
	}
	return nil
}

// DownloadF32 copies float32 data from GPU to CPU.
func DownloadF32(src Buf, dst []float32) error {
	if len(dst) == 0 {
		return nil
	}
	size := len(dst) * 4
	rc := C.gpu_download(unsafe.Pointer(&dst[0]), C.GpuBuf(src), C.uint64_t(size), 0)
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: download failed (%d)", rc)
	}
	return nil
}

// MatVec performs quantized matrix-vector multiply on GPU.
func MatVec(out, weights, x Buf, rows, cols int, qtype uint32) error {
	rc := C.gpu_matvec(C.GpuBuf(out), C.GpuBuf(weights), C.GpuBuf(x),
		C.int(rows), C.int(cols), C.int(qtype))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: matvec failed (%d)", rc)
	}
	return nil
}

// RMSNorm performs RMS normalization on GPU.
func RMSNorm(out, x, weight Buf, n int, eps float32) error {
	rc := C.gpu_rmsnorm(C.GpuBuf(out), C.GpuBuf(x), C.GpuBuf(weight), C.int(n), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rmsnorm failed (%d)", rc)
	}
	return nil
}

// RMSNormHeads performs per-head in-place RMS normalization on GPU.
func RMSNormHeads(data, weight Buf, numHeads, headDim int, eps float32) error {
	rc := C.gpu_rmsnorm_heads(C.GpuBuf(data), C.GpuBuf(weight), C.int(numHeads), C.int(headDim), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rmsnorm_heads failed (%d)", rc)
	}
	return nil
}

// Softmax performs in-place softmax on GPU.
func Softmax(buf Buf, n int) error {
	rc := C.gpu_softmax(C.GpuBuf(buf), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: softmax failed (%d)", rc)
	}
	return nil
}

// RoPE applies rotary position embedding on GPU.
func RoPE(q, k Buf, numHeads, numKVHeads, headDim, pos int, freqBase float32, neox bool) error {
	n := 0
	if neox {
		n = 1
	}
	rc := C.gpu_rope(C.GpuBuf(q), C.GpuBuf(k),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(pos), C.float(freqBase), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: rope failed (%d)", rc)
	}
	return nil
}

// SwiGLU performs SwiGLU activation on GPU.
func SwiGLU(out, gate, up Buf, n int) error {
	rc := C.gpu_swiglu(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: swiglu failed (%d)", rc)
	}
	return nil
}

// GeGLU performs GeGLU activation on GPU.
func GeGLU(out, gate, up Buf, n int) error {
	rc := C.gpu_geglu(C.GpuBuf(out), C.GpuBuf(gate), C.GpuBuf(up), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: geglu failed (%d)", rc)
	}
	return nil
}

// GELU performs in-place GELU activation on GPU.
func GELU(buf Buf, n int) error {
	rc := C.gpu_gelu(C.GpuBuf(buf), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: gelu failed (%d)", rc)
	}
	return nil
}

// Add performs element-wise addition on GPU.
func Add(out, a, b Buf, n int) error {
	rc := C.gpu_add(C.GpuBuf(out), C.GpuBuf(a), C.GpuBuf(b), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: add failed (%d)", rc)
	}
	return nil
}

// Scale performs in-place scaling on GPU.
func Scale(buf Buf, s float32, n int) error {
	rc := C.gpu_scale(C.GpuBuf(buf), C.float(s), C.int(n))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: scale failed (%d)", rc)
	}
	return nil
}

// Attention performs fused multi-head attention entirely on GPU.
func Attention(out, q, kCache, vCache Buf, numHeads, numKVHeads, headDim, kvDim, seqLen int, scale float32) error {
	rc := C.gpu_attention(C.GpuBuf(out), C.GpuBuf(q), C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim), C.int(kvDim), C.int(seqLen), C.float(scale))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: attention failed (%d)", rc)
	}
	return nil
}

// KVStore copies K and V vectors into cache buffers at the given position.
func KVStore(kCache, vCache, k, v Buf, pos, kvDim int) error {
	rc := C.gpu_kv_store(C.GpuBuf(kCache), C.GpuBuf(vCache),
		C.GpuBuf(k), C.GpuBuf(v), C.int(pos), C.int(kvDim))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: kv_store failed (%d)", rc)
	}
	return nil
}

// Sync waits for all GPU operations to complete.
func Sync() { C.gpu_sync() }

// BeginBatch starts recording GPU operations into a single command buffer.
// All subsequent GPU calls are batched until EndBatch.
func BeginBatch() { C.gpu_begin_batch() }

// EndBatch submits all batched operations at once and waits for completion.
func EndBatch() { C.gpu_end_batch() }

// Barrier inserts a compute memory barrier so subsequent dispatches see prior writes.
func Barrier() { C.gpu_barrier() }

// AddRMSNorm performs fused Add + RMSNorm: sumOut = a+b, normOut = RMSNorm(sumOut, weight).
func AddRMSNorm(normOut, sumOut, a, b, weight Buf, n int, eps float32) error {
	rc := C.gpu_add_rmsnorm(C.GpuBuf(normOut), C.GpuBuf(sumOut),
		C.GpuBuf(a), C.GpuBuf(b), C.GpuBuf(weight), C.int(n), C.float(eps))
	if rc != C.GPU_OK {
		return fmt.Errorf("gpu: add_rmsnorm failed (%d)", rc)
	}
	return nil
}
