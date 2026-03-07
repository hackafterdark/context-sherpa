//go:build amd64 && cgo

package quant

/*
#cgo CFLAGS: -O3 -march=native

#include <stdint.h>

int q8_buffer_size(uint32_t w_type, int n);
void quantize_for_type(const float* x, uint8_t* out, uint32_t w_type, int n);
void qq_dot_batch(const uint8_t* w_data, uint32_t w_type,
                  const uint8_t* q_data, int cols,
                  float* out, int nrows, int bpr);
void qq_batch_gemm(const uint8_t* w_data, uint32_t w_type,
                   const uint8_t* q8_flat, int q8_stride, int n_inputs,
                   int cols, float* out_flat, int n_rows,
                   int out_stride, int bpr);
*/
import "C"
import "unsafe"

// Q8BufferSize returns the number of bytes needed to quantize n float32
// elements to the Q8 format appropriate for weight type wType.
func Q8BufferSize(wType uint32, n int) int {
	return int(C.q8_buffer_size(C.uint32_t(wType), C.int(n)))
}

// QuantizeForType quantizes x into the Q8 format matching wType.
// out must be at least Q8BufferSize(wType, len(x)) bytes.
func QuantizeForType(x []float32, out []byte, wType uint32) {
	C.quantize_for_type(
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		C.uint32_t(wType),
		C.int(len(x)),
	)
}

// QQDotBatch computes dot products between quantized weight rows and a
// pre-quantized Q8 input vector. This uses integer SIMD (maddubs) which
// is ~4x faster than the float dequant+dot path.
func QQDotBatch(wData []byte, wType uint32, qData []byte, cols int,
	out []float32, nrows int, bytesPerRow int) {
	C.qq_dot_batch(
		(*C.uint8_t)(unsafe.Pointer(&wData[0])),
		C.uint32_t(wType),
		(*C.uint8_t)(unsafe.Pointer(&qData[0])),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(nrows),
		C.int(bytesPerRow),
	)
}

// QQBatchGEMM computes batch GEMM with row-major loop order for optimal cache usage.
// For each weight row, computes dot products with ALL input vectors before moving to next row.
// q8Flat: contiguous buffer of nInputs pre-quantized vectors, each q8Stride bytes apart.
// outFlat: output array. outStride: distance between position outputs (typically total rows).
// Output for position p, local row r at outFlat[p*outStride + r].
func QQBatchGEMM(wData []byte, wType uint32, q8Flat []byte, q8Stride int, nInputs int, cols int,
	outFlat []float32, nRows int, outStride int, bytesPerRow int) {
	if nInputs == 0 || nRows == 0 {
		return
	}
	C.qq_batch_gemm(
		(*C.uint8_t)(unsafe.Pointer(&wData[0])),
		C.uint32_t(wType),
		(*C.uint8_t)(unsafe.Pointer(&q8Flat[0])),
		C.int(q8Stride),
		C.int(nInputs),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&outFlat[0])),
		C.int(nRows),
		C.int(outStride),
		C.int(bytesPerRow),
	)
}

// HasQQDot returns true if the Q×Q integer dot product path is available
// for the given quantization type.
func HasQQDot(ggmlType uint32) bool {
	switch ggmlType {
	case 2, 6, 8, 10, 11, 12, 13, 14:
		return true
	default:
		return false
	}
}
