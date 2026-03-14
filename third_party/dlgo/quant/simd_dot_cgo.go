//go:build amd64 && cgo

package quant

/*
#cgo CFLAGS: -O3 -march=native

#include <stdint.h>

float vec_dot_q4_0(const uint8_t* data, const float* x, int n);
float vec_dot_q4_1(const uint8_t* data, const float* x, int n);
float vec_dot_q5_0(const uint8_t* data, const float* x, int n);
float vec_dot_q5_1(const uint8_t* data, const float* x, int n);
float vec_dot_f16(const uint8_t* data, const float* x, int n);
float vec_dot_q8_0(const uint8_t* data, const float* x, int n);
float vec_dot_q2_k(const uint8_t* data, const float* x, int n);
float vec_dot_q3_k(const uint8_t* data, const float* x, int n);
float vec_dot_q4_k(const uint8_t* data, const float* x, int n);
float vec_dot_q5_k(const uint8_t* data, const float* x, int n);
float vec_dot_q6_k(const uint8_t* data, const float* x, int n);

void vec_dot_q4_0_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q4_1_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q5_0_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q5_1_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_f16_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q8_0_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q2_k_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q3_k_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q4_k_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q5_k_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);
void vec_dot_q6_k_batch(const uint8_t* data, const float* x, int n, float* out, int nrows, int bpr);

float vec_dot_f32(const float* a, const float* b, int n);
void vec_dot_f32_batch(const float* a_flat, const float* x, int cols,
                       float* out, int nrows);

void vec_scale_add(float* out, float scale, const float* src, int n);
void vec_swiglu(float* out, const float* gate, const float* up, int n);
void vec_softmax(float* x, int n);
*/
import "C"
import "unsafe"

// SIMDDotProduct computes the dot product of a quantized vector and a float32
// vector using AVX2+FMA SIMD intrinsics via CGo. Falls back to Go scalar
// implementation for unsupported types.
//
// This is 5-15x faster than the pure Go scalar fused dot product for the
// common quantization types (Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).
func SIMDDotProduct(data []byte, ggmlType uint32, x []float32, n int) float32 {
	dp := unsafe.Pointer(&data[0])
	xp := unsafe.Pointer(&x[0])

	switch ggmlType {
	case 1: // F16
		return float32(C.vec_dot_f16((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 2: // Q4_0
		return float32(C.vec_dot_q4_0((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 3: // Q4_1
		return float32(C.vec_dot_q4_1((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 6: // Q5_0
		return float32(C.vec_dot_q5_0((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 7: // Q5_1
		return float32(C.vec_dot_q5_1((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 8: // Q8_0
		return float32(C.vec_dot_q8_0((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 10: // Q2_K
		return float32(C.vec_dot_q2_k((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 11: // Q3_K
		return float32(C.vec_dot_q3_k((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 12: // Q4_K
		return float32(C.vec_dot_q4_k((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 13: // Q5_K
		return float32(C.vec_dot_q5_k((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	case 14: // Q6_K
		return float32(C.vec_dot_q6_k((*C.uint8_t)(dp), (*C.float)(xp), C.int(n)))
	default:
		return FusedDotProduct(data, ggmlType, x, n)
	}
}

// SIMDDotBatch computes dot products for multiple rows in a single CGo call,
// amortizing the Go→C→Go transition overhead across all rows.
// data: quantized data starting at the first row
// out[0..nrows-1] receives the dot product of each row with x.
func SIMDDotBatch(data []byte, ggmlType uint32, x []float32, cols int, out []float32, nrows int, bytesPerRow int) {
	if nrows <= 0 {
		return
	}
	dp := unsafe.Pointer(&data[0])
	xp := unsafe.Pointer(&x[0])
	op := unsafe.Pointer(&out[0])

	switch ggmlType {
	case 1: // F16
		C.vec_dot_f16_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 2: // Q4_0
		C.vec_dot_q4_0_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 3: // Q4_1
		C.vec_dot_q4_1_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 6: // Q5_0
		C.vec_dot_q5_0_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 7: // Q5_1
		C.vec_dot_q5_1_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 8: // Q8_0
		C.vec_dot_q8_0_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 10: // Q2_K
		C.vec_dot_q2_k_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 11: // Q3_K
		C.vec_dot_q3_k_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 12: // Q4_K
		C.vec_dot_q4_k_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 13: // Q5_K
		C.vec_dot_q5_k_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	case 14: // Q6_K
		C.vec_dot_q6_k_batch((*C.uint8_t)(dp), (*C.float)(xp), C.int(cols), (*C.float)(op), C.int(nrows), C.int(bytesPerRow))
	default:
		for r := 0; r < nrows; r++ {
			rowData := data[r*bytesPerRow : (r+1)*bytesPerRow]
			out[r] = FusedDotProduct(rowData, ggmlType, x, cols)
		}
	}
}

// SIMDDotF32 computes the dot product of two float32 slices using AVX2+FMA.
// Numerically matches gollm's DotProductAVX2 assembly implementation.
func SIMDDotF32(a, b []float32, n int) float32 {
	return float32(C.vec_dot_f32((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(n)))
}

// SIMDDotF32Batch computes dot products for nrows rows of a float32 matrix against x.
func SIMDDotF32Batch(aFlat []float32, x []float32, cols int, out []float32, nrows int) {
	C.vec_dot_f32_batch((*C.float)(unsafe.Pointer(&aFlat[0])), (*C.float)(unsafe.Pointer(&x[0])),
		C.int(cols), (*C.float)(unsafe.Pointer(&out[0])), C.int(nrows))
}

// SIMDScaleAdd computes out[i] += scale * src[i] using AVX2+FMA.
// Used for attention weighted value accumulation.
func SIMDScaleAdd(out []float32, scale float32, src []float32, n int) {
	C.vec_scale_add((*C.float)(unsafe.Pointer(&out[0])), C.float(scale), (*C.float)(unsafe.Pointer(&src[0])), C.int(n))
}

// SIMDSwiGLU computes out[i] = SiLU(gate[i]) * up[i] using AVX2+FMA.
// Uses fast polynomial exp approximation + Newton-Raphson reciprocal.
func SIMDSwiGLU(out, gate, up []float32, n int) {
	C.vec_swiglu((*C.float)(unsafe.Pointer(&out[0])), (*C.float)(unsafe.Pointer(&gate[0])), (*C.float)(unsafe.Pointer(&up[0])), C.int(n))
}

// SIMDSoftmax performs in-place softmax using AVX2 fast exp approximation.
func SIMDSoftmax(x []float32) {
	C.vec_softmax((*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
}

// HasSIMDDot returns true if the AVX2+FMA fused dot product supports the given
// type. Types not listed here use the dequantize-then-dot path which can produce
// slightly different floating-point rounding due to accumulation order.
func HasSIMDDot(ggmlType uint32) bool {
	switch ggmlType {
	case 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14:
		return true
	default:
		return false
	}
}
