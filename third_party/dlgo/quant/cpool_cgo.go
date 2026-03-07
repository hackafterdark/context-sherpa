//go:build amd64 && cgo

package quant

/*
#cgo CFLAGS: -O3 -march=native

#include <stdint.h>

void cpool_init(int n);
void cpool_shutdown(void);
int  cpool_workers_count(void);

void cpool_qq_matvec(
    const uint8_t* w_data, uint32_t w_type,
    const float* x, int cols,
    float* out, int nrows, int bpr,
    uint8_t* q8_buf);

void cpool_fused_matvec(
    const uint8_t* w_data, uint32_t w_type,
    const float* x, int cols,
    float* out, int nrows, int bpr);

void cpool_qq_batch_gemm(
    const uint8_t* w_data, uint32_t w_type,
    const uint8_t* q8_flat, int q8_stride, int n_inputs,
    int cols, float* out_flat, int nrows, int out_stride, int bpr);

void cpool_qq_dual_batch_gemm(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const uint8_t* q8_flat, int q8_stride, int n_inputs,
    int cols, int out_stride1, int out_stride2);

void cpool_qq_triple_batch_gemm(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const uint8_t* w3, uint32_t t3, int r3, int bpr3, float* o3,
    const uint8_t* q8_flat, int q8_stride, int n_inputs,
    int cols, int out_stride1, int out_stride2, int out_stride3);

void cpool_qq_dual_matvec(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const float* x, int cols, uint8_t* q8_buf);

void cpool_qq_triple_matvec(
    const uint8_t* w1, uint32_t t1, int r1, int bpr1, float* o1,
    const uint8_t* w2, uint32_t t2, int r2, int bpr2, float* o2,
    const uint8_t* w3, uint32_t t3, int r3, int bpr3, float* o3,
    const float* x, int cols, uint8_t* q8_buf);
*/
import "C"
import (
	"os"
	"runtime"
	"strconv"
	"sync"
	"unsafe"
)

var cpoolOnce sync.Once

func cpoolEnsure() {
	cpoolOnce.Do(func() {
		n := runtime.GOMAXPROCS(0)
		if n < 1 {
			n = 1
		}
		if v := os.Getenv("DLGO_NUM_THREADS"); v != "" {
			if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
				n = parsed
			}
		}
		C.cpool_init(C.int(n))
	})
}

func CPoolInit() { cpoolEnsure() }

func CPoolShutdown() {
	C.cpool_shutdown()
}

func CPoolHas() bool    { return false }
func CPoolBatchHas() bool { return false }

func CPoolQQMatVec(wData []byte, wType uint32, x []float32, cols int,
	out []float32, nrows, bpr int, q8Buf []byte) {
	cpoolEnsure()
	C.cpool_qq_matvec(
		(*C.uint8_t)(unsafe.Pointer(&wData[0])),
		C.uint32_t(wType),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(nrows), C.int(bpr),
		(*C.uint8_t)(unsafe.Pointer(&q8Buf[0])),
	)
}

func CPoolFusedMatVec(wData []byte, wType uint32, x []float32, cols int,
	out []float32, nrows, bpr int) {
	cpoolEnsure()
	C.cpool_fused_matvec(
		(*C.uint8_t)(unsafe.Pointer(&wData[0])),
		C.uint32_t(wType),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(nrows), C.int(bpr),
	)
}

func CPoolQQDualMatVec(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	x []float32, cols int, q8Buf []byte,
) {
	cpoolEnsure()
	C.cpool_qq_dual_matvec(
		(*C.uint8_t)(unsafe.Pointer(&w1[0])), C.uint32_t(t1), C.int(r1), C.int(bpr1), (*C.float)(unsafe.Pointer(&o1[0])),
		(*C.uint8_t)(unsafe.Pointer(&w2[0])), C.uint32_t(t2), C.int(r2), C.int(bpr2), (*C.float)(unsafe.Pointer(&o2[0])),
		(*C.float)(unsafe.Pointer(&x[0])), C.int(cols),
		(*C.uint8_t)(unsafe.Pointer(&q8Buf[0])),
	)
}

func CPoolQQBatchGEMM(wData []byte, wType uint32, q8Flat []byte, q8Stride, nInputs, cols int,
	outFlat []float32, nrows, outStride, bpr int) {
	cpoolEnsure()
	C.cpool_qq_batch_gemm(
		(*C.uint8_t)(unsafe.Pointer(&wData[0])),
		C.uint32_t(wType),
		(*C.uint8_t)(unsafe.Pointer(&q8Flat[0])),
		C.int(q8Stride), C.int(nInputs),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&outFlat[0])),
		C.int(nrows), C.int(outStride), C.int(bpr),
	)
}

func CPoolQQDualBatchGEMM(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	q8Flat []byte, q8Stride, nInputs, cols, outStride1, outStride2 int,
) {
	cpoolEnsure()
	C.cpool_qq_dual_batch_gemm(
		(*C.uint8_t)(unsafe.Pointer(&w1[0])), C.uint32_t(t1), C.int(r1), C.int(bpr1), (*C.float)(unsafe.Pointer(&o1[0])),
		(*C.uint8_t)(unsafe.Pointer(&w2[0])), C.uint32_t(t2), C.int(r2), C.int(bpr2), (*C.float)(unsafe.Pointer(&o2[0])),
		(*C.uint8_t)(unsafe.Pointer(&q8Flat[0])), C.int(q8Stride), C.int(nInputs),
		C.int(cols), C.int(outStride1), C.int(outStride2),
	)
}

func CPoolQQTripleBatchGEMM(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	w3 []byte, t3 uint32, r3, bpr3 int, o3 []float32,
	q8Flat []byte, q8Stride, nInputs, cols, outStride1, outStride2, outStride3 int,
) {
	cpoolEnsure()
	C.cpool_qq_triple_batch_gemm(
		(*C.uint8_t)(unsafe.Pointer(&w1[0])), C.uint32_t(t1), C.int(r1), C.int(bpr1), (*C.float)(unsafe.Pointer(&o1[0])),
		(*C.uint8_t)(unsafe.Pointer(&w2[0])), C.uint32_t(t2), C.int(r2), C.int(bpr2), (*C.float)(unsafe.Pointer(&o2[0])),
		(*C.uint8_t)(unsafe.Pointer(&w3[0])), C.uint32_t(t3), C.int(r3), C.int(bpr3), (*C.float)(unsafe.Pointer(&o3[0])),
		(*C.uint8_t)(unsafe.Pointer(&q8Flat[0])), C.int(q8Stride), C.int(nInputs),
		C.int(cols), C.int(outStride1), C.int(outStride2), C.int(outStride3),
	)
}

func CPoolQQTripleMatVec(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	w3 []byte, t3 uint32, r3, bpr3 int, o3 []float32,
	x []float32, cols int, q8Buf []byte,
) {
	cpoolEnsure()
	C.cpool_qq_triple_matvec(
		(*C.uint8_t)(unsafe.Pointer(&w1[0])), C.uint32_t(t1), C.int(r1), C.int(bpr1), (*C.float)(unsafe.Pointer(&o1[0])),
		(*C.uint8_t)(unsafe.Pointer(&w2[0])), C.uint32_t(t2), C.int(r2), C.int(bpr2), (*C.float)(unsafe.Pointer(&o2[0])),
		(*C.uint8_t)(unsafe.Pointer(&w3[0])), C.uint32_t(t3), C.int(r3), C.int(bpr3), (*C.float)(unsafe.Pointer(&o3[0])),
		(*C.float)(unsafe.Pointer(&x[0])), C.int(cols),
		(*C.uint8_t)(unsafe.Pointer(&q8Buf[0])),
	)
}
