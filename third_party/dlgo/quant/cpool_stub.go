//go:build !(amd64 && cgo)

package quant

func CPoolInit()        {}
func CPoolShutdown()    {}
func CPoolHas() bool      { return false }
func CPoolBatchHas() bool { return false }

func CPoolQQMatVec(wData []byte, wType uint32, x []float32, cols int,
	out []float32, nrows, bpr int, q8Buf []byte) {
}

func CPoolFusedMatVec(wData []byte, wType uint32, x []float32, cols int,
	out []float32, nrows, bpr int) {
}

func CPoolQQDualMatVec(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	x []float32, cols int, q8Buf []byte,
) {
}

func CPoolQQTripleMatVec(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	w3 []byte, t3 uint32, r3, bpr3 int, o3 []float32,
	x []float32, cols int, q8Buf []byte,
) {
}

func CPoolQQBatchGEMM(wData []byte, wType uint32, q8Flat []byte, q8Stride, nInputs, cols int,
	outFlat []float32, nrows, outStride, bpr int) {
}

func CPoolQQDualBatchGEMM(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	q8Flat []byte, q8Stride, nInputs, cols, outStride1, outStride2 int,
) {
}

func CPoolQQTripleBatchGEMM(
	w1 []byte, t1 uint32, r1, bpr1 int, o1 []float32,
	w2 []byte, t2 uint32, r2, bpr2 int, o2 []float32,
	w3 []byte, t3 uint32, r3, bpr3 int, o3 []float32,
	q8Flat []byte, q8Stride, nInputs, cols, outStride1, outStride2, outStride3 int,
) {
}
