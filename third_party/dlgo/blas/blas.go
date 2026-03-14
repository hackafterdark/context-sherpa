// Package blas provides quantized linear algebra operations for neural network inference.
//
// It bridges the core.QuantizedTensor type with the quant package's dequantization and
// fused dot product routines, providing high-level matrix-vector and batch GEMM operations.
package blas

import (
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/ops"
	"github.com/computerex/dlgo/quant"
)

var q8Pool sync.Pool

func getQ8Buf(size int) []byte {
	if v := q8Pool.Get(); v != nil {
		buf := v.([]byte)
		if cap(buf) >= size {
			return buf[:size]
		}
	}
	return make([]byte, size)
}

func putQ8Buf(buf []byte) {
	q8Pool.Put(buf)
}

// Pool manages persistent goroutines for parallel matrix operations.
type Pool struct {
	numWorkers int
	taskChs    []chan func()
	wg         sync.WaitGroup
	alive      atomic.Bool
}

var defaultPool *Pool
var defaultPoolOnce sync.Once

// DefaultPool returns a shared worker pool sized to the number of CPUs.
func DefaultPool() *Pool {
	defaultPoolOnce.Do(func() {
		n := runtime.GOMAXPROCS(0)
		if n < 1 {
			n = 1
		}
		if v := os.Getenv("DLGO_NUM_THREADS"); v != "" {
			if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
				n = parsed
			}
		}
		defaultPool = NewPool(n)
	})
	return defaultPool
}

// NewPool creates a worker pool with n persistent goroutines.
func NewPool(n int) *Pool {
	if n < 1 {
		n = 1
	}
	p := &Pool{
		numWorkers: n,
		taskChs:    make([]chan func(), n),
	}
	p.alive.Store(true)
	for i := 0; i < n; i++ {
		ch := make(chan func(), 1)
		p.taskChs[i] = ch
		p.wg.Add(1)
		go func(c chan func()) {
			defer p.wg.Done()
			for fn := range c {
				fn()
			}
		}(ch)
	}
	return p
}

func (p *Pool) dispatch(total, numActive int, work func(start, end int)) {
	if total <= 0 || numActive <= 0 || !p.alive.Load() {
		return
	}
	if numActive > p.numWorkers {
		numActive = p.numWorkers
	}
	if numActive > total {
		numActive = total
	}
	if numActive <= 1 {
		work(0, total)
		return
	}
	chunk := (total + numActive - 1) / numActive
	var done sync.WaitGroup

	for w := 0; w < numActive; w++ {
		s := w * chunk
		e := s + chunk
		if s >= total {
			break
		}
		if e > total {
			e = total
		}
		done.Add(1)
		start, end := s, e
		p.taskChs[w] <- func() {
			work(start, end)
			done.Done()
		}
	}
	done.Wait()
}

// ParallelFor executes fn(i) for i in [0, n) across pool workers.
func (p *Pool) ParallelFor(n int, fn func(i int)) {
	if n <= 0 || !p.alive.Load() {
		return
	}
	if n == 1 {
		fn(0)
		return
	}
	p.dispatch(n, p.numWorkers, func(start, end int) {
		for i := start; i < end; i++ {
			fn(i)
		}
	})
}

// Shutdown stops all workers.
func (p *Pool) Shutdown() {
	if p.alive.CompareAndSwap(true, false) {
		for _, ch := range p.taskChs {
			close(ch)
		}
		p.wg.Wait()
	}
}

// QMatVecMul performs quantized matrix-vector multiply (single-threaded).
//
//	out[r] = dot(qt[r, :], x)   for r in [0, qt.Rows)
func QMatVecMul(out []float32, qt *core.QuantizedTensor, x []float32) {
	if qt.FP32Data != nil {
		ops.MatVecMul(out[:qt.Rows], qt.FP32Data, x, qt.Rows, qt.Cols)
		return
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)

	if quant.HasQQDot(qt.Type) {
		q8Size := quant.Q8BufferSize(qt.Type, qt.Cols)
		q8Buf := getQ8Buf(q8Size)
		quant.QuantizeForType(x, q8Buf, qt.Type)
		quant.QQDotBatch(qt.Data, qt.Type, q8Buf, qt.Cols, out[:qt.Rows], qt.Rows, bytesPerRow)
		putQ8Buf(q8Buf)
		return
	}

	useFused := quant.HasSIMDDot(qt.Type)
	if useFused {
		quant.SIMDDotBatch(qt.Data, qt.Type, x, qt.Cols, out[:qt.Rows], qt.Rows, bytesPerRow)
	} else {
		buf := make([]float32, qt.Cols)
		for r := 0; r < qt.Rows; r++ {
			rowData := qt.Data[r*bytesPerRow : (r+1)*bytesPerRow]
			quant.DequantizeInto(buf, rowData, qt.Type, qt.Cols)
			out[r] = quant.SIMDDotF32(buf, x, qt.Cols)
		}
	}
}

// QMatVecMulParallel performs parallel quantized matrix-vector multiply.
// Uses the C thread pool for Q×Q and fused paths (single CGo call, no goroutines),
// falling back to the Go pool for unsupported types.
func QMatVecMulParallel(out []float32, qt *core.QuantizedTensor, x []float32, pool *Pool) {
	if pool == nil || qt.Rows < 256 {
		QMatVecMul(out, qt, x)
		return
	}

	if qt.FP32Data != nil {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			quant.SIMDDotF32Batch(qt.FP32Data[start*qt.Cols:end*qt.Cols], x, qt.Cols, out[start:end], end-start)
		})
		return
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)

	// C thread pool path: single CGo call, C handles parallelization
	if quant.CPoolHas() {
		if quant.HasQQDot(qt.Type) {
			q8Size := quant.Q8BufferSize(qt.Type, qt.Cols)
			q8Buf := getQ8Buf(q8Size)
			quant.CPoolQQMatVec(qt.Data, qt.Type, x, qt.Cols, out[:qt.Rows], qt.Rows, bytesPerRow, q8Buf)
			putQ8Buf(q8Buf)
			return
		}
		if quant.HasSIMDDot(qt.Type) {
			quant.CPoolFusedMatVec(qt.Data, qt.Type, x, qt.Cols, out[:qt.Rows], qt.Rows, bytesPerRow)
			return
		}
	}

	// Go pool: Q×Q integer SIMD path (quantize once, parallel QQ dot)
	if quant.HasQQDot(qt.Type) {
		q8Size := quant.Q8BufferSize(qt.Type, qt.Cols)
		q8Buf := getQ8Buf(q8Size)
		quant.QuantizeForType(x, q8Buf, qt.Type)
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			nrows := end - start
			startByte := start * bytesPerRow
			endByte := end * bytesPerRow
			quant.QQDotBatch(qt.Data[startByte:endByte], qt.Type, q8Buf, qt.Cols, out[start:end], nrows, bytesPerRow)
		})
		putQ8Buf(q8Buf)
		return
	}

	// Go pool fallback for other types
	useFused := quant.HasSIMDDot(qt.Type)
	if useFused {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			nrows := end - start
			startByte := start * bytesPerRow
			endByte := end * bytesPerRow
			quant.SIMDDotBatch(qt.Data[startByte:endByte], qt.Type, x, qt.Cols, out[start:end], nrows, bytesPerRow)
		})
	} else {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			buf := make([]float32, qt.Cols)
			for r := start; r < end; r++ {
				rowData := qt.Data[r*bytesPerRow : (r+1)*bytesPerRow]
				quant.DequantizeInto(buf, rowData, qt.Type, qt.Cols)
				out[r] = quant.SIMDDotF32(buf, x, qt.Cols)
			}
		})
	}
}

// QDualMatVecMulParallel runs two matrix-vector multiplies sharing a single
// Q8 quantization and a single dispatch wave.
func QDualMatVecMulParallel(out1 []float32, qt1 *core.QuantizedTensor,
	out2 []float32, qt2 *core.QuantizedTensor, x []float32, pool *Pool) {
	if pool == nil {
		QMatVecMul(out1, qt1, x)
		QMatVecMul(out2, qt2, x)
		return
	}

	useQQ1 := qt1.FP32Data == nil && quant.HasQQDot(qt1.Type)
	useQQ2 := qt2.FP32Data == nil && quant.HasQQDot(qt2.Type)

	if useQQ1 && useQQ2 && qt1.Type == qt2.Type && qt1.Cols == qt2.Cols {
		q8Size := quant.Q8BufferSize(qt1.Type, qt1.Cols)
		q8Buf := getQ8Buf(q8Size)
		quant.QuantizeForType(x, q8Buf, qt1.Type)

		bpr1 := quant.BytesForN(qt1.Type, qt1.Cols)
		bpr2 := quant.BytesForN(qt2.Type, qt2.Cols)
		rows1, rows2 := qt1.Rows, qt2.Rows
		totalRows := rows1 + rows2

		pool.dispatch(totalRows, pool.numWorkers, func(start, end int) {
			s1, e1 := start, end
			if s1 < rows1 {
				if e1 > rows1 {
					e1 = rows1
				}
				n := e1 - s1
				quant.QQDotBatch(qt1.Data[s1*bpr1:e1*bpr1], qt1.Type, q8Buf, qt1.Cols, out1[s1:e1], n, bpr1)
			}
			s2, e2 := start-rows1, end-rows1
			if s2 < 0 {
				s2 = 0
			}
			if e2 > rows2 {
				e2 = rows2
			}
			if s2 < e2 {
				n := e2 - s2
				quant.QQDotBatch(qt2.Data[s2*bpr2:e2*bpr2], qt2.Type, q8Buf, qt2.Cols, out2[s2:e2], n, bpr2)
			}
		})
		putQ8Buf(q8Buf)
		return
	}

	QMatVecMulParallel(out1, qt1, x, pool)
	QMatVecMulParallel(out2, qt2, x, pool)
}

// QTripleMatVecMulParallel runs three matrix-vector multiplies sharing a single
// Q8 quantization and a single dispatch wave.
func QTripleMatVecMulParallel(
	out1 []float32, qt1 *core.QuantizedTensor,
	out2 []float32, qt2 *core.QuantizedTensor,
	out3 []float32, qt3 *core.QuantizedTensor,
	x []float32, pool *Pool,
) {
	if pool == nil {
		QMatVecMul(out1, qt1, x)
		QMatVecMul(out2, qt2, x)
		QMatVecMul(out3, qt3, x)
		return
	}

	useQQ := qt1.FP32Data == nil && qt2.FP32Data == nil && qt3.FP32Data == nil &&
		quant.HasQQDot(qt1.Type) && qt1.Type == qt2.Type && qt1.Type == qt3.Type &&
		qt1.Cols == qt2.Cols && qt1.Cols == qt3.Cols

	if useQQ {
		q8Size := quant.Q8BufferSize(qt1.Type, qt1.Cols)
		q8Buf := getQ8Buf(q8Size)
		quant.QuantizeForType(x, q8Buf, qt1.Type)

		bpr1 := quant.BytesForN(qt1.Type, qt1.Cols)
		bpr2 := quant.BytesForN(qt2.Type, qt2.Cols)
		bpr3 := quant.BytesForN(qt3.Type, qt3.Cols)
		r1, r2, r3 := qt1.Rows, qt2.Rows, qt3.Rows
		totalRows := r1 + r2 + r3

		pool.dispatch(totalRows, pool.numWorkers, func(start, end int) {
			s, e := start, end
			if s < r1 {
				e1 := e
				if e1 > r1 {
					e1 = r1
				}
				n := e1 - s
				quant.QQDotBatch(qt1.Data[s*bpr1:e1*bpr1], qt1.Type, q8Buf, qt1.Cols, out1[s:e1], n, bpr1)
			}
			s2, e2 := start-r1, end-r1
			if s2 < 0 {
				s2 = 0
			}
			if e2 > r2 {
				e2 = r2
			}
			if s2 < e2 {
				n := e2 - s2
				quant.QQDotBatch(qt2.Data[s2*bpr2:e2*bpr2], qt2.Type, q8Buf, qt2.Cols, out2[s2:e2], n, bpr2)
			}
			s3, e3 := start-r1-r2, end-r1-r2
			if s3 < 0 {
				s3 = 0
			}
			if e3 > r3 {
				e3 = r3
			}
			if s3 < e3 {
				n := e3 - s3
				quant.QQDotBatch(qt3.Data[s3*bpr3:e3*bpr3], qt3.Type, q8Buf, qt3.Cols, out3[s3:e3], n, bpr3)
			}
		})
		putQ8Buf(q8Buf)
		return
	}

	QMatVecMulParallel(out1, qt1, x, pool)
	QMatVecMulParallel(out2, qt2, x, pool)
	QMatVecMulParallel(out3, qt3, x, pool)
}

// QBatchGEMM performs batch quantized matrix multiply.
//
//	outFlat[p*qt.Rows + r] = dot(qt[r, :], xFlat[p*qt.Cols : (p+1)*qt.Cols])
//
// For each of the nPos input positions, computes the full matrix-vector product.
// outFlat: [nPos * qt.Rows], xFlat: [nPos * qt.Cols].
func QBatchGEMM(outFlat []float32, qt *core.QuantizedTensor, xFlat []float32, nPos int) {
	if nPos <= 0 {
		return
	}

	if qt.FP32Data != nil && nPos >= 4 {
		for p := 0; p < nPos; p++ {
			x := xFlat[p*qt.Cols : (p+1)*qt.Cols]
			out := outFlat[p*qt.Rows : (p+1)*qt.Rows]
			ops.MatVecMul(out, qt.FP32Data, x, qt.Rows, qt.Cols)
		}
		return
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)
	useFused := quant.HasSIMDDot(qt.Type)

	if useFused && nPos == 1 {
		quant.SIMDDotBatch(qt.Data, qt.Type, xFlat[:qt.Cols], qt.Cols, outFlat[:qt.Rows], qt.Rows, bytesPerRow)
		return
	}

	for p := 0; p < nPos; p++ {
		x := xFlat[p*qt.Cols : (p+1)*qt.Cols]
		out := outFlat[p*qt.Rows : (p+1)*qt.Rows]
		if useFused {
			quant.SIMDDotBatch(qt.Data, qt.Type, x, qt.Cols, out, qt.Rows, bytesPerRow)
		} else {
			buf := make([]float32, qt.Cols)
			for r := 0; r < qt.Rows; r++ {
				rowData := qt.Data[r*bytesPerRow : (r+1)*bytesPerRow]
				quant.DequantizeInto(buf, rowData, qt.Type, qt.Cols)
				out[r] = quant.SIMDDotF32(buf, x, qt.Cols)
			}
		}
	}
}

// QBatchGEMMParallel performs parallel batch quantized matrix multiply.
// Parallelizes across weight rows: each worker processes all nPos positions for its row range.
// This maximizes weight data cache reuse (each row loaded once, used for all positions).
// outFlat layout: [nPos][qt.Rows] (position-major). xFlat: [nPos][qt.Cols].
func QBatchGEMMParallel(outFlat []float32, qt *core.QuantizedTensor, xFlat []float32, nPos int, pool *Pool) {
	if nPos <= 0 {
		return
	}
	if pool == nil || nPos == 1 {
		if pool != nil {
			QMatVecMulParallel(outFlat, qt, xFlat, pool)
		} else {
			QBatchGEMM(outFlat, qt, xFlat, nPos)
		}
		return
	}

	if qt.FP32Data != nil {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			for p := 0; p < nPos; p++ {
				x := xFlat[p*qt.Cols : (p+1)*qt.Cols]
				out := outFlat[p*qt.Rows : (p+1)*qt.Rows]
				quant.SIMDDotF32Batch(qt.FP32Data[start*qt.Cols:end*qt.Cols], x, qt.Cols, out[start:end], end-start)
			}
		})
		return
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)

	if quant.HasQQDot(qt.Type) {
		q8Size := quant.Q8BufferSize(qt.Type, qt.Cols)

		// Contiguous Q8 buffer for all positions
		q8Flat := getQ8Buf(q8Size * nPos)
		for p := 0; p < nPos; p++ {
			quant.QuantizeForType(xFlat[p*qt.Cols:(p+1)*qt.Cols], q8Flat[p*q8Size:(p+1)*q8Size], qt.Type)
		}

		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			nrows := end - start
			sb := start * bytesPerRow
			quant.QQBatchGEMM(qt.Data[sb:sb+nrows*bytesPerRow], qt.Type, q8Flat, q8Size, nPos, qt.Cols, outFlat[start:], nrows, qt.Rows, bytesPerRow)
		})

		putQ8Buf(q8Flat)
		return
	}

	if quant.HasSIMDDot(qt.Type) {
		pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
			nrows := end - start
			sb := start * bytesPerRow
			eb := end * bytesPerRow
			for p := 0; p < nPos; p++ {
				x := xFlat[p*qt.Cols : (p+1)*qt.Cols]
				out := outFlat[p*qt.Rows : (p+1)*qt.Rows]
				quant.SIMDDotBatch(qt.Data[sb:eb], qt.Type, x, qt.Cols, out[start:end], nrows, bytesPerRow)
			}
		})
		return
	}

	pool.dispatch(qt.Rows, pool.numWorkers, func(start, end int) {
		buf := make([]float32, qt.Cols)
		for p := 0; p < nPos; p++ {
			x := xFlat[p*qt.Cols : (p+1)*qt.Cols]
			out := outFlat[p*qt.Rows : (p+1)*qt.Rows]
			for r := start; r < end; r++ {
				rowData := qt.Data[r*bytesPerRow : (r+1)*bytesPerRow]
				quant.DequantizeInto(buf, rowData, qt.Type, qt.Cols)
				out[r] = quant.SIMDDotF32(buf, x, qt.Cols)
			}
		}
	})
}

// QTripleBatchGEMMParallel computes three batch GEMMs sharing the same input.
// Quantizes the input once and dispatches all three projections in a combined pass.
// Ideal for Q/K/V projections in attention where the input (xNorm) is the same.
func QTripleBatchGEMMParallel(
	out1 []float32, qt1 *core.QuantizedTensor,
	out2 []float32, qt2 *core.QuantizedTensor,
	out3 []float32, qt3 *core.QuantizedTensor,
	xFlat []float32, nPos int, pool *Pool,
) {
	if nPos <= 0 || pool == nil {
		return
	}
	if nPos == 1 {
		QTripleMatVecMulParallel(out1, qt1, out2, qt2, out3, qt3, xFlat, pool)
		return
	}

	cols := qt1.Cols
	bpr1 := quant.BytesForN(qt1.Type, cols)
	bpr2 := quant.BytesForN(qt2.Type, cols)
	bpr3 := quant.BytesForN(qt3.Type, cols)

	if quant.HasQQDot(qt1.Type) {
		q8Size := quant.Q8BufferSize(qt1.Type, cols)

		// Contiguous Q8 buffer for all positions
		q8Flat := getQ8Buf(q8Size * nPos)
		for p := 0; p < nPos; p++ {
			quant.QuantizeForType(xFlat[p*cols:(p+1)*cols], q8Flat[p*q8Size:(p+1)*q8Size], qt1.Type)
		}

		totalRows := qt1.Rows + qt2.Rows + qt3.Rows
		pool.dispatch(totalRows, pool.numWorkers, func(start, end int) {
			r1 := qt1.Rows
			r12 := r1 + qt2.Rows
			s, e := start, end
			if s < r1 {
				e1 := e
				if e1 > r1 {
					e1 = r1
				}
				n := e1 - s
				sb := s * bpr1
				quant.QQBatchGEMM(qt1.Data[sb:sb+n*bpr1], qt1.Type, q8Flat, q8Size, nPos, cols, out1[s:], n, qt1.Rows, bpr1)
			}
			if e > r1 && s < r12 {
				s2 := s - r1
				if s2 < 0 {
					s2 = 0
				}
				e2 := e - r1
				if e2 > qt2.Rows {
					e2 = qt2.Rows
				}
				n := e2 - s2
				sb := s2 * bpr2
				quant.QQBatchGEMM(qt2.Data[sb:sb+n*bpr2], qt2.Type, q8Flat, q8Size, nPos, cols, out2[s2:], n, qt2.Rows, bpr2)
			}
			if e > r12 {
				s3 := s - r12
				if s3 < 0 {
					s3 = 0
				}
				e3 := e - r12
				if e3 > qt3.Rows {
					e3 = qt3.Rows
				}
				n := e3 - s3
				sb := s3 * bpr3
				quant.QQBatchGEMM(qt3.Data[sb:sb+n*bpr3], qt3.Type, q8Flat, q8Size, nPos, cols, out3[s3:], n, qt3.Rows, bpr3)
			}
		})

		putQ8Buf(q8Flat)
		return
	}

	QBatchGEMMParallel(out1, qt1, xFlat, nPos, pool)
	QBatchGEMMParallel(out2, qt2, xFlat, nPos, pool)
	QBatchGEMMParallel(out3, qt3, xFlat, nPos, pool)
}

// QDualBatchGEMMParallel computes two batch GEMMs sharing the same input.
// Quantizes the input once and dispatches both projections in a combined pass.
func QDualBatchGEMMParallel(
	out1 []float32, qt1 *core.QuantizedTensor,
	out2 []float32, qt2 *core.QuantizedTensor,
	xFlat []float32, nPos int, pool *Pool,
) {
	if nPos <= 0 || pool == nil {
		return
	}
	if nPos == 1 {
		QDualMatVecMulParallel(out1, qt1, out2, qt2, xFlat, pool)
		return
	}

	cols := qt1.Cols
	bpr1 := quant.BytesForN(qt1.Type, cols)
	bpr2 := quant.BytesForN(qt2.Type, cols)

	if quant.HasQQDot(qt1.Type) {
		q8Size := quant.Q8BufferSize(qt1.Type, cols)

		// Contiguous Q8 buffer for all positions
		q8Flat := getQ8Buf(q8Size * nPos)
		for p := 0; p < nPos; p++ {
			quant.QuantizeForType(xFlat[p*cols:(p+1)*cols], q8Flat[p*q8Size:(p+1)*q8Size], qt1.Type)
		}

		totalRows := qt1.Rows + qt2.Rows
		pool.dispatch(totalRows, pool.numWorkers, func(start, end int) {
			r1 := qt1.Rows
			s, e := start, end
			if s < r1 {
				e1 := e
				if e1 > r1 {
					e1 = r1
				}
				n := e1 - s
				sb := s * bpr1
				quant.QQBatchGEMM(qt1.Data[sb:sb+n*bpr1], qt1.Type, q8Flat, q8Size, nPos, cols, out1[s:], n, qt1.Rows, bpr1)
			}
			if e > r1 {
				s2 := s - r1
				if s2 < 0 {
					s2 = 0
				}
				e2 := e - r1
				if e2 > qt2.Rows {
					e2 = qt2.Rows
				}
				n := e2 - s2
				sb := s2 * bpr2
				quant.QQBatchGEMM(qt2.Data[sb:sb+n*bpr2], qt2.Type, q8Flat, q8Size, nPos, cols, out2[s2:], n, qt2.Rows, bpr2)
			}
		})

		putQ8Buf(q8Flat)
		return
	}

	QBatchGEMMParallel(out1, qt1, xFlat, nPos, pool)
	QBatchGEMMParallel(out2, qt2, xFlat, nPos, pool)
}

// DequantizeAll returns the entire quantized tensor dequantized to float32.
func DequantizeAll(qt *core.QuantizedTensor) []float32 {
	if qt.FP32Data != nil {
		out := make([]float32, len(qt.FP32Data))
		copy(out, qt.FP32Data)
		return out
	}
	n := qt.Rows * qt.Cols
	result, _ := quant.Dequantize(qt.Data, qt.Type, n)
	return result
}

// PreDequantize converts a quantized tensor to FP32 and caches the result.
// Subsequent QMatVecMul / QBatchGEMM calls will use the cached FP32 data.
// The raw quantized bytes are released to save memory.
func PreDequantize(qt *core.QuantizedTensor) {
	if qt == nil || qt.FP32Data != nil {
		return
	}
	qt.FP32Data = DequantizeAll(qt)
	qt.Data = nil
}
