package llm

import (
	"math"
	"sync"
	"sync/atomic"
)

// WorkerPool manages persistent goroutines for parallel inference.
// Each worker has its own task channel; work is distributed round-robin.
type WorkerPool struct {
	numWorkers int
	taskChs    []chan func()
	wg         sync.WaitGroup
	asyncWg    sync.WaitGroup
	alive      atomic.Bool
}

// NewWorkerPool creates a pool with n persistent workers.
// Callers must call Shutdown when done.
func NewWorkerPool(n int) *WorkerPool {
	if n < 1 {
		n = 1
	}
	wp := &WorkerPool{
		numWorkers: n,
		taskChs:    make([]chan func(), n),
	}
	wp.alive.Store(true)
	for i := 0; i < n; i++ {
		ch := make(chan func(), 1)
		wp.taskChs[i] = ch
		wp.wg.Add(1)
		go wp.worker(i, ch)
	}
	return wp
}

func (wp *WorkerPool) worker(id int, ch chan func()) {
	defer wp.wg.Done()
	for fn := range ch {
		if fn != nil {
			fn()
		}
	}
}

// Dispatch distributes work across numActive workers and blocks until complete.
// total is the total number of items; work(workerID, start, end) processes [start, end).
func (wp *WorkerPool) Dispatch(total, numActive int, work func(workerID, start, end int)) {
	if total <= 0 || numActive <= 0 || !wp.alive.Load() {
		return
	}
	if numActive > wp.numWorkers {
		numActive = wp.numWorkers
	}
	if numActive > total {
		numActive = total
	}

	chunk := (total + numActive - 1) / numActive
	var done sync.WaitGroup
	done.Add(numActive)

	for w := 0; w < numActive; w++ {
		start := w * chunk
		end := start + chunk
		if start >= total {
			done.Done()
			continue
		}
		if end > total {
			end = total
		}
		workerID := w
		s, e := start, end
		task := func() {
			defer done.Done()
			work(workerID, s, e)
		}
		ch := wp.taskChs[w%wp.numWorkers]
		ch <- task
	}

	done.Wait()
}

// DispatchAsync distributes work without waiting.
// Call Wait() to block until all async work completes.
func (wp *WorkerPool) DispatchAsync(total, numActive int, work func(workerID, start, end int)) {
	if total <= 0 || numActive <= 0 || !wp.alive.Load() {
		return
	}
	if numActive > wp.numWorkers {
		numActive = wp.numWorkers
	}
	if numActive > total {
		numActive = total
	}

	chunk := (total + numActive - 1) / numActive
	wp.asyncWg.Add(numActive)

	for w := 0; w < numActive; w++ {
		start := w * chunk
		end := start + chunk
		if start >= total {
			wp.asyncWg.Done()
			continue
		}
		if end > total {
			end = total
		}
		workerID := w
		s, e := start, end
		task := func() {
			defer wp.asyncWg.Done()
			work(workerID, s, e)
		}
		ch := wp.taskChs[w%wp.numWorkers]
		select {
		case ch <- task:
		default:
			// Channel full; run synchronously to avoid deadlock
			task()
		}
	}
}

// Wait blocks until all work dispatched via DispatchAsync has completed.
func (wp *WorkerPool) Wait() {
	wp.asyncWg.Wait()
}

// Shutdown stops all workers and waits for them to exit.
func (wp *WorkerPool) Shutdown() {
	wp.alive.Store(false)
	for _, ch := range wp.taskChs {
		close(ch)
	}
	wp.wg.Wait()
}

// PrecomputeRoPE fills RunState with precomputed cos/sin tables for RoPE.
// maxSeqLen: maximum sequence length; ropeDim, headDim: dimensions; freqBase: RoPE frequency base.
// ropeDim may be less than headDim for partial RoPE (e.g. Phi-2: 32 of 80).
func (rs *RunState) PrecomputeRoPE(maxSeqLen, ropeDim, headDim int, freqBase float32) {
	if maxSeqLen <= 0 || headDim <= 0 {
		return
	}
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}
	half := ropeDim / 2
	if half <= 0 {
		return
	}
	rs.ropeHeadDim = headDim
	rs.ropeDim = ropeDim
	rs.ropeNeox = false
	rs.ropeCos = make([]float32, maxSeqLen*half)
	rs.ropeSin = make([]float32, maxSeqLen*half)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < half; i++ {
			theta := 1.0 / math.Pow(float64(freqBase), float64(2*i)/float64(ropeDim))
			angle := float64(pos) * theta
			rs.ropeCos[pos*half+i] = float32(math.Cos(angle))
			rs.ropeSin[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// SetRopeNeox sets whether to use NeoX-style (split-half) RoPE pairing.
// Call after PrecomputeRoPE if needed.
func (rs *RunState) SetRopeNeox(neox bool) {
	rs.ropeNeox = neox
}

// ApplyRoPEFast applies precomputed RoPE to vec in-place.
// vec must be one head's worth [headDim]; pos is the token position.
// Only the first ropeDim dimensions are rotated; the rest pass through unchanged.
func (rs *RunState) ApplyRoPEFast(vec []float32, pos int) {
	if rs.ropeCos == nil || rs.ropeSin == nil || len(vec) < rs.ropeHeadDim {
		return
	}
	rd := rs.ropeDim
	if rd <= 0 {
		rd = rs.ropeHeadDim
	}
	half := rd / 2
	base := pos * half
	if base+half > len(rs.ropeCos) {
		return
	}

	if rs.ropeNeox {
		for i := 0; i < half; i++ {
			cos := rs.ropeCos[base+i]
			sin := rs.ropeSin[base+i]
			x0 := vec[i]
			x1 := vec[i+half]
			vec[i] = x0*cos - x1*sin
			vec[i+half] = x0*sin + x1*cos
		}
	} else {
		for i := 0; i < half; i++ {
			cos := rs.ropeCos[base+i]
			sin := rs.ropeSin[base+i]
			x0 := vec[2*i]
			x1 := vec[2*i+1]
			vec[2*i] = x0*cos - x1*sin
			vec[2*i+1] = x0*sin + x1*cos
		}
	}
}
