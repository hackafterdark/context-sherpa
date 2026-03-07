// Package memory provides buffer management utilities for neural network inference.
//
// BufferPool reduces allocation pressure during inference by reusing float32 slices.
// KVCache provides key-value caching for transformer attention layers.
package memory

import "sync"

// BufferPool caches float32 slices for reuse across inference calls.
// Buffers are stored by approximate capacity tier to avoid excessive fragmentation.
type BufferPool struct {
	mu    sync.Mutex
	pools map[int][]float32Slice
}

type float32Slice struct {
	data []float32
}

// NewBufferPool creates a new buffer pool.
func NewBufferPool() *BufferPool {
	return &BufferPool{
		pools: make(map[int][]float32Slice),
	}
}

// tier rounds n up to the nearest power of 2 for bucketing.
func tier(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// Get returns a float32 slice of at least n elements.
// The returned slice has length n but may have larger capacity.
func (p *BufferPool) Get(n int) []float32 {
	t := tier(n)
	p.mu.Lock()
	pool := p.pools[t]
	if len(pool) > 0 {
		s := pool[len(pool)-1]
		p.pools[t] = pool[:len(pool)-1]
		p.mu.Unlock()
		return s.data[:n]
	}
	p.mu.Unlock()
	return make([]float32, n, t)
}

// Put returns a slice to the pool for reuse.
func (p *BufferPool) Put(s []float32) {
	if cap(s) == 0 {
		return
	}
	t := tier(cap(s))
	p.mu.Lock()
	p.pools[t] = append(p.pools[t], float32Slice{data: s[:cap(s)]})
	p.mu.Unlock()
}

// GetZeroed returns a zeroed float32 slice of at least n elements.
func (p *BufferPool) GetZeroed(n int) []float32 {
	s := p.Get(n)
	for i := range s {
		s[i] = 0
	}
	return s
}
