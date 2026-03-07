package memory

// KVCache stores key and value tensors for one attention layer.
// Keys and Vals are stored as [maxPos][dim], with Len tracking the current fill level.
type KVCache struct {
	Keys [][]float32 // [maxPos][dim]
	Vals [][]float32 // [maxPos][dim]
	Len  int
}

// NewKVCache creates a KV cache that can hold up to maxPos positions of dimension dim.
func NewKVCache(maxPos, dim int) *KVCache {
	c := &KVCache{
		Keys: make([][]float32, maxPos),
		Vals: make([][]float32, maxPos),
	}
	for p := 0; p < maxPos; p++ {
		c.Keys[p] = make([]float32, dim)
		c.Vals[p] = make([]float32, dim)
	}
	return c
}

// Reset clears the cache without deallocating.
func (c *KVCache) Reset() {
	c.Len = 0
}

// Store writes key and value vectors at the given position.
func (c *KVCache) Store(pos int, key, val []float32) {
	copy(c.Keys[pos], key)
	copy(c.Vals[pos], val)
	if pos+1 > c.Len {
		c.Len = pos + 1
	}
}

// Clone creates a deep copy of the cache up to currentPos.
func (c *KVCache) Clone(currentPos int) *KVCache {
	d := &KVCache{
		Keys: make([][]float32, len(c.Keys)),
		Vals: make([][]float32, len(c.Vals)),
		Len:  c.Len,
	}
	dim := len(c.Keys[0])
	for p := 0; p < len(c.Keys); p++ {
		d.Keys[p] = make([]float32, dim)
		d.Vals[p] = make([]float32, dim)
		if p < currentPos {
			copy(d.Keys[p], c.Keys[p])
			copy(d.Vals[p], c.Vals[p])
		}
	}
	return d
}

// MultiLayerKVCache holds KV caches for all layers of a transformer model.
type MultiLayerKVCache struct {
	Layers []*KVCache
}

// NewMultiLayerKVCache creates caches for nLayers transformer layers.
func NewMultiLayerKVCache(nLayers, maxPos, dim int) *MultiLayerKVCache {
	m := &MultiLayerKVCache{
		Layers: make([]*KVCache, nLayers),
	}
	for l := 0; l < nLayers; l++ {
		m.Layers[l] = NewKVCache(maxPos, dim)
	}
	return m
}

// Reset clears all layer caches.
func (m *MultiLayerKVCache) Reset() {
	for _, l := range m.Layers {
		l.Reset()
	}
}
