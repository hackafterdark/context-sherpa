package memory

import "testing"

func TestBufferPool_GetPut(t *testing.T) {
	pool := NewBufferPool()

	s := pool.Get(100)
	if len(s) != 100 {
		t.Errorf("Get(100) length = %d, want 100", len(s))
	}

	pool.Put(s)

	s2 := pool.Get(100)
	if len(s2) != 100 {
		t.Errorf("second Get(100) length = %d, want 100", len(s2))
	}
}

func TestBufferPool_GetZeroed(t *testing.T) {
	pool := NewBufferPool()

	s := pool.Get(10)
	for i := range s {
		s[i] = float32(i + 1)
	}
	pool.Put(s)

	z := pool.GetZeroed(10)
	for i, v := range z {
		if v != 0 {
			t.Errorf("GetZeroed: z[%d] = %f, want 0", i, v)
		}
	}
}

func TestBufferPool_DifferentSizes(t *testing.T) {
	pool := NewBufferPool()

	s1 := pool.Get(10)
	s2 := pool.Get(100)
	s3 := pool.Get(1000)

	pool.Put(s1)
	pool.Put(s2)
	pool.Put(s3)

	// Retrieving should work for any size
	g1 := pool.Get(10)
	g2 := pool.Get(100)
	if len(g1) != 10 || len(g2) != 100 {
		t.Errorf("wrong sizes: %d, %d", len(g1), len(g2))
	}
}

func TestKVCache(t *testing.T) {
	cache := NewKVCache(10, 4)

	if cache.Len != 0 {
		t.Errorf("initial Len = %d, want 0", cache.Len)
	}

	key := []float32{1, 2, 3, 4}
	val := []float32{5, 6, 7, 8}
	cache.Store(0, key, val)

	if cache.Len != 1 {
		t.Errorf("Len after Store(0) = %d, want 1", cache.Len)
	}

	if cache.Keys[0][0] != 1 || cache.Vals[0][0] != 5 {
		t.Error("stored values don't match")
	}

	// Clone
	clone := cache.Clone(1)
	if clone.Keys[0][0] != 1 {
		t.Error("clone should have same values")
	}

	// Modify original, verify independence
	cache.Keys[0][0] = 99
	if clone.Keys[0][0] != 1 {
		t.Error("clone should be independent")
	}

	// Reset
	cache.Reset()
	if cache.Len != 0 {
		t.Errorf("Len after Reset = %d, want 0", cache.Len)
	}
}

func TestMultiLayerKVCache(t *testing.T) {
	mlc := NewMultiLayerKVCache(3, 10, 4)

	if len(mlc.Layers) != 3 {
		t.Errorf("nLayers = %d, want 3", len(mlc.Layers))
	}

	mlc.Layers[0].Store(0, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8})
	mlc.Layers[1].Store(0, []float32{9, 10, 11, 12}, []float32{13, 14, 15, 16})

	mlc.Reset()
	for l, layer := range mlc.Layers {
		if layer.Len != 0 {
			t.Errorf("layer %d Len = %d after Reset, want 0", l, layer.Len)
		}
	}
}

func TestKVCache_StoreMultiple(t *testing.T) {
	cache := NewKVCache(5, 2)

	for i := 0; i < 5; i++ {
		cache.Store(i, []float32{float32(i), float32(i * 10)}, []float32{float32(i + 100), float32(i + 200)})
	}

	if cache.Len != 5 {
		t.Errorf("Len = %d, want 5", cache.Len)
	}

	if cache.Keys[3][0] != 3 || cache.Keys[3][1] != 30 {
		t.Errorf("Keys[3] = %v, want [3, 30]", cache.Keys[3])
	}
}

func TestTier(t *testing.T) {
	tests := []struct {
		input int
		want  int
	}{
		{1, 1},
		{2, 2},
		{3, 4},
		{5, 8},
		{100, 128},
		{256, 256},
		{257, 512},
	}
	for _, tt := range tests {
		got := tier(tt.input)
		if got != tt.want {
			t.Errorf("tier(%d) = %d, want %d", tt.input, got, tt.want)
		}
	}
}
