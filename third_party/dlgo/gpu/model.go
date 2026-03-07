//go:build cgo && vulkan

package gpu

import (
	"fmt"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/quant"
)

// GpuTensor mirrors core.QuantizedTensor but with data on the GPU.
type GpuTensor struct {
	Buf  Buf
	Type uint32
	Rows int
	Cols int
}

// UploadTensor copies a QuantizedTensor's raw data to GPU memory.
func UploadTensor(qt *core.QuantizedTensor) (*GpuTensor, error) {
	if qt == nil {
		return nil, nil
	}

	var data []byte
	var size uint64

	if qt.FP32Data != nil {
		size = uint64(len(qt.FP32Data) * 4)
		buf := Alloc(size)
		if buf == 0 {
			return nil, fmt.Errorf("gpu: alloc failed for tensor %dx%d", qt.Rows, qt.Cols)
		}
		if err := UploadF32(buf, qt.FP32Data); err != nil {
			Free(buf)
			return nil, err
		}
		return &GpuTensor{Buf: buf, Type: 0, Rows: qt.Rows, Cols: qt.Cols}, nil
	}

	data = qt.Data
	size = uint64(len(data))
	buf := Alloc(size)
	if buf == 0 {
		return nil, fmt.Errorf("gpu: alloc failed for tensor %dx%d (%d bytes)", qt.Rows, qt.Cols, size)
	}
	if err := Upload(buf, data); err != nil {
		Free(buf)
		return nil, err
	}
	return &GpuTensor{Buf: buf, Type: qt.Type, Rows: qt.Rows, Cols: qt.Cols}, nil
}

// UploadF32Slice uploads a float32 slice to a new GPU buffer.
func UploadF32Slice(data []float32) (Buf, error) {
	if len(data) == 0 {
		return 0, nil
	}
	buf := Alloc(uint64(len(data) * 4))
	if buf == 0 {
		return 0, fmt.Errorf("gpu: alloc failed for %d floats", len(data))
	}
	if err := UploadF32(buf, data); err != nil {
		Free(buf)
		return 0, err
	}
	return buf, nil
}

// BytesPerRow returns the byte size of one row for the tensor's quant type.
func (gt *GpuTensor) BytesPerRow() int {
	if gt.Type == 0 {
		return gt.Cols * 4
	}
	return quant.BytesForN(gt.Type, gt.Cols)
}

// GpuLayer holds GPU buffers for one transformer layer's weights.
type GpuLayer struct {
	AttnNorm     Buf
	AttnNormBias Buf
	Wq           *GpuTensor
	Wk           *GpuTensor
	Wv           *GpuTensor
	Wo           *GpuTensor
	Bq, Bk, Bv  Buf
	Bo           Buf
	AttnQNorm    Buf
	AttnKNorm    Buf
	PostAttnNorm Buf
	FFNNorm      Buf
	FFNGate      *GpuTensor
	FFNUp        *GpuTensor
	FFNDown      *GpuTensor
	FFNUpBias    Buf
	FFNDownBias  Buf
	PostFFNNorm  Buf
}

// GpuModel holds all model weights on the GPU.
type GpuModel struct {
	TokenEmbed   *GpuTensor
	OutputNorm   Buf
	OutputNormBias Buf
	Output       *GpuTensor
	OutputBias   Buf
	Layers       []GpuLayer
}

// GpuRunState holds GPU buffers for intermediate activations during inference.
type GpuRunState struct {
	X        Buf // [dim]
	XNorm    Buf // [dim]
	Q        Buf // [qDim]
	K        Buf // [kvDim]
	V        Buf // [kvDim]
	AttnOut  Buf // [qDim]
	AttnProj Buf // [dim]
	FFNIn    Buf // [dim]
	FFNNorm  Buf // [dim]
	Gate     Buf // [ffnDim]
	Up       Buf // [ffnDim]
	Hidden   Buf // [ffnDim]
	FFNOut   Buf // [dim]
	Logits   Buf // [vocabSize]
}

// NewGpuRunState allocates all GPU activation buffers.
func NewGpuRunState(dim, qDim, kvDim, ffnDim, vocabSize int) *GpuRunState {
	return &GpuRunState{
		X:        Alloc(uint64(dim * 4)),
		XNorm:    Alloc(uint64(dim * 4)),
		Q:        Alloc(uint64(qDim * 4)),
		K:        Alloc(uint64(kvDim * 4)),
		V:        Alloc(uint64(kvDim * 4)),
		AttnOut:  Alloc(uint64(qDim * 4)),
		AttnProj: Alloc(uint64(dim * 4)),
		FFNIn:    Alloc(uint64(dim * 4)),
		FFNNorm:  Alloc(uint64(dim * 4)),
		Gate:     Alloc(uint64(ffnDim * 4)),
		Up:       Alloc(uint64(ffnDim * 4)),
		Hidden:   Alloc(uint64(ffnDim * 4)),
		FFNOut:   Alloc(uint64(dim * 4)),
		Logits:   Alloc(uint64(vocabSize * 4)),
	}
}

// GpuKVCache holds GPU-resident KV cache for all layers.
type GpuKVCache struct {
	KeyBufs []Buf // [nLayers] each is [maxSeqLen * kvDim] floats
	ValBufs []Buf
	KVDim   int
	MaxSeq  int
	Len     int
}

// NewGpuKVCache allocates GPU buffers for KV cache.
func NewGpuKVCache(nLayers, maxSeqLen, kvDim int) *GpuKVCache {
	c := &GpuKVCache{
		KeyBufs: make([]Buf, nLayers),
		ValBufs: make([]Buf, nLayers),
		KVDim:   kvDim,
		MaxSeq:  maxSeqLen,
	}
	size := uint64(maxSeqLen * kvDim * 4)
	for l := 0; l < nLayers; l++ {
		c.KeyBufs[l] = Alloc(size)
		c.ValBufs[l] = Alloc(size)
	}
	return c
}

func (c *GpuKVCache) Reset() { c.Len = 0 }
