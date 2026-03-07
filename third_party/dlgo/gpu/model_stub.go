//go:build !vulkan || !cgo

package gpu

import "github.com/computerex/dlgo/core"

type GpuTensor struct {
	Buf  Buf
	Type uint32
	Rows int
	Cols int
}

type GpuLayer struct{}
type GpuModel struct {
	TokenEmbed *GpuTensor
	OutputNorm Buf
	Output     *GpuTensor
	Layers     []GpuLayer
}
type GpuRunState struct{}
type GpuKVCache struct{}

func UploadTensor(*core.QuantizedTensor) (*GpuTensor, error) { return nil, errNoGPU }
func UploadF32Slice([]float32) (Buf, error)                  { return 0, errNoGPU }
func NewGpuRunState(_, _, _, _, _ int) *GpuRunState           { return nil }
func NewGpuKVCache(_, _, _ int) *GpuKVCache                   { return nil }
func (c *GpuKVCache) Reset()                                  {}
