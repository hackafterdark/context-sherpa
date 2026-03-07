package memory

// SSMLayerState holds the recurrent state for one Gated Delta Net layer.
type SSMLayerState struct {
	// State holds the recurrent matrix S[head][kDim][vDim], stored flat as
	// [numHeads * headKDim * headVDim]. Updated every token.
	State []float32

	// ConvBuf holds the last (kernelSize-1) input vectors for the causal conv1d.
	// Stored as [(kernelSize-1) * channels], newest at the end.
	ConvBuf []float32

	NumHeads int
	HeadKDim int
	HeadVDim int
	Channels int // in-projection output dim (key_dim*2 + value_dim)
	ConvK    int // kernel size
}

// NewSSMLayerState allocates a fresh SSM layer state.
func NewSSMLayerState(numHeads, headKDim, headVDim, channels, convKernel int) *SSMLayerState {
	return &SSMLayerState{
		State:    make([]float32, numHeads*headKDim*headVDim),
		ConvBuf:  make([]float32, convKernel*channels),
		NumHeads: numHeads,
		HeadKDim: headKDim,
		HeadVDim: headVDim,
		Channels: channels,
		ConvK:    convKernel,
	}
}

// Reset zeros the recurrent state and conv buffer.
func (s *SSMLayerState) Reset() {
	for i := range s.State {
		s.State[i] = 0
	}
	for i := range s.ConvBuf {
		s.ConvBuf[i] = 0
	}
}

// SSMStateCache holds SSM states for all layers.
type SSMStateCache struct {
	Layers []*SSMLayerState
}

// NewSSMStateCache creates SSM state caches for the specified layer indices.
// layerIndices maps absolute layer index to whether it's an SSM layer.
func NewSSMStateCache(numLayers, numHeads, headKDim, headVDim, channels, convKernel int, isSSMLayer func(int) bool) *SSMStateCache {
	c := &SSMStateCache{
		Layers: make([]*SSMLayerState, numLayers),
	}
	for l := 0; l < numLayers; l++ {
		if isSSMLayer(l) {
			c.Layers[l] = NewSSMLayerState(numHeads, headKDim, headVDim, channels, convKernel)
		}
	}
	return c
}

// Reset zeros all layer states.
func (c *SSMStateCache) Reset() {
	for _, l := range c.Layers {
		if l != nil {
			l.Reset()
		}
	}
}
