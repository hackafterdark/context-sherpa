//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
)

func maxDiff(a, b []float32) (float32, int) {
	maxE := float32(0)
	maxI := 0
	for i := range a {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > maxE {
			maxE = d
			maxI = i
		}
	}
	return maxE, maxI
}

func main() {
	if err := gpu.Init(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer gpu.Shutdown()

	path := `C:\projects\evoke\models\gemma-3-1b-it-Q4_K_M.gguf`
	pipe, err := llm.NewPipeline(path, 512)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	cfg := pipe.Model.Config
	dim := cfg.EmbeddingDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	vocabSize := cfg.VocabSize

	cpuRS := llm.NewRunState(cfg, 512)
	cpuKV := memory.NewMultiLayerKVCache(cfg.NumLayers, 512, kvDim)

	gpuModel, _ := gpu.UploadModel(pipe.Model)
	qDim := cfg.NumHeads * cfg.HeadDim
	gpuRS := gpu.NewGpuRunState(dim, qDim, kvDim, cfg.FFNDim, vocabSize)
	gpuKV := gpu.NewGpuKVCache(cfg.NumLayers, 512, kvDim)

	token := int32(2) // BOS or some token
	pos := 0

	// CPU forward pass (single token)
	llm.Forward(pipe.Model, token, pos, cpuKV, cpuRS)
	cpuLogits := make([]float32, vocabSize)
	copy(cpuLogits, cpuRS.Logits)

	// GPU forward pass (single token) - then compare
	gpuLogits := make([]float32, vocabSize)
	gpu.GpuForward(pipe.Model, gpuModel, token, pos, gpuKV, gpuRS, gpuLogits)

	maxE, maxI := maxDiff(cpuLogits, gpuLogits)
	fmt.Printf("Full forward: maxErr=%.4f at %d\n", maxE, maxI)

	// Now let's compare intermediate state: download GPU X after each layer
	// We can't easily do per-layer comparison without modifying the forward pass
	// Instead, let's check if the hidden state X matches after the embedding
	cpuX := cpuRS.X[:dim]
	gpuX := make([]float32, dim)
	gpu.DownloadF32(gpuRS.X, gpuX)
	maxE, maxI = maxDiff(cpuX, gpuX)
	fmt.Printf("Hidden X after forward: maxErr=%.4f at idx %d\n", maxE, maxI)

	// Check the GPU logits buffer at specific regions
	fmt.Printf("\nCPU logit[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
		cpuLogits[0], cpuLogits[1], cpuLogits[2], cpuLogits[3], cpuLogits[4])
	fmt.Printf("GPU logit[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
		gpuLogits[0], gpuLogits[1], gpuLogits[2], gpuLogits[3], gpuLogits[4])

	cpuTop := argmax(cpuLogits)
	gpuTop := argmax(gpuLogits)
	fmt.Printf("CPU top=%d (%.4f), GPU top=%d (%.4f)\n", cpuTop, cpuLogits[cpuTop], gpuTop, gpuLogits[gpuTop])
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
