//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/computerex/dlgo/core"
	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/quant"
)

func typeName(t uint32) string {
	names := map[uint32]string{
		0: "F32", 1: "F16", 2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K",
	}
	if n, ok := names[t]; ok {
		return n
	}
	return fmt.Sprintf("?%d", t)
}

func testMatvec(name string, qt *core.QuantizedTensor) {
	if qt == nil {
		return
	}
	rows, cols := qt.Rows, qt.Cols

	// Create a simple input vector
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%7-3) * 0.01
	}

	// CPU reference: dequantize each row and dot
	cpuOut := make([]float32, rows)
	rowBuf := make([]float32, cols)
	for r := 0; r < rows; r++ {
		qt.DequantizeRow(r, rowBuf)
		var dot float32
		for c := 0; c < cols; c++ {
			dot += rowBuf[c] * x[c]
		}
		cpuOut[r] = dot
	}

	// GPU
	gpuTensor, err := gpu.UploadTensor(qt)
	if err != nil {
		fmt.Printf("  %s: upload fail: %v\n", name, err)
		return
	}
	xBuf := gpu.Alloc(uint64(cols * 4))
	outBuf := gpu.Alloc(uint64(rows * 4))
	gpu.UploadF32(xBuf, x)

	if err := gpu.MatVec(outBuf, gpuTensor.Buf, xBuf, rows, cols, gpuTensor.Type); err != nil {
		fmt.Printf("  %s [%s %dx%d]: matvec fail: %v\n", name, typeName(qt.Type), rows, cols, err)
		return
	}
	gpu.Sync()

	gpuOut := make([]float32, rows)
	gpu.DownloadF32(outBuf, gpuOut)

	maxErr := float32(0)
	maxIdx := 0
	for i := 0; i < rows; i++ {
		diff := float32(math.Abs(float64(cpuOut[i] - gpuOut[i])))
		if diff > maxErr {
			maxErr = diff
			maxIdx = i
		}
	}
	status := "OK"
	if maxErr > 1.0 {
		status = "FAIL"
	} else if maxErr > 0.1 {
		status = "WARN"
	}
	fmt.Printf("  %s [%s %dx%d]: maxErr=%.6f row=%d (cpu=%.4f gpu=%.4f) %s\n",
		name, typeName(qt.Type), rows, cols, maxErr, maxIdx, cpuOut[maxIdx], gpuOut[maxIdx], status)

	bytesPerRow := quant.BytesForN(qt.Type, cols)
	_ = bytesPerRow
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
	mdl := pipe.Model

	fmt.Println("Testing individual matvec correctness for Gemma 3 1B Q4_K_M:")
	testMatvec("token_embed", mdl.TokenEmbed)
	testMatvec("output", mdl.Output)

	for l := 0; l < min(3, len(mdl.Layers)); l++ {
		cl := &mdl.Layers[l]
		prefix := fmt.Sprintf("L%d", l)
		testMatvec(prefix+".wq", cl.Wq)
		testMatvec(prefix+".wk", cl.Wk)
		testMatvec(prefix+".wv", cl.Wv)
		testMatvec(prefix+".wo", cl.Wo)
		testMatvec(prefix+".ffn_gate", cl.FFNGate)
		testMatvec(prefix+".ffn_up", cl.FFNUp)
		testMatvec(prefix+".ffn_down", cl.FFNDown)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
