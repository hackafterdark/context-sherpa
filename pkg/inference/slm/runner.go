package slm

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/computerex/dlgo"
)

var (
	activeModelPath string
	activeModel     *dlgo.LLM
	modelMutex      sync.Mutex
)

// RunInference executes inference using the loaded dlgo model.
// If the model at modelPath is not already loaded, it will be loaded.
func RunInference(ctx context.Context, modelPath, prompt string, maxTokens int, temperature float32) (string, error) {
	modelMutex.Lock()
	defer modelMutex.Unlock()

	// Load or swap model if necessary
	if activeModelPath != modelPath || activeModel == nil {
		// Hijack stdout to prevent dlgo from corrupting the MCP pipe
		oldStdout := os.Stdout
		r, w, _ := os.Pipe()
		os.Stdout = w

		model, err := dlgo.LoadLLM(modelPath)

		// Restore stdout
		w.Close()
		os.Stdout = oldStdout
		_ = r.Close()

		if err != nil {
			return "", fmt.Errorf("failed to load model %s: %v", modelPath, err)
		}
		activeModel = model
		activeModelPath = modelPath
	}

	// Safeguards and defaults
	info := activeModel.ModelInfo()

	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Cap at a reasonable hard limit to prevent runaway generations
	const hardMaxTokens = 4096
	if maxTokens > hardMaxTokens {
		maxTokens = hardMaxTokens
	}

	// Ensure we don't exceed the model's context window
	if maxTokens > info.ContextLen {
		maxTokens = info.ContextLen
	}

	if temperature <= 0 {
		temperature = 0.1
	} else if temperature > 1.0 {
		temperature = 1.0
	}

	// We'll use a channel to capture the result since dlgo doesn't natively take a context for cancellation
	// but it's fast enough or we could just wait for it to finish.
	// Actually, Chat doesn't take context. We'll wrap it in a goroutine for timeout enforcement.

	type chatResult struct {
		response string
		err      error
	}

	resultCh := make(chan chatResult, 1)

	go func() {
		// Hijack stdout to prevent dlgo from corrupting the MCP pipe during inference
		oldStdout := os.Stdout
		r, w, _ := os.Pipe()
		os.Stdout = w

		// Run inference
		// We use 512 max tokens as we did before
		response, err := activeModel.Chat("", prompt, dlgo.WithMaxTokens(maxTokens), dlgo.WithTemperature(temperature))

		// Restore stdout
		w.Close()
		os.Stdout = oldStdout
		_ = r.Close()

		resultCh <- chatResult{response: response, err: err}
	}()

	select {
	case <-ctx.Done():
		return "", fmt.Errorf("inference timed out or cancelled")
	case res := <-resultCh:
		if res.err != nil {
			return "", fmt.Errorf("inference failed: %v", res.err)
		}
		return strings.TrimSpace(res.response), nil
	}
}
