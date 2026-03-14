// Interactive example: multi-turn chat with an LLM and per-turn performance stats.
//
// Usage:
//
//	go run . [--ctx N] [--max-tokens N] [--temp T] [model.gguf]
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	ctx := flag.Int("ctx", 8192, "runtime context length (tokens)")
	maxTokens := flag.Int("max-tokens", 256, "max tokens per assistant response")
	temp := flag.Float64("temp", 0.7, "sampling temperature (0 = greedy)")
	flag.Parse()

	modelPath := `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`
	if flag.NArg() > 0 {
		modelPath = flag.Arg(0)
	}

	if *ctx <= 0 {
		fmt.Fprintln(os.Stderr, "Error: --ctx must be > 0")
		os.Exit(1)
	}
	if *maxTokens <= 0 {
		fmt.Fprintln(os.Stderr, "Error: --max-tokens must be > 0")
		os.Exit(1)
	}
	if *temp < 0 {
		fmt.Fprintln(os.Stderr, "Error: --temp must be >= 0")
		os.Exit(1)
	}

	pipe, err := llm.NewPipeline(modelPath, *ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = *maxTokens
	cfg.Sampler.Temperature = float32(*temp)

	system := "You are a helpful assistant."
	messages := []llm.Message{{Role: "system", Content: system}}

	fmt.Printf("Model: %s (%d layers, %d dim, %d heads, vocab %d, ctx %d)\n",
		pipe.Model.Config.Architecture,
		pipe.Model.Config.NumLayers,
		pipe.Model.Config.EmbeddingDim,
		pipe.Model.Config.NumHeads,
		pipe.Model.Config.VocabSize,
		pipe.Model.Config.ContextLength,
	)
	fmt.Printf("Runtime context (--ctx): %d tokens\n", pipe.MaxSeqLen)
	fmt.Printf("Generation: max-tokens=%d temp=%.2f\n", cfg.MaxTokens, cfg.Sampler.Temperature)
	fmt.Println("Interactive chat ready. Type 'exit' or 'quit' to leave.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024), 1024*1024)
	for {
		fmt.Print("You> ")
		if !scanner.Scan() {
			fmt.Println()
			break
		}

		user := strings.TrimSpace(scanner.Text())
		if user == "" {
			continue
		}
		if strings.EqualFold(user, "exit") || strings.EqualFold(user, "quit") {
			break
		}

		messages = append(messages, llm.Message{Role: "user", Content: user})
		prompt := llm.FormatMessages(pipe.Model.Config, messages)

		result, err := pipe.GenerateDetailed(prompt, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
			continue
		}

		response := strings.TrimSpace(result.Text)
		if response == "" {
			response = "(empty response)"
		}

		fmt.Println("AI>", response)
		fmt.Printf("   [%.1f tok/s | prefill %.0f ms | gen %.0f ms | prompt %d tok | output %d tok]\n\n",
			result.TokensPerSec,
			result.PrefillTimeMs,
			result.GenerateTimeMs,
			result.PromptTokens,
			result.TotalTokens,
		)

		messages = append(messages, llm.Message{Role: "assistant", Content: response})
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Input error: %v\n", err)
		os.Exit(1)
	}
}
