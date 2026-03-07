// Example: streaming token generation with dlgo.
//
// Usage:
//
//	go run . [model.gguf]
package main

import (
	"fmt"
	"os"

	dlgo "github.com/computerex/dlgo"
)

func main() {
	modelPath := `C:\projects\evoke\models\tinyllama-1.1b-chat-v1.0.Q4_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	model, err := dlgo.LoadLLM(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Print("AI: ")
	err = model.ChatStream("You are a helpful assistant.", "Tell me a fun fact about space.",
		func(token string) { fmt.Print(token) },
		dlgo.WithMaxTokens(128),
		dlgo.WithTemperature(0.7),
		dlgo.WithSeed(42),
	)
	fmt.Println()

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
