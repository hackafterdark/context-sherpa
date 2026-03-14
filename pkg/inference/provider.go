package inference

import (
	"context"
)

// GenOptions represents the parameters for a generation request
type GenOptions struct {
	ModelID     string  `json:"modelId"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
}

// InferenceProvider defines the interface for external inference engines
type InferenceProvider interface {
	// Generate performs a full generation request to the provider
	Generate(ctx context.Context, prompt string, options GenOptions) (string, error)
	// TestConnection verifies that the provider is reachable and functional
	TestConnection(ctx context.Context) (string, error)
}
