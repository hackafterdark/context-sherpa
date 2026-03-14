package inference

import (
	"context"
	"fmt"
)

// InferenceService manages the external inference execution
type InferenceService struct {
	provider InferenceProvider
}

// NewInferenceService creates a new inference service
func NewInferenceService(provider InferenceProvider) *InferenceService {
	return &InferenceService{
		provider: provider,
	}
}

// Execute performs inference using the configured provider
func (s *InferenceService) Execute(ctx context.Context, req InferenceRequest) (*InferenceResponse, error) {
	if s.provider == nil {
		return nil, fmt.Errorf("no inference provider configured")
	}

	// Prepare prompt
	prompt := req.Prompt
	if req.Context != "" {
		prompt = fmt.Sprintf("Context:\n%s\n\nQuestion: %s", req.Context, req.Prompt)
	}

	// Delegate to the provider
	text, err := s.provider.Generate(ctx, prompt, GenOptions{
		ModelID:     req.ModelID,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	})
	if err != nil {
		return nil, err
	}

	return &InferenceResponse{
		Text: text,
	}, nil
}
