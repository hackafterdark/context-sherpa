package inference

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hackafterdark/context-sherpa/pkg/inference/slm"
)

// InferenceService manages the local SLM execution natively
type InferenceService struct {
	modelsDir string
}

// NewInferenceService creates a new inference service
func NewInferenceService(modelsDir string) *InferenceService {
	return &InferenceService{
		modelsDir: modelsDir,
	}
}

// Execute performs inference using the local model natively
func (s *InferenceService) Execute(ctx context.Context, req InferenceRequest) (*InferenceResponse, error) {
	if req.ModelID == "" {
		req.ModelID = "qwen2.5-0.5b"
	}

	// 1. Find the model file
	modelPath := filepath.Join(s.modelsDir, req.ModelID)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model %s not found at %s", req.ModelID, modelPath)
	}

	// 2. Prepare prompt
	prompt := req.Prompt
	if req.Context != "" {
		prompt = fmt.Sprintf("Context:\n%s\n\nQuestion: %s", req.Context, req.Prompt)
	}

	// 3. Delegate to the native SLM runner
	text, err := slm.RunInference(ctx, modelPath, prompt, req.MaxTokens, req.Temperature)
	if err != nil {
		return nil, err
	}

	return &InferenceResponse{
		Text: text,
	}, nil
}
