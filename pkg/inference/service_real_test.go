package inference

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestInferenceService_RealInference(t *testing.T) {
	configDir := filepath.Join(os.Getenv("LOCALAPPDATA"), "context-sherpa")
	modelsDir := filepath.Join(configDir, "models")

	s := NewInferenceService(modelsDir)
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := s.Execute(ctx, InferenceRequest{
		ModelID: "qwen2.5-0.5b",
		Prompt:  "Hello, are you there?",
	})

	if err != nil {
		t.Fatalf("Inference failed: %v", err)
	}

	fmt.Printf("Inference Response: %s\n", resp.Text)
	if resp.Text == "" {
		t.Error("expected non-empty response")
	}
}
