package inference

import (
	"time"
)

// ModelInfo represents metadata about a local model
type ModelInfo struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Size        int64     `json:"size"`
	Path        string    `json:"path"`
	Downloaded  bool      `json:"downloaded"`
	DownloadURL string    `json:"downloadUrl"`
	Description string    `json:"description"`
	LastUsed    time.Time `json:"lastUsed"`
}

// InferenceRequest represents a request to the Little Brain
type InferenceRequest struct {
	ModelID     string  `json:"modelId"`
	Prompt      string  `json:"prompt"`
	Context     string  `json:"context"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
}

// InferenceResponse represents the distilled output from Little Brain
type InferenceResponse struct {
	Text string `json:"text"`
	Error string `json:"error,omitempty"`
}
