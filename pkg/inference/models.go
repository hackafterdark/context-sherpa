package inference

import (
)

// ModelInfo represents metadata about a local model
type ModelInfo struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Size        int64  `json:"size"`
	Path        string `json:"path"`
	Downloaded  bool   `json:"downloaded"`
	DownloadURL string `json:"downloadUrl"`
	Description string `json:"description"`
	LastUsed    string `json:"lastUsed"`
}

// InferenceRequest represents a request to the Local SLM
type InferenceRequest struct {
	ModelID     string  `json:"modelId"`
	Prompt      string  `json:"prompt"`
	Context     string  `json:"context"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
}

// InferenceResponse represents the distilled output from Local SLM
type InferenceResponse struct {
	Text  string `json:"text"`
	Error string `json:"error,omitempty"`
}
