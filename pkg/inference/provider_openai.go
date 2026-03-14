package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// OpenAIProvider implements InferenceProvider for OpenAI-compatible APIs
type OpenAIProvider struct {
	BaseURL string
	Client  *http.Client
}

// NewOpenAIProvider creates a new OpenAI compatible provider
func NewOpenAIProvider(baseURL string) *OpenAIProvider {
	return &OpenAIProvider{
		BaseURL: baseURL,
		Client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

type openAICompletionRequest struct {
	Model       string         `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	MaxTokens   int            `json:"max_tokens,omitempty"`
	Temperature float32        `json:"temperature,omitempty"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAICompletionResponse struct {
	Choices []struct {
		Message openAIMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (p *OpenAIProvider) Generate(ctx context.Context, prompt string, options GenOptions) (string, error) {
	url := fmt.Sprintf("%s/chat/completions", p.BaseURL)

	reqBody := openAICompletionRequest{
		Model: options.ModelID,
		Messages: []openAIMessage{
			{Role: "user", Content: prompt},
		},
		MaxTokens:   options.MaxTokens,
		Temperature: options.Temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.Client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result openAICompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if result.Error != nil {
		return "", fmt.Errorf("openai error: %s", result.Error.Message)
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from provider")
	}

	return result.Choices[0].Message.Content, nil
}

func (p *OpenAIProvider) TestConnection(ctx context.Context) (string, error) {
	// Attempt to list models to verify connectivity
	url := fmt.Sprintf("%s/models", p.BaseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to reach provider: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("provider returned status: %d", resp.StatusCode)
	}

	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to parse models list: %w", err)
	}

	if len(result.Data) > 0 {
		return result.Data[0].ID, nil
	}

	return "Connected", nil
}
