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

func (p *OpenAIProvider) ListModels(ctx context.Context) ([]string, error) {
	url := fmt.Sprintf("%s/models", p.BaseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to reach provider: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("provider returned status: %d", resp.StatusCode)
	}

	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse models list: %w", err)
	}

	var models []string
	for _, m := range result.Data {
		models = append(models, m.ID)
	}

	return models, nil
}

func (p *OpenAIProvider) PullModel(ctx context.Context, modelID string) error {
	return fmt.Errorf("model pulling is not supported by LM Studio / OpenAI providers")
}

func (p *OpenAIProvider) TestConnection(ctx context.Context) (string, error) {
	models, err := p.ListModels(ctx)
	if err != nil {
		return "", err
	}

	if len(models) > 0 {
		return models[0], nil
	}

	return "Connected", nil
}
