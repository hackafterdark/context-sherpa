package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// OpenAIProvider implements InferenceProvider for OpenAI-compatible APIs
type OpenAIProvider struct {
	BaseURL string
	Client  *http.Client
}

// NewOpenAIProvider creates a new OpenAI compatible provider
func NewOpenAIProvider(baseURL string) *OpenAIProvider {
	// If the URL is provided but missing the version suffix, add /v1
	// (Common for LM Studio and other local OpenAI-compatible servers)
	if u, err := url.Parse(baseURL); err == nil {
		if u.Path == "" || u.Path == "/" {
			u.Path = "/v1"
			baseURL = u.String()
		}
	}

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

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		var errRes struct {
			Error interface{} `json:"error"`
		}
		if err := json.Unmarshal(bodyBytes, &errRes); err == nil {
			switch v := errRes.Error.(type) {
			case string:
				return "", fmt.Errorf("openai error: %s", v)
			case map[string]interface{}:
				if msg, ok := v["message"].(string); ok {
					return "", fmt.Errorf("openai error: %s", msg)
				}
			}
		}
		return "", fmt.Errorf("openai returned status: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	var result openAICompletionResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return "", fmt.Errorf("failed to parse openai response: %w (body: %s)", err, string(bodyBytes))
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

	var models []string = []string{}
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
