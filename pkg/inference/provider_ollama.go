package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// OllamaProvider implements InferenceProvider for Ollama native API
type OllamaProvider struct {
	BaseURL string
	Client  *http.Client
}

// NewOllamaProvider creates a new Ollama provider
func NewOllamaProvider(baseURL string) *OllamaProvider {
	return &OllamaProvider{
		BaseURL: baseURL,
		Client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

type ollamaGenerateRequest struct {
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	Stream  bool   `json:"stream"`
	Options struct {
		NumPredict  int     `json:"num_predict,omitempty"`
		Temperature float32 `json:"temperature,omitempty"`
	} `json:"options,omitempty"`
}

type ollamaGenerateResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func (p *OllamaProvider) Generate(ctx context.Context, prompt string, options GenOptions) (string, error) {
	url := fmt.Sprintf("%s/api/generate", p.BaseURL)

	reqBody := ollamaGenerateRequest{
		Model:  options.ModelID,
		Prompt: prompt,
		Stream: false,
	}
	reqBody.Options.NumPredict = options.MaxTokens
	reqBody.Options.Temperature = options.Temperature

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

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama returned status: %d", resp.StatusCode)
	}

	var result ollamaGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	return result.Response, nil
}

func (p *OllamaProvider) ListModels(ctx context.Context) ([]string, error) {
	url := fmt.Sprintf("%s/api/tags", p.BaseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to reach ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status: %d", resp.StatusCode)
	}

	var result struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse ollama models: %w", err)
	}

	var models []string
	for _, m := range result.Models {
		models = append(models, m.Name)
	}

	return models, nil
}

func (p *OllamaProvider) PullModel(ctx context.Context, modelID string) error {
	url := fmt.Sprintf("%s/api/pull", p.BaseURL)

	reqBody := map[string]interface{}{
		"name":   modelID,
		"stream": false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResult struct {
			Error string `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&errResult); err == nil && errResult.Error != "" {
			return fmt.Errorf("ollama pull failed: %s", errResult.Error)
		}
		return fmt.Errorf("ollama pull failed with status: %d", resp.StatusCode)
	}

	return nil
}

func (p *OllamaProvider) TestConnection(ctx context.Context) (string, error) {
	models, err := p.ListModels(ctx)
	if err != nil {
		return "", err
	}

	if len(models) > 0 {
		return models[0], nil
	}

	return "Connected", nil
}
