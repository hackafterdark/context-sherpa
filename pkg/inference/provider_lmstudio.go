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

// LMStudioProvider implements InferenceProvider for LM Studio Native API (/api/v1)
type LMStudioProvider struct {
	BaseURL string
	Client  *http.Client
}

// NewLMStudioProvider creates a new LM Studio native provider
func NewLMStudioProvider(baseURL string) *LMStudioProvider {
	// Ensure the base URL includes the /api/v1 prefix if not present
	// (LM Studio native API requires it for the endpoints we use)
	if u, err := url.Parse(baseURL); err == nil {
		if u.Path == "" || u.Path == "/" {
			u.Path = "/api/v1"
			baseURL = u.String()
		}
	}

	return &LMStudioProvider{
		BaseURL: baseURL,
		Client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

type lmStudioChatRequest struct {
	Model        string  `json:"model"`
	SystemPrompt string  `json:"system_prompt,omitempty"`
	Input        string  `json:"input"`
	MaxTokens    int     `json:"max_tokens,omitempty"`
	Temperature  float32 `json:"temperature,omitempty"`
}

type lmStudioChatResponse struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error interface{} `json:"error,omitempty"`
}

func (p *LMStudioProvider) Generate(ctx context.Context, prompt string, options GenOptions) (string, error) {
	url := fmt.Sprintf("%s/chat", p.BaseURL)

	reqBody := lmStudioChatRequest{
		Model: options.ModelID,
		Input: prompt,
		// In native API, we use system_prompt for context if available
		// But usually prompt contains both context and question already in Service.Execute
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
		return "", fmt.Errorf("lm studio error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	var result lmStudioChatResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return "", fmt.Errorf("failed to parse lm studio response: %w (body: %s)", err, string(bodyBytes))
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from LM Studio")
	}

	return result.Choices[0].Message.Content, nil
}

func (p *LMStudioProvider) ListModels(ctx context.Context) ([]string, error) {
	url := fmt.Sprintf("%s/models", p.BaseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to reach LM Studio: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("LM Studio returned status: %d", resp.StatusCode)
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

func (p *LMStudioProvider) PullModel(ctx context.Context, modelID string) error {
	url := fmt.Sprintf("%s/models/download", p.BaseURL)
	
	reqBody := map[string]string{
		"model_id": modelID,
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

	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusAccepted {
		return nil
	}

	bodyBytes, _ := io.ReadAll(resp.Body)
	return fmt.Errorf("LM Studio download failed (status %d): %s", resp.StatusCode, string(bodyBytes))
}

func (p *LMStudioProvider) TestConnection(ctx context.Context) (string, error) {
	models, err := p.ListModels(ctx)
	if err != nil {
		return "", err
	}

	if len(models) > 0 {
		return models[0], nil
	}

	return "Connected", nil
}
