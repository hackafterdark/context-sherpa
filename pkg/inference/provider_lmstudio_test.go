package inference

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLMStudioProvider_ListModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/models", r.URL.Path)
		assert.Equal(t, "GET", r.Method)

		resp := struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}{
			Data: []struct {
				ID string `json:"id"`
			}{
				{ID: "llama-3-8b"},
				{ID: "qwen-2.5-0.5b"},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewLMStudioProvider(server.URL)
	models, err := provider.ListModels(context.Background())

	assert.NoError(t, err)
	assert.Equal(t, []string{"llama-3-8b", "qwen-2.5-0.5b"}, models)
}

func TestLMStudioProvider_Generate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/chat", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var req lmStudioChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		assert.Equal(t, "test-model", req.Model)
		assert.Equal(t, "Hello", req.Input)

		resp := lmStudioChatResponse{
			Choices: []struct {
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
			}{
				{
					Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{
						Role:    "assistant",
						Content: "Hi there!",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewLMStudioProvider(server.URL)
	response, err := provider.Generate(context.Background(), "Hello", GenOptions{ModelID: "test-model"})

	assert.NoError(t, err)
	assert.Equal(t, "Hi there!", response)
}

func TestLMStudioProvider_PullModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/models/download", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var req map[string]string
		json.NewDecoder(r.Body).Decode(&req)
		assert.Equal(t, "new-model", req["model_id"])

		w.WriteHeader(http.StatusAccepted)
	}))
	defer server.Close()

	provider := NewLMStudioProvider(server.URL)
	err := provider.PullModel(context.Background(), "new-model")

	assert.NoError(t, err)
}
