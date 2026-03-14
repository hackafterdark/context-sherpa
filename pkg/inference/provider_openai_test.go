package inference

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOpenAIProvider_Normalization(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		expected string
	}{
		{
			name:     "bare host",
			baseURL:  "http://localhost:1234",
			expected: "http://localhost:1234/v1",
		},
		{
			name:     "host with slash",
			baseURL:  "http://localhost:1234/",
			expected: "http://localhost:1234/v1",
		},
		{
			name:     "already has v1",
			baseURL:  "http://localhost:1234/v1",
			expected: "http://localhost:1234/v1",
		},
		{
			name:     "has other version",
			baseURL:  "http://localhost:1234/v2",
			expected: "http://localhost:1234/v2",
		},
		{
			name:     "has api prefix",
			baseURL:  "http://localhost:1234/api/v1",
			expected: "http://localhost:1234/api/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewOpenAIProvider(tt.baseURL)
			assert.Equal(t, tt.expected, p.BaseURL)
		})
	}
}

func TestOpenAIProvider_ListModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/models", r.URL.Path)
		assert.Equal(t, "GET", r.Method)

		resp := struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}{
			Data: []struct {
				ID string `json:"id"`
			}{
				{ID: "gpt-3.5-turbo"},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// NewOpenAIProvider(server.URL) should normalize to include /v1
	provider := NewOpenAIProvider(server.URL)
	models, err := provider.ListModels(context.Background())

	assert.NoError(t, err)
	assert.Equal(t, []string{"gpt-3.5-turbo"}, models)
}
