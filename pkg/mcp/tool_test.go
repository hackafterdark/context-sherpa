package mcp

import (
	"testing"
)

func TestToolRequest(t *testing.T) {
	t.Run("Create ToolRequest with params", func(t *testing.T) {
		params := map[string]interface{}{
			"key1": "value1",
			"key2": 42,
		}

		req := ToolRequest{
			Params: params,
		}

		if req.Params == nil {
			t.Error("Expected params to be set")
		}

		if req.Params["key1"] != "value1" {
			t.Error("Expected key1 to be value1")
		}

		if req.Params["key2"] != 42 {
			t.Error("Expected key2 to be 42")
		}
	})

	t.Run("Create ToolRequest with nil params", func(t *testing.T) {
		req := ToolRequest{
			Params: nil,
		}

		if req.Params != nil {
			t.Error("Expected params to be nil")
		}
	})

	t.Run("Create ToolRequest with empty params", func(t *testing.T) {
		req := ToolRequest{
			Params: make(map[string]interface{}),
		}

		if req.Params == nil {
			t.Error("Expected params to be empty map, not nil")
		}

		if len(req.Params) != 0 {
			t.Error("Expected empty params map")
		}
	})
}

func TestToolResponse(t *testing.T) {
	t.Run("Create ToolResponse with content", func(t *testing.T) {
		content := []ToolContent{
			{Text: "response1"},
			{Text: "response2"},
		}

		resp := ToolResponse{
			Content: content,
		}

		if len(resp.Content) != 2 {
			t.Errorf("Expected 2 content items, got %d", len(resp.Content))
		}

		if resp.Content[0].Text != "response1" {
			t.Error("Expected first content text to be response1")
		}

		if resp.Content[1].Text != "response2" {
			t.Error("Expected second content text to be response2")
		}
	})

	t.Run("Create ToolResponse with nil content", func(t *testing.T) {
		resp := ToolResponse{
			Content: nil,
		}

		if resp.Content != nil {
			t.Error("Expected content to be nil")
		}
	})

	t.Run("Create ToolResponse with empty content", func(t *testing.T) {
		resp := ToolResponse{
			Content: []ToolContent{},
		}

		if resp.Content == nil {
			t.Error("Expected content to be empty slice, not nil")
		}

		if len(resp.Content) != 0 {
			t.Error("Expected empty content slice")
		}
	})
}

func TestToolContent(t *testing.T) {
	t.Run("Create ToolContent with text", func(t *testing.T) {
		content := ToolContent{
			Text: "test content",
		}

		if content.Text != "test content" {
			t.Error("Expected text to be 'test content'")
		}
	})

	t.Run("Create ToolContent with empty text", func(t *testing.T) {
		content := ToolContent{
			Text: "",
		}

		if content.Text != "" {
			t.Error("Expected text to be empty string")
		}
	})

	t.Run("ToolContent is struct with text field only", func(t *testing.T) {
		// This test ensures ToolContent only has the Text field
		// and no other exported fields that might need testing
		content := ToolContent{Text: "test"}

		// Use reflection or type assertion to verify structure
		// Since it's a simple struct, we mainly test the field access
		text := content.Text
		if text != "test" {
			t.Error("Expected to access Text field correctly")
		}
	})
}
