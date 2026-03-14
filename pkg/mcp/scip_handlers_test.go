package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/sourcegraph/scip/bindings/go/scip"
	"google.golang.org/protobuf/proto"
)

// Helper to safely get text from mcp.CallToolResult
func safeGetText(res *mcp.CallToolResult) string {
	if res == nil || len(res.Content) == 0 {
		return ""
	}
	switch c := res.Content[0].(type) {
	case mcp.TextContent:
		return c.Text
		// Remove invalid case for map[string]interface{} since Content is an interface
	}
	// Fallback to json marshaling/unmarshaling if it's not a known type
	data, _ := json.Marshal(res.Content[0])
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err == nil {
		if text, ok := m["text"].(string); ok {
			return text
		}
	}
	return fmt.Sprintf("%v", res.Content[0])
}

// Helper to create a temporary SCIP index
func createTestSCIPIndex(t *testing.T, workspaceRoot string, index *scip.Index) {
	sherpaDir := filepath.Join(workspaceRoot, ".context-sherpa")
	if err := os.MkdirAll(sherpaDir, 0755); err != nil {
		t.Fatalf("Failed to create .context-sherpa dir: %v", err)
	}

	data, err := proto.Marshal(index)
	if err != nil {
		t.Fatalf("Failed to marshal SCIP index: %v", err)
	}

	scipPath := filepath.Join(sherpaDir, "index.scip")
	if err := os.WriteFile(scipPath, data, 0644); err != nil {
		t.Fatalf("Failed to write index.scip: %v", err)
	}

	// Also create sgconfig.yml to mark this as a workspace root
	if err := os.WriteFile(filepath.Join(workspaceRoot, "sgconfig.yml"), []byte("{}"), 0644); err != nil {
		t.Fatalf("Failed to write sgconfig.yml: %v", err)
	}
}

func TestListSymbolsInFileHandler(t *testing.T) {
	// Reset global override for tests
	workspaceRootOverride = ""

	tempDir, err := os.MkdirTemp("", "scip-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Mock SCIP Index
	index := &scip.Index{
		Documents: []*scip.Document{
			{
				RelativePath: "main.go",
				Occurrences: []*scip.Occurrence{
					{
						Symbol:      "scip-go gomod github.com/user/project `main`#",
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{10, 5, 9}, // line 11, col 6-10
					},
					{
						Symbol:      "scip-go gomod github.com/user/project `Config`#",
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{20, 5, 11},
					},
					{
						Symbol:      "fmt/Println().",
						SymbolRoles: int32(scip.SymbolRole_ReadAccess), // Reference
						Range:       []int32{15, 5, 12},
					},
				},
			},
		},
	}
	createTestSCIPIndex(t, tempDir, index)

	// Test Case 1: Existing file
	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "list_symbols_in_file",
			Arguments: map[string]interface{}{
				"file_path":     "main.go",
				"workspaceRoot": tempDir,
			},
		},
	}

	res, err := listSymbolsInFileHandler(context.Background(), req)
	if err != nil {
		t.Fatalf("Handler failed: %v", err)
	}

	if len(res.Content) != 1 {
		t.Fatalf("Expected 1 content block, got %d", len(res.Content))
	}

	responseText := safeGetText(res)
	t.Logf("Response text: %s", responseText)

	var symbols []map[string]interface{}
	if err := json.Unmarshal([]byte(responseText), &symbols); err != nil {
		t.Fatalf("Failed to unmarshal result: %v. Response was: %s", err, responseText)
	}

	// Should only contain 2 symbols (definitions)
	if len(symbols) != 2 {
		t.Errorf("Expected 2 symbols, got %d", len(symbols))
	}

	// Test Case 2: Non-existent file
	req.Params.Arguments = map[string]interface{}{
		"file_path":     "missing.go",
		"workspaceRoot": tempDir,
	}
	res, err = listSymbolsInFileHandler(context.Background(), req)
	if err != nil {
		t.Fatalf("Handler failed: %v", err)
	}
	responseText = safeGetText(res)
	if responseText != "No symbols found in file: missing.go" {
		t.Errorf("Expected 'No symbols found', got %s", responseText)
	}
}

func TestSearchDefinitionsHandler(t *testing.T) {
	// Reset global override for tests
	workspaceRootOverride = ""

	tempDir, err := os.MkdirTemp("", "scip-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Mock SCIP Index
	index := &scip.Index{
		Documents: []*scip.Document{
			{
				RelativePath: "pkg/api.go",
				Occurrences: []*scip.Occurrence{
					{
						Symbol:      "scip-go gomod github.com/user/project `Service`#",
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{10, 5, 12},
					},
				},
			},
			{
				RelativePath: "main.go",
				Occurrences: []*scip.Occurrence{
					{
						Symbol:      "scip-go gomod github.com/user/project `Service`#",
						SymbolRoles: int32(scip.SymbolRole_ReadAccess), // Usage
						Range:       []int32{20, 5, 12},
					},
				},
			},
		},
	}
	createTestSCIPIndex(t, tempDir, index)

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "search_definitions",
			Arguments: map[string]interface{}{
				"query":         "Service",
				"workspaceRoot": tempDir,
			},
		},
	}

	res, err := searchDefinitionsHandler(context.Background(), req)
	if err != nil {
		t.Fatalf("Handler failed: %v", err)
	}

	responseText := safeGetText(res)
	t.Logf("Response text: %s", responseText)

	var results []map[string]interface{}
	if err := json.Unmarshal([]byte(responseText), &results); err != nil {
		t.Fatalf("Failed to unmarshal result: %v. Response was: %s", err, responseText)
	}

	// Should only find 1 definition in pkg/api.go
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	} else if results[0]["file"] != "pkg/api.go" {
		t.Errorf("Expected pkg/api.go, got %s", results[0]["file"])
	}
}

func TestGetSymbolMapHandler(t *testing.T) {
	// Reset global override for tests
	workspaceRootOverride = ""

	tempDir, err := os.MkdirTemp("", "scip-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Mock SCIP Index
	index := &scip.Index{
		Documents: []*scip.Document{
			{
				RelativePath: "lib.go",
				Occurrences: []*scip.Occurrence{
					{
						Symbol:      "scip-go gomod github.com/user/project `Auth`#",
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{5, 5, 9},
					},
				},
			},
			{
				RelativePath: "main.go",
				Occurrences: []*scip.Occurrence{
					{
						Symbol:      "scip-go gomod github.com/user/project `Auth`#",
						SymbolRoles: int32(scip.SymbolRole_ReadAccess),
						Range:       []int32{100, 1, 5},
					},
				},
			},
		},
	}
	createTestSCIPIndex(t, tempDir, index)

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "get_symbol_map",
			Arguments: map[string]interface{}{
				"symbolName":    "Auth",
				"workspaceRoot": tempDir,
			},
		},
	}

	res, err := getSymbolMapHandler(context.Background(), req)
	if err != nil {
		t.Fatalf("Handler failed: %v", err)
	}

	responseText := safeGetText(res)
	t.Logf("Response text: %s", responseText)

	var symbolMap map[string]interface{}
	if err := json.Unmarshal([]byte(responseText), &symbolMap); err != nil {
		t.Fatalf("Failed to unmarshal result: %v. Response was: %s", err, responseText)
	}

	def := symbolMap["definition"]
	refs := symbolMap["references"].([]interface{})

	if def == nil {
		t.Errorf("Expected definition, got nil")
	}
	if len(refs) != 1 {
		t.Errorf("Expected 1 reference, got %d", len(refs))
	}
}

func TestInitializeScipHandler(t *testing.T) {
	// Reset global override for tests
	workspaceRootOverride = ""

	tempDir, err := os.MkdirTemp("", "scip-init-test-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	// Mock go.mod
	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte("module test"), 0644); err != nil {
		t.Fatal(err)
	}
	// Mock sgconfig.yml
	if err := os.WriteFile(filepath.Join(tempDir, "sgconfig.yml"), []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "initialize_scip",
			Arguments: map[string]interface{}{
				"workspaceRoot": tempDir,
				"language":      "go",
			},
		},
	}

	// Indexing will likely fail because scip-go is not installed in the test environment,
	// but we check if it reaches the indexing logic or fails with a recognizable error.
	res, err := initializeScipHandler(context.Background(), req)
	if err != nil {
		t.Fatalf("Handler failed: %v", err)
	}

	text := safeGetText(res)
	t.Logf("Response text: %s", text)

	if !strings.Contains(text, "Workspace indexed successfully") &&
		!strings.Contains(text, "Indexer for go not found") &&
		!strings.Contains(text, "Indexing failed") &&
		!strings.Contains(text, "indexer for go not found") {
		t.Errorf("Unexpected result text: %s", text)
	}
}
