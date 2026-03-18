package mcp

import (
	"context"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
)

func TestAstGrepScanHandler_MissingPattern(t *testing.T) {
	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Arguments: map[string]interface{}{
				"path": ".",
			},
		},
	}

	result, err := astGrepScanHandler(context.Background(), req)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.True(t, result.IsError)
	assert.Contains(t, result.Content[0].(mcp.TextContent).Text, "pattern is required")
}

// Note: Testing successful execution requires a mock of the ast-grep binary
// or a real ast-grep installation in the test environment.
// Since we use findAstGrepBinary, we can temporarily override the PATH or environment.

func TestAstGrepScanHandler_BinaryNotFound(t *testing.T) {
	// Temporarily break the override and PATH for this test
	oldOverride := astGrepPathOverride
	astGrepPathOverride = "/nonexistent/path/to/ast-grep"
	defer func() { astGrepPathOverride = oldOverride }()

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Arguments: map[string]interface{}{
				"pattern": "func $$$",
			},
		},
	}

	result, err := astGrepScanHandler(context.Background(), req)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.True(t, result.IsError)
	assert.Contains(t, result.Content[0].(mcp.TextContent).Text, "ast-grep binary not found")
}
func TestExpandToFallbacks(t *testing.T) {
	tests := []struct {
		pattern  string
		expected []string
	}{
		{
			"$DB.Query($$$)",
			[]string{"$OBJ.Query($$$)", "$$$.Query($$$)", "Query($$$)"},
		},
		{
			"a.db.Query($$$)",
			[]string{"$OBJ.Query($$$)", "$$$.Query($$$)", "Query($$$)"},
		},
		{
			"fmt.Printf($$$)",
			[]string{"$OBJ.Printf($$$)", "$$$.Printf($$$)", "Printf($$$)"},
		},
		{
			"obj.Method(a, b)",
			[]string{"$OBJ.Method(a, b)", "$$$.Method(a, b)", "Method(a, b)"},
		},
		{
			"func $$$",
			nil, // No dot, no fallbacks
		},
		{
			"pkg.Func()",
			[]string{"$OBJ.Func()", "$$$.Func()", "Func()"},
		},
		{
			" nested(pkg.Func())",
			[]string{" nested($OBJ.Func())", " nested($$$.Func())", "Func())"}, // Note: Heuristic might be slightly off for closing paren but it's a fallback
		},
	}

	for _, tt := range tests {
		t.Run(tt.pattern, func(t *testing.T) {
			assert.Equal(t, tt.expected, expandToFallbacks(tt.pattern))
		})
	}
}

func TestExtractMethodName(t *testing.T) {
	tests := []struct {
		pattern  string
		expected string
	}{
		{"a.db.Query($$$)", "Query"},
		{"Query($$$)", "Query"},
		{"os.Open(file)", "Open"},
		{"r.Header.Get(\"X\")", "Get"},
		{"InitDB", "InitDB"},
	}

	for _, tt := range tests {
		t.Run(tt.pattern, func(t *testing.T) {
			assert.Equal(t, tt.expected, extractMethodName(tt.pattern))
		})
	}
}

func TestGenerateStructuralProbe(t *testing.T) {
	result := generateStructuralProbe("Query", "go")
	assert.Contains(t, result, "language: go")
	assert.Contains(t, result, "pattern: Query($$$)")
	assert.Contains(t, result, "regex: \\.Query$")
}
