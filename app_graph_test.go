package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	scip "github.com/sourcegraph/scip/bindings/go/scip"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func TestGetGraphData_Pruning(t *testing.T) {
	app := NewApp()
	defer app.Shutdown(context.Background())
	tempDir := t.TempDir()
	scipPath := filepath.Join(tempDir, "test.scip")

	// Create a mock SCIP index
	index := &scip.Index{
		Documents: []*scip.Document{
			{
				RelativePath: "pkg/main.go",
				Symbols: []*scip.SymbolInformation{
					{Symbol: "scip-go . . . pkg/Main#"},
					{Symbol: "scip-go . . . pkg/main()."},
				},
				Occurrences: []*scip.Occurrence{
					{
						Symbol:      "scip-go . . . pkg/Main#",
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{0, 0, 0, 4},
					},
					{
						Symbol:      "scip-go . . . pkg/main().",
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{5, 0, 5, 4},
					},
					{
						Symbol:      "scip-go . . . pkg/main().local1",
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{6, 0, 6, 4},
					},
					{
						Symbol:      "scip-go . . . pkg/main().$anon1", // Anonymous
						SymbolRoles: int32(scip.SymbolRole_Definition),
						Range:       []int32{7, 0, 7, 4},
					},
					// Reference to stdlib
					{
						Symbol: "go/fmt/Println().",
						Range:  []int32{8, 0, 8, 4},
					},
				},
			},
		},
	}

	data, err := proto.Marshal(index)
	require.NoError(t, err)
	err = os.WriteFile(scipPath, data, 0644)
	require.NoError(t, err)

	// Test 1: Full load without scope
	graph, err := app.GetGraphData(scipPath, "")
	require.NoError(t, err)

	// Verify pruning
	for _, el := range graph.Elements {
		if el.Group == "nodes" {
			node, ok := el.Data.(GraphNode)
			if ok {
				assert.NotContains(t, node.ID, "local", "Local variables should be pruned")
				assert.NotContains(t, node.ID, "$anon", "Anonymous closures should be pruned")
			}
		}
		if el.Group == "edges" {
			data := el.Data.(map[string]interface{})
			assert.NotContains(t, data["target"].(string), "go/fmt", "StdLib edges should be pruned")
		}
	}

	// Test 2: Hierarchical filter (Wrong scope)
	graphScope, err := app.GetGraphData(scipPath, "otherpkg")
	require.NoError(t, err)
	
	symbolNodeCount := 0
	for _, el := range graphScope.Elements {
		if el.Group == "nodes" {
			node := el.Data.(GraphNode)
			if node.Kind != "Folder" {
				symbolNodeCount++
			}
		}
	}
	assert.Equal(t, 0, symbolNodeCount, "Should have no symbols outside scope")

	// Test 3: Hierarchical filter (Correct scope)
	graphInScope, err := app.GetGraphData(scipPath, "pkg")
	require.NoError(t, err)
	
	foundMain := false
	for _, el := range graphInScope.Elements {
		if el.Group == "nodes" {
			node := el.Data.(GraphNode)
			if node.Name == "main" {
				foundMain = true
			}
		}
	}
	assert.True(t, foundMain, "Should find 'main' symbol in 'pkg' scope")
}
