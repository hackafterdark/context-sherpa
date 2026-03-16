package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiscoverMarkdownFiles_Exclusions(t *testing.T) {
	// 1. Setup temporary directory structure
	tmpDir, err := os.MkdirTemp("", "sherpa-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Directories to scan
	dirs := []string{
		".agents/my-skill",
		".git",
		".ssh",
		"node_modules",
		"some-feature",
	}

	for _, d := range dirs {
		err := os.MkdirAll(filepath.Join(tmpDir, d), 0755)
		require.NoError(t, err)
	}

	// Files to create
	files := map[string]string{
		".agents/my-skill/SKILL.md": "---\nname: My Custom Skill\ndescription: A very useful agent skill.\n---\n\nskill content",
		".git/config.md":            "should be skipped",
		".ssh/known_hosts.md":       "should be skipped",
		"node_modules/lib/doc.md":   "should be skipped",
		"some-feature/README.md":     "---\nname: Normal Readme\n---\n\nshould be found",
		"AGENTS.md":                  "should be found",
	}

	for path, content := range files {
		fullPath := filepath.Join(tmpDir, path)
		err := os.MkdirAll(filepath.Dir(fullPath), 0755)
		require.NoError(t, err)
		err = os.WriteFile(fullPath, []byte(content), 0644)
		require.NoError(t, err)
	}

	// 2. Run discovery
	app := &App{
		workspaces: []Workspace{{Root: tmpDir, IsManaged: true}},
	}
	
	// DiscoverMarkdownFiles uses isPathInWorkspace which needs absolute paths to match
	absTmpDir, _ := filepath.Abs(tmpDir)
	app.workspaces[0].Root = absTmpDir

	foundEntries, err := app.DiscoverMarkdownFiles(absTmpDir)
	require.NoError(t, err)

	// 3. Assertions
	foundMap := make(map[string]MarkdownEntry)
	for _, entry := range foundEntries {
		rel, _ := filepath.Rel(absTmpDir, entry.Path)
		foundMap[filepath.ToSlash(rel)] = entry
	}

	// Should find these
	require.Contains(t, foundMap, ".agents/my-skill/SKILL.md")
	assert.Equal(t, "My Custom Skill", foundMap[".agents/my-skill/SKILL.md"].FrontMatter["name"])
	assert.Equal(t, "A very useful agent skill.", foundMap[".agents/my-skill/SKILL.md"].FrontMatter["description"])

	require.Contains(t, foundMap, "some-feature/README.md")
	assert.Equal(t, "Normal Readme", foundMap["some-feature/README.md"].FrontMatter["name"])

	require.Contains(t, foundMap, "AGENTS.md")
	assert.Empty(t, foundMap["AGENTS.md"].FrontMatter)

	// Should NOT find these
	assert.NotContains(t, foundMap, ".git/config.md")
	assert.NotContains(t, foundMap, ".ssh/known_hosts.md")
	assert.NotContains(t, foundMap, "node_modules/lib/doc.md")
}

func TestNormalizePath(t *testing.T) {
	app := &App{}
	
	// Test on Windows specifically if running on Windows
	if runtime.GOOS == "windows" {
		tests := []struct {
			input    string
			expected string
		}{
			{"c:\\foo\\bar", "C:\\foo\\bar"},
			{"F:\\Context-Sherpa", "F:\\Context-Sherpa"},
			{"d:/lower/case", "D:\\lower\\case"},
		}
		
		for _, tt := range tests {
			abs, _ := filepath.Abs(tt.expected)
			normalized := app.normalizePath(tt.input)
			assert.Equal(t, abs, normalized)
		}
	} else {
		// Non-windows normalization
		path := "/usr/local/bin"
		normalized := app.normalizePath(path)
		assert.Equal(t, path, normalized)
	}
}

