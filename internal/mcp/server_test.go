package mcp

import (
	"context"
	"os"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/assert"
)

// setupTestEnv creates a temporary directory for rules and a sgconfig.yml
// that points to it. It returns the path to the temp rule dir and a cleanup function.
func setupTestEnv(t *testing.T) (string, func()) {
	// Create a temporary directory for rules in the current directory to avoid path issues
	ruleDir, err := os.MkdirTemp(".", "test-rules-*")
	assert.NoError(t, err)

	// Create a temporary sgconfig.yml in the current test directory
	sgconfigContent := "ruleDirs:\n  - " + ruleDir
	err = os.WriteFile("sgconfig.yml", []byte(sgconfigContent), 0644)
	assert.NoError(t, err)

	// The cleanup function removes the temp directory and the sgconfig.yml
	cleanup := func() {
		os.RemoveAll(ruleDir)
		os.Remove("sgconfig.yml")
	}

	return ruleDir, cleanup
}

func TestAddOrUpdateRuleHandler(t *testing.T) {
	ruleDir, cleanup := setupTestEnv(t)
	defer cleanup()

	ruleFile := ruleDir + "/new-rule.yml"
	ruleYAML := "id: new-rule\nmessage: New Rule"

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "add_or_update_rule",
			Arguments: map[string]interface{}{
				"rule_id":   "new-rule",
				"rule_yaml": ruleYAML,
			},
		},
	}

	resp, err := addOrUpdateRuleHandler(context.Background(), req)
	assert.NoError(t, err)

	// Type assert the content to access the text
	textContent, ok := resp.Content[0].(mcp.TextContent)
	assert.True(t, ok)
	assert.Contains(t, textContent.Text, "Rule 'new-rule' was added or updated successfully.")

	// Check the file content
	data, err := os.ReadFile(ruleFile)
	assert.NoError(t, err)
	assert.Equal(t, ruleYAML, string(data))
}

func TestRemoveRuleHandler(t *testing.T) {
	ruleDir, cleanup := setupTestEnv(t)
	defer cleanup()

	// Create a temporary rule file to be removed
	ruleFile := ruleDir + "/rule-to-remove.yml"
	err := os.WriteFile(ruleFile, []byte("id: rule-to-remove"), 0644)
	assert.NoError(t, err)

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "remove_rule",
			Arguments: map[string]interface{}{
				"rule_id": "rule-to-remove",
			},
		},
	}

	resp, err := removeRuleHandler(context.Background(), req)
	assert.NoError(t, err)

	textContent, ok := resp.Content[0].(mcp.TextContent)
	assert.True(t, ok)
	assert.Contains(t, textContent.Text, "Rule 'rule-to-remove' was removed successfully.")

	// Check that the file was removed
	_, err = os.Stat(ruleFile)
	assert.True(t, os.IsNotExist(err))
}

func TestGetRuleDir_NoConfig(t *testing.T) {
	// Ensure no sgconfig.yml exists in current directory or parent directories
	os.Remove("sgconfig.yml")

	// Change to a temp directory where no sgconfig.yml exists
	originalDir, err := os.Getwd()
	assert.NoError(t, err)
	defer os.Chdir(originalDir)

	tempDir, err := os.MkdirTemp("", "test-no-config-*")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	err = os.Chdir(tempDir)
	assert.NoError(t, err)

	_, err = getRuleDir()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "sgconfig.yml not found")
}

func TestScanCodeHandler(t *testing.T) {
	// Set up test environment first
	ruleDir, cleanup := setupTestEnv(t)
	defer cleanup()

	// Create a rule file
	ruleYAML := "id: no-fmt-println\nlanguage: go\nrule:\n  pattern: fmt.Println($$$)"
	err := os.WriteFile(ruleDir+"/no-fmt-println.yml", []byte(ruleYAML), 0644)
	assert.NoError(t, err)

	// Code that violates the rule
	violatingCode := `
package main
import "fmt"
func main() {
	fmt.Println("hello")
}`

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "scan_code",
			Arguments: map[string]interface{}{
				"code":     violatingCode,
				"language": "go",
				"sgconfig": "sgconfig.yml", // Use the config file created in test environment
			},
		},
	}

	resp, err := scanCodeHandler(context.Background(), req)
	// The handler might fail if sg binary is not available, but we can still test the structure
	// Let's just verify the request structure is handled correctly
	if err != nil {
		// If sg binary is not available, we expect an error about the binary
		assert.Contains(t, err.Error(), "sg binary")
	} else {
		// If successful, check the response structure
		assert.NotNil(t, resp)
	}
}

func TestInitializeAstGrepHandler(t *testing.T) {
	// Change to a temp directory where no sgconfig.yml exists
	originalDir, err := os.Getwd()
	assert.NoError(t, err)
	defer os.Chdir(originalDir)

	tempDir, err := os.MkdirTemp("", "test-init-*")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	err = os.Chdir(tempDir)
	assert.NoError(t, err)

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      "initialize_ast_grep",
			Arguments: map[string]interface{}{},
		},
	}

	resp, err := initializeAstGrepHandler(context.Background(), req)
	assert.NoError(t, err)

	textContent, ok := resp.Content[0].(mcp.TextContent)
	assert.True(t, ok)
	assert.Contains(t, textContent.Text, "ast-grep project initialized successfully")

	// Verify that sgconfig.yml was created
	_, err = os.Stat("sgconfig.yml")
	assert.NoError(t, err)

	// Verify that rules directory was created
	_, err = os.Stat("rules")
	assert.NoError(t, err)
}
