package mcp

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"gopkg.in/yaml.v3"
)

var sgBinaryData []byte

// SgConfig represents the structure of sgconfig.yml
type SgConfig struct {
	RuleDirs []string `yaml:"ruleDirs"`
}

// Start initializes and starts the MCP server.
func Start(sgBinary []byte) {
	sgBinaryData = sgBinary

	// Create a new MCP server
	s := server.NewMCPServer(
		"context-sherpa 🚀",
		"1.0.0",
		server.WithToolCapabilities(false),
	)

	// Add scan_code tool
	scanCodeTool := mcp.NewTool("scan_code",
		mcp.WithDescription("Scan code for linting violations using ast-grep"),
		mcp.WithString("code",
			mcp.Required(),
			mcp.Description("Source code to scan"),
		),
		mcp.WithString("language",
			mcp.Required(),
			mcp.Description("Programming language of the code"),
		),
		mcp.WithString("sgconfig",
			mcp.Description("Path to sgconfig file (default: sgconfig.yml)"),
		),
	)

	// Add add_or_update_rule tool
	addOrUpdateRuleTool := mcp.NewTool("add_or_update_rule",
		mcp.WithDescription(`Create or update an ast-grep rule for pattern-based code analysis.
ast-grep uses abstract syntax trees to find specific code patterns, making it more accurate than text-based tools.

Use this when you want to:
- Detect specific code patterns or anti-patterns
- Enforce coding standards and best practices
- Find security vulnerabilities (SQL injection, etc.)
- Catch maintenance issues or code smells
- Analyze code quality and consistency

Example: "Create a rule to catch SQL injection" → generates ast-grep YAML rules`),
		mcp.WithString("rule_id",
			mcp.Required(),
			mcp.Description("Unique identifier for the rule (e.g., 'no-sql-injection', 'require-tests', 'no-todo-comments')"),
		),
		mcp.WithString("rule_yaml",
			mcp.Required(),
			mcp.Description(`Complete YAML rule definition. Use this format:
id: your-rule-name
language: go
rule:
	 pattern: your-pattern-here
message: "Clear description of the issue"
severity: error|warning

Example for catching fmt.Sprintf in database calls:
id: no-sprintf-db
language: go
rule:
	 pattern: $DB.Exec(ctx, fmt.Sprintf($$$))
message: "Use parameterized queries"
severity: error`),
		),
	)

	// Add remove_rule tool
	removeRuleTool := mcp.NewTool("remove_rule",
		mcp.WithDescription("Remove an ast-grep rule"),
		mcp.WithString("rule_id",
			mcp.Required(),
			mcp.Description("Unique identifier of the rule to remove"),
		),
	)

	// Add initialize_ast_grep tool
	initializeAstGrepTool := mcp.NewTool("initialize_ast_grep",
		mcp.WithDescription("Initialize an ast-grep project by creating sgconfig.yml and rules directory"),
	)

	// Add tool handlers
	s.AddTool(scanCodeTool, scanCodeHandler)
	s.AddTool(addOrUpdateRuleTool, addOrUpdateRuleHandler)
	s.AddTool(removeRuleTool, removeRuleHandler)
	s.AddTool(initializeAstGrepTool, initializeAstGrepHandler)

	log.Println("Starting MCP server...")

	// Start the stdio server
	if err := server.ServeStdio(s); err != nil {
		log.Printf("Server error: %v\n", err)
	}
}

func scanCodeHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	code, err := req.RequireString("code")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	language, err := req.RequireString("language")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	sgconfigStr := "sgconfig.yml" // Default value
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if sgconfig, ok := args["sgconfig"].(string); ok && sgconfig != "" {
			sgconfigStr = sgconfig
		}
	}

	// --- DEBUG LOGGING ---
	log.Printf("Using sgconfig file: %s", sgconfigStr)
	// --- END DEBUG LOGGING ---

	// Check if the configuration file exists
	if _, err := os.Stat(sgconfigStr); os.IsNotExist(err) {
		return mcp.NewToolResultText(fmt.Sprintf("Error: Configuration file '%s' not found. Please run the 'initialize_ast_grep' tool first to set up the project.", sgconfigStr)), nil
	}

	tmpfile, err := os.CreateTemp("", "ast-grep-scan.*."+language)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error creating temporary file: %v", err)), nil
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte(code)); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error writing to temporary file: %v", err)), nil
	}

	if err := tmpfile.Close(); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error closing temporary file: %v", err)), nil
	}

	sgPath, err := extractSgBinary(sgBinaryData)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error extracting sg binary: %v", err)), nil
	}
	defer os.Remove(sgPath)

	// Find the project root where sgconfig.yml is located
	projectRoot, err := findProjectRoot()
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	cmd := exec.Command(sgPath, "scan", "--config", sgconfigStr, tmpfile.Name(), "--json")
	cmd.Dir = projectRoot // Run ast-grep from the project root
	output, err := cmd.CombinedOutput()
	if err != nil {
		// ast-grep exits with non-zero status code if issues are found.
		// We still want to parse the output.
	}

	return mcp.NewToolResultText(string(output)), nil
}

func addOrUpdateRuleHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	ruleID, err := req.RequireString("rule_id")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	ruleYAML, err := req.RequireString("rule_yaml")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Get the rule directory from sgconfig.yml
	ruleDir, err := getRuleDir()
	if err != nil {
		// If sgconfig.yml doesn't exist, suggest using the initialize tool
		if strings.Contains(err.Error(), "sgconfig.yml not found") {
			return mcp.NewToolResultText(fmt.Sprintf("Error: %s. Please run the 'initialize_ast_grep' tool first to set up the project.", err.Error())), nil
		}
		return mcp.NewToolResultError(err.Error()), nil
	}

	ruleFile := fmt.Sprintf("%s/%s.yml", ruleDir, ruleID)
	if err := os.WriteFile(ruleFile, []byte(ruleYAML), 0644); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error writing rule file: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Rule '%s' was added or updated successfully.", ruleID)), nil
}

// findProjectRoot finds the project root by searching for sgconfig.yml
// in the current and parent directories.
func findProjectRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("could not get current directory: %v", err)
	}

	for {
		configPath := dir + "/sgconfig.yml"
		if _, err := os.Stat(configPath); err == nil {
			return dir, nil
		}

		// Move to parent directory
		parentDir := filepath.Dir(dir)
		if parentDir == dir {
			break // Reached root
		}
		dir = parentDir
	}

	return "", fmt.Errorf("sgconfig.yml not found. Please run 'ast-grep new' to initialize an ast-grep project first")
}

func extractSgBinary(sgBinary []byte) (string, error) {
	tmpfile, err := os.CreateTemp("", "sg")
	if err != nil {
		return "", err
	}
	defer tmpfile.Close()

	if _, err := tmpfile.Write(sgBinary); err != nil {
		return "", err
	}

	if err := tmpfile.Chmod(0755); err != nil {
		return "", err
	}

	return tmpfile.Name(), nil
}

// getRuleDir determines the directory where rules should be stored by searching
// for sgconfig.yml in the current and parent directories.
func getRuleDir() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("could not get current directory: %v", err)
	}

	for {
		configPath := dir + "/sgconfig.yml"
		if _, err := os.Stat(configPath); err == nil {
			// Read and parse sgconfig.yml
			data, err := os.ReadFile(configPath)
			if err != nil {
				return "", fmt.Errorf("error reading sgconfig.yml: %v", err)
			}

			var config SgConfig
			if err := yaml.Unmarshal(data, &config); err != nil {
				return "", fmt.Errorf("error parsing sgconfig.yml: %v", err)
			}

			if len(config.RuleDirs) == 0 {
				return "", fmt.Errorf("ruleDirs not specified in sgconfig.yml")
			}
			// Return the first rule directory, relative to the config file's location
			return dir + "/" + strings.TrimSpace(config.RuleDirs[0]), nil
		}

		// Move to parent directory
		parentDir := filepath.Dir(dir)
		if parentDir == dir {
			break // Reached root
		}
		dir = parentDir
	}

	return "", fmt.Errorf("sgconfig.yml not found. Please run 'ast-grep new' to initialize an ast-grep project first")
}

func removeRuleHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	ruleID, err := req.RequireString("rule_id")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Get the rule directory from sgconfig.yml
	ruleDir, err := getRuleDir()
	if err != nil {
		// If sgconfig.yml doesn't exist, suggest using the initialize tool
		if strings.Contains(err.Error(), "sgconfig.yml not found") {
			return mcp.NewToolResultText(fmt.Sprintf("Error: %s. Please run the 'initialize_ast_grep' tool first to set up the project.", err.Error())), nil
		}
		return mcp.NewToolResultError(err.Error()), nil
	}

	ruleFile := fmt.Sprintf("%s/%s.yml", ruleDir, ruleID)
	if err := os.Remove(ruleFile); err != nil {
		if os.IsNotExist(err) {
			return mcp.NewToolResultText(fmt.Sprintf("Rule '%s' not found.", ruleID)), nil
		}
		return mcp.NewToolResultError(fmt.Sprintf("Error removing rule file: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Rule '%s' was removed successfully.", ruleID)), nil
}

func initializeAstGrepHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// Get the current working directory as the project root
	projectRoot, err := os.Getwd()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error getting current directory: %v", err)), nil
	}

	// Create the rules directory if it doesn't exist
	rulesDir := filepath.Join(projectRoot, "rules")
	if err := os.MkdirAll(rulesDir, 0755); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error creating rules directory: %v", err)), nil
	}

	// Create a basic sgconfig.yml file
	sgconfigPath := filepath.Join(projectRoot, "sgconfig.yml")
	sgconfigContent := `ruleDirs:
  - rules
`
	if err := os.WriteFile(sgconfigPath, []byte(sgconfigContent), 0644); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error creating sgconfig.yml: %v", err)), nil
	}

	return mcp.NewToolResultText("ast-grep project initialized successfully. Created sgconfig.yml and rules/ directory."), nil
}
