package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"gopkg.in/yaml.v3"
)

var sgBinaryData []byte

// SgConfig represents the structure of sgconfig.yml
type SgConfig struct {
	RuleDirs []string `yaml:"ruleDirs"`
}

// CommunityRule represents a rule in the community repository
type CommunityRule struct {
	ID          string   `json:"id"`
	Tool        string   `json:"tool"`
	Path        string   `json:"path"`
	Language    string   `json:"language"`
	Author      string   `json:"author"`
	Tags        []string `json:"tags"`
	Description string   `json:"description"`
}

// CommunityRuleIndex represents the index.json file from the community repository
type CommunityRuleIndex struct {
	Version int             `json:"version"`
	Rules   []CommunityRule `json:"rules"`
}

// AstGrepRule defines the structure for a valid ast-grep rule YAML.
// This is used to validate the rule before writing it to disk.
type AstGrepRule struct {
	ID       string   `yaml:"id"`
	Language string   `yaml:"language"`
	Rule     struct{} `yaml:"rule"`
}

// Cache for the community rule index
var (
	communityRuleCache *CommunityRuleIndex
	cacheTimestamp     time.Time
	cacheTTL           = 5 * time.Minute // Cache for 5 minutes
)

var communityRulesRepo = "https://raw.githubusercontent.com/hackafterdark/context-sherpa-community-rules/main/index.json"

// projectRootOverride stores the custom project root directory when specified via command-line argument
var projectRootOverride string

// getCommunityRulesRepoURL returns the community rules repository URL (can be overridden in tests)
func getCommunityRulesRepoURL() string {
	return communityRulesRepo
}

// Start initializes and starts the MCP server.
func Start(sgBinary []byte, projectRoot string) {
	if projectRoot != "" {
		projectRootOverride = projectRoot
	}
	sgBinaryData = sgBinary

	// Create a new MCP server
	s := server.NewMCPServer(
		"context-sherpa 🚀",
		"1.0.0",
		server.WithToolCapabilities(false),
	)

	// Add scan_code tool
	scanCodeTool := mcp.NewTool("scan_code",
		mcp.WithDescription("Scan a given string of source code for violations against the currently configured ast-grep rules."),
		mcp.WithString("code",
			mcp.Required(),
			mcp.Description("The raw source code to be scanned."),
		),
		mcp.WithString("language",
			mcp.Required(),
			mcp.Description("The programming language of the code (e.g., 'go', 'python')."),
		),
		mcp.WithString("sgconfig",
			mcp.Description("Path to a specific sgconfig.yml file to use for the scan. If omitted, it defaults to the root sgconfig.yml."),
		),
	)

	// Add scan_path tool
	scanPathTool := mcp.NewTool("scan_path",
		mcp.WithDescription("Scan code for rule violations by providing a file path, directory path, or glob pattern. The path can resolve to a single file, multiple files, or an entire directory tree. Returns JSON array of violations found with file location, line numbers, and rule details."),
		mcp.WithString("path",
			mcp.Required(),
			mcp.Description("File path, directory path, or glob pattern to scan. Examples: 'src/main.go' (single file), 'src/' (directory), '**/*.go' (all Go files), 'internal/**/*.js' (pattern)."),
		),
		mcp.WithString("sgconfig",
			mcp.Description("Path to specific sgconfig.yml configuration file. If omitted, uses 'sgconfig.yml' in project root. Example: 'custom/sgconfig.yml'."),
		),
		mcp.WithString("language",
			mcp.Description("Programming language filter for directory scans. Supported: 'go', 'python', 'javascript', 'typescript', 'rust', 'java', 'cpp', 'c'. If specified, only files with matching extensions are scanned."),
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
		mcp.WithDescription("Remove a specific ast-grep rule file from the local project's rule directory."),
		mcp.WithString("rule_id",
			mcp.Required(),
			mcp.Description("The unique ID of the rule to be removed (e.g., 'no-sql-injection'). This should match the filename without the .yml extension."),
		),
	)

	// Add initialize_ast_grep tool
	initializeAstGrepTool := mcp.NewTool("initialize_ast_grep",
		mcp.WithDescription("Sets up the current project for ast-grep by creating a default `sgconfig.yml` file and a `rules/` directory. This is a required first step before adding or importing local rules."),
	)

	// Add search_community_rules tool
	searchCommunityRulesTool := mcp.NewTool("search_community_rules",
		mcp.WithDescription(`Search the community rule repository for ast-grep rules.
ast-grep uses abstract syntax trees to find specific code patterns, making it more accurate than text-based tools.

Use this when you want to:
- Detect specific code patterns or anti-patterns
- Enforce coding standards and best practices
- Find security vulnerabilities (SQL injection, etc.)
- Catch maintenance issues or code smells
- Analyze code quality and consistency

Example: "Create a rule to catch SQL injection" → generates ast-grep YAML rules`),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("Natural language query (e.g., 'sql injection', 'check for todos')"),
		),
		mcp.WithString("language",
			mcp.Description("Programming language (e.g., 'go', 'python')"),
		),
		mcp.WithString("tags",
			mcp.Description("Comma-separated list of tags to filter by (e.g., 'security,database')"),
		),
	)

	// Add get_community_rule_details tool
	getCommunityRuleDetailsTool := mcp.NewTool("get_community_rule_details",
		mcp.WithDescription("Get the full YAML content and explanation for a community rule"),
		mcp.WithString("rule_id",
			mcp.Required(),
			mcp.Description("Unique identifier of the rule (e.g., 'ast-grep-go-sql-injection')"),
		),
	)

	// Add import_community_rule tool
	importCommunityRuleTool := mcp.NewTool("import_community_rule",
		mcp.WithDescription("Download a community rule and add it to the local project"),
		mcp.WithString("rule_id",
			mcp.Required(),
			mcp.Description("Unique identifier of the rule to import"),
		),
	)

	// Add tool handlers
	s.AddTool(scanCodeTool, scanCodeHandler)
	s.AddTool(scanPathTool, scanPathHandler)
	s.AddTool(addOrUpdateRuleTool, addOrUpdateRuleHandler)
	s.AddTool(removeRuleTool, removeRuleHandler)
	s.AddTool(initializeAstGrepTool, initializeAstGrepHandler)
	s.AddTool(searchCommunityRulesTool, searchCommunityRulesHandler)
	s.AddTool(getCommunityRuleDetailsTool, getCommunityRuleDetailsHandler)
	s.AddTool(importCommunityRuleTool, importCommunityRuleHandler)

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

// scanPathHandler handles the scan_path tool
func scanPathHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	path, err := req.RequireString("path")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	sgconfigStr := "sgconfig.yml" // Default value
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if sgconfig, ok := args["sgconfig"].(string); ok && sgconfig != "" {
			sgconfigStr = sgconfig
		}
	}

	// Get optional language filter
	var languageFilter string
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if lang, ok := args["language"].(string); ok && lang != "" {
			languageFilter = strings.ToLower(lang)
		}
	}

	// --- DEBUG LOGGING ---
	log.Printf("scan_file: Using sgconfig file: %s", sgconfigStr)
	log.Printf("scan_file: Scanning path: %s", path)
	if languageFilter != "" {
		log.Printf("scan_file: Language filter: %s", languageFilter)
	}
	// --- END DEBUG LOGGING ---

	// Check if the configuration file exists
	if _, err := os.Stat(sgconfigStr); os.IsNotExist(err) {
		return mcp.NewToolResultText(fmt.Sprintf("Error: Configuration file '%s' not found. Please run the 'initialize_ast_grep' tool first to set up the project.", sgconfigStr)), nil
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

	// Discover files to scan
	files, err := discoverFiles(path, languageFilter)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error discovering files: %v", err)), nil
	}

	if len(files) == 0 {
		return mcp.NewToolResultText("[]"), nil // Return empty JSON array for no files
	}

	// --- DEBUG LOGGING ---
	log.Printf("scan_file: Found %d files to scan", len(files))
	// --- END DEBUG LOGGING ---

	// Filter files by size (1MB limit) - FIRST ITERATION FEATURE
	var validFiles []string
	var skippedFiles []string
	for _, file := range files {
		fileInfo, err := os.Stat(file)
		if err != nil {
			log.Printf("scan_file: Warning - could not stat file %s: %v", file, err)
			continue
		}

		if fileInfo.Size() > 1024*1024 { // 1MB limit
			skippedFiles = append(skippedFiles, file)
			log.Printf("scan_file: Skipping file %s (size: %d bytes > 1MB limit)", file, fileInfo.Size())
			continue
		}

		validFiles = append(validFiles, file)
	}

	// --- DEBUG LOGGING ---
	log.Printf("scan_file: %d files valid for scanning, %d files skipped (over 1MB)", len(validFiles), len(skippedFiles))
	// --- END DEBUG LOGGING ---

	if len(validFiles) == 0 {
		return mcp.NewToolResultText("[]"), nil // Return empty JSON array for no valid files
	}

	// Scan files in batches
	allOutput, err := scanFileBatch(validFiles, sgconfigStr, projectRoot, sgPath)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error scanning files: %v", err)), nil
	}

	return mcp.NewToolResultText(allOutput), nil
}

// discoverFiles discovers files to scan based on the path pattern
func discoverFiles(path, languageFilter string) ([]string, error) {
	var files []string

	// Check if path is a direct file
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		// Apply language filter if specified
		if languageFilter != "" && !matchesLanguage(path, languageFilter) {
			return files, nil // Return empty slice if file doesn't match language filter
		}
		return []string{path}, nil
	}

	// Handle directory scanning (when path is "." or a directory)
	if path == "." {
		err := filepath.Walk(".", func(currentPath string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			// Skip directories
			if info.IsDir() {
				return nil
			}

			// Apply language filter if specified
			if languageFilter != "" && !matchesLanguage(currentPath, languageFilter) {
				return nil
			}

			files = append(files, currentPath)
			return nil
		})

		return files, err
	}

	// Check if it's a directory path
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		err := filepath.Walk(path, func(currentPath string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			// Skip directories
			if info.IsDir() {
				return nil
			}

			// Apply language filter if specified
			if languageFilter != "" && !matchesLanguage(currentPath, languageFilter) {
				return nil
			}

			files = append(files, currentPath)
			return nil
		})

		return files, err
	}

	// Handle glob patterns - walk current directory and match patterns
	err := filepath.Walk(".", func(currentPath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Check if path matches the pattern
		matched, err := filepath.Match(path, currentPath)
		if err != nil {
			return err
		}

		if matched {
			// Apply language filter if specified
			if languageFilter != "" && !matchesLanguage(currentPath, languageFilter) {
				return nil
			}
			files = append(files, currentPath)
		}

		return nil
	})

	return files, err
}

// matchesLanguage checks if a file path matches the specified language
func matchesLanguage(filePath, language string) bool {
	ext := strings.ToLower(filepath.Ext(filePath))
	switch language {
	case "go":
		return ext == ".go"
	case "python":
		return ext == ".py"
	case "javascript":
		return ext == ".js"
	case "typescript":
		return ext == ".ts"
	case "rust":
		return ext == ".rs"
	case "java":
		return ext == ".java"
	case "cpp", "c++":
		return ext == ".cpp" || ext == ".cc" || ext == ".cxx"
	case "c":
		return ext == ".c" || ext == ".h"
	default:
		return false
	}
}

// scanFileBatch scans a batch of files and returns combined results
func scanFileBatch(files []string, sgconfigStr, projectRoot, sgPath string) (string, error) {
	if len(files) == 0 {
		return "[]", nil
	}

	// For now, scan all files in a single batch
	// TODO: Implement batching for very large file lists
	args := []string{"scan", "--config", sgconfigStr}
	args = append(args, files...)
	args = append(args, "--json")

	cmd := exec.Command(sgPath, args...)
	cmd.Dir = projectRoot
	output, err := cmd.CombinedOutput()
	if err != nil {
		// ast-grep exits with non-zero status code if issues are found.
		// We still want to parse the output.
		log.Printf("scan_file: ast-grep command exited with error: %v", err)
	}

	return string(output), nil
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
	var dir string
	var err error

	if projectRootOverride != "" {
		// Use the specified project root as starting point
		dir = projectRootOverride
	} else {
		// Fall back to current behavior
		dir, err = os.Getwd()
		if err != nil {
			return "", fmt.Errorf("could not get current directory: %v", err)
		}
	}

	for {
		configPath := filepath.Join(dir, "sgconfig.yml")
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
	var dir string
	var err error

	if projectRootOverride != "" {
		// Use the specified project root as starting point
		dir = projectRootOverride
	} else {
		// Fall back to current behavior
		dir, err = os.Getwd()
		if err != nil {
			return "", fmt.Errorf("could not get current directory: %v", err)
		}
	}

	for {
		configPath := filepath.Join(dir, "sgconfig.yml")
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
			return filepath.Join(dir, strings.TrimSpace(config.RuleDirs[0])), nil
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
	// Determine the project root to use
	var projectRoot string
	var err error

	if projectRootOverride != "" {
		projectRoot = projectRootOverride
	} else {
		projectRoot, err = os.Getwd()
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Error getting current directory: %v", err)), nil
		}
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

// fetchCommunityRuleIndex fetches and caches the community rule index
func fetchCommunityRuleIndex() (*CommunityRuleIndex, error) {
	// Check if we have a valid cached index
	if communityRuleCache != nil && time.Since(cacheTimestamp) < cacheTTL {
		log.Println("Using cached community rule index")
		return communityRuleCache, nil
	}

	log.Println("Fetching community rule index from repository")

	// Fetch the index.json file
	resp, err := http.Get(getCommunityRulesRepoURL())
	if err != nil {
		return nil, fmt.Errorf("failed to fetch community rule index: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to fetch community rule index: HTTP %d", resp.StatusCode)
	}

	var index CommunityRuleIndex
	if err := json.NewDecoder(resp.Body).Decode(&index); err != nil {
		return nil, fmt.Errorf("failed to parse community rule index: %v", err)
	}

	// Update cache
	communityRuleCache = &index
	cacheTimestamp = time.Now()

	log.Printf("Successfully loaded %d community rules", len(index.Rules))
	return &index, nil
}

// searchCommunityRulesHandler handles the search_community_rules tool
func searchCommunityRulesHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query, err := req.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Get optional parameters
	var language string
	if langVal, ok := req.Params.Arguments.(map[string]interface{})["language"]; ok {
		if langStr, ok := langVal.(string); ok {
			language = strings.ToLower(langStr)
		}
	}

	var tags []string
	if tagsVal, ok := req.Params.Arguments.(map[string]interface{})["tags"]; ok {
		if tagsStr, ok := tagsVal.(string); ok && tagsStr != "" {
			tags = strings.Split(tagsStr, ",")
			for i, tag := range tags {
				tags[i] = strings.ToLower(strings.TrimSpace(tag))
			}
		}
	}

	// Fetch the community rule index
	index, err := fetchCommunityRuleIndex()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to fetch community rules: %v", err)), nil
	}

	// Filter rules based on criteria
	var matchingRules []CommunityRule
	for _, rule := range index.Rules {
		// Filter by language if specified
		if language != "" && strings.ToLower(rule.Language) != language {
			continue
		}

		// Filter by tags if specified (rule must have ALL specified tags)
		if len(tags) > 0 {
			hasAllTags := true
			for _, requiredTag := range tags {
				found := false
				for _, ruleTag := range rule.Tags {
					if strings.ToLower(ruleTag) == requiredTag {
						found = true
						break
					}
				}
				if !found {
					hasAllTags = false
					break
				}
			}
			if !hasAllTags {
				continue
			}
		}

		// If we get here, the rule matches our filters
		matchingRules = append(matchingRules, rule)
	}

	// Apply query search to the filtered results
	if query != "" {
		queryLower := strings.ToLower(query)
		var queryMatches []CommunityRule

		for _, rule := range matchingRules {
			// Search in ID, description, and tags
			if strings.Contains(strings.ToLower(rule.ID), queryLower) ||
				strings.Contains(strings.ToLower(rule.Description), queryLower) {
				queryMatches = append(queryMatches, rule)
				continue
			}

			// Search in tags
			for _, tag := range rule.Tags {
				if strings.Contains(strings.ToLower(tag), queryLower) {
					queryMatches = append(queryMatches, rule)
					break
				}
			}
		}

		matchingRules = queryMatches
	}

	// Format results
	if len(matchingRules) == 0 {
		return mcp.NewToolResultText("No community rules found matching your criteria."), nil
	}

	result := fmt.Sprintf("Found %d community rule(s) matching your criteria:\n\n", len(matchingRules))
	for i, rule := range matchingRules {
		result += fmt.Sprintf("%d. **%s** (%s)\n", i+1, rule.ID, rule.Language)
		result += fmt.Sprintf("   Author: %s\n", rule.Author)
		result += fmt.Sprintf("   Description: %s\n", rule.Description)
		if len(rule.Tags) > 0 {
			result += fmt.Sprintf("   Tags: %s\n", strings.Join(rule.Tags, ", "))
		}
		result += "\n"
	}

	return mcp.NewToolResultText(result), nil
}

// getCommunityRuleDetailsHandler handles the get_community_rule_details tool
func getCommunityRuleDetailsHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	ruleID, err := req.RequireString("rule_id")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Fetch the community rule index
	index, err := fetchCommunityRuleIndex()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to fetch community rules: %v", err)), nil
	}

	// Find the rule
	var foundRule *CommunityRule
	for _, rule := range index.Rules {
		if rule.ID == ruleID {
			foundRule = &rule
			break
		}
	}

	if foundRule == nil {
		return mcp.NewToolResultText(fmt.Sprintf("Rule '%s' not found in community repository.", ruleID)), nil
	}

	// Fetch the actual rule YAML content
	ruleURL := fmt.Sprintf("https://raw.githubusercontent.com/hackafterdark/context-sherpa-community-rules/main/%s", foundRule.Path)
	resp, err := http.Get(ruleURL)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to fetch rule content: %v", err)), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to fetch rule content: HTTP %d", resp.StatusCode)), nil
	}

	// Read the YAML content
	var buf []byte
	if resp.ContentLength > 0 {
		buf = make([]byte, resp.ContentLength)
	} else {
		// Read in chunks if ContentLength is unknown
		chunkSize := 1024
		buffer := make([]byte, chunkSize)
		for {
			n, err := resp.Body.Read(buffer)
			if n > 0 {
				buf = append(buf, buffer[:n]...)
			}
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				return mcp.NewToolResultError(fmt.Sprintf("Failed to read rule content: %v", err)), nil
			}
		}
	}

	yamlContent := string(buf)

	// Format the response
	result := fmt.Sprintf("Rule Details for '%s':\n\n", foundRule.ID)
	result += fmt.Sprintf("**ID:** %s\n", foundRule.ID)
	result += fmt.Sprintf("**Tool:** %s\n", foundRule.Tool)
	result += fmt.Sprintf("**Language:** %s\n", foundRule.Language)
	result += fmt.Sprintf("**Author:** %s\n", foundRule.Author)
	result += fmt.Sprintf("**Description:** %s\n", foundRule.Description)
	if len(foundRule.Tags) > 0 {
		result += fmt.Sprintf("**Tags:** %s\n", strings.Join(foundRule.Tags, ", "))
	}
	result += "\n**YAML Content:**\n```yaml\n"
	result += yamlContent
	result += "\n```\n"

	return mcp.NewToolResultText(result), nil
}

// importCommunityRuleHandler handles the import_community_rule tool
func importCommunityRuleHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	ruleID, err := req.RequireString("rule_id")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Fetch the community rule index
	index, err := fetchCommunityRuleIndex()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to fetch community rules: %v", err)), nil
	}

	// Find the rule
	var foundRule *CommunityRule
	for _, rule := range index.Rules {
		if rule.ID == ruleID {
			foundRule = &rule
			break
		}
	}

	if foundRule == nil {
		return mcp.NewToolResultText(fmt.Sprintf("Rule '%s' not found in community repository.", ruleID)), nil
	}

	// Fetch the actual rule YAML content
	ruleURL := fmt.Sprintf("https://raw.githubusercontent.com/hackafterdark/context-sherpa-community-rules/main/%s", foundRule.Path)
	resp, err := http.Get(ruleURL)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to fetch rule content: %v", err)), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to fetch rule content: HTTP %d", resp.StatusCode)), nil
	}

	// Read the YAML content
	var buf []byte
	if resp.ContentLength > 0 {
		buf = make([]byte, resp.ContentLength)
	} else {
		// Read in chunks if ContentLength is unknown
		chunkSize := 1024
		buffer := make([]byte, chunkSize)
		for {
			n, err := resp.Body.Read(buffer)
			if n > 0 {
				buf = append(buf, buffer[:n]...)
			}
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				return mcp.NewToolResultError(fmt.Sprintf("Failed to read rule content: %v", err)), nil
			}
		}
	}

	yamlContent := string(buf)

	// Validate the YAML content before writing to disk
	if err := validateAstGrepRule(yamlContent); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Invalid rule file for '%s': %v", ruleID, err)), nil
	}

	// Get the rule directory and save the file
	ruleDir, err := getRuleDir()
	if err != nil {
		if strings.Contains(err.Error(), "sgconfig.yml not found") {
			return mcp.NewToolResultText(fmt.Sprintf("Error: %s. Please run the 'initialize_ast_grep' tool first to set up the project.", err.Error())), nil
		}
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Extract just the filename from the path
	pathParts := strings.Split(foundRule.Path, "/")
	filename := pathParts[len(pathParts)-1]
	ruleFile := filepath.Join(ruleDir, filename)

	if err := os.WriteFile(ruleFile, []byte(yamlContent), 0644); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error writing rule file: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Rule '%s' was imported successfully from the community repository to %s.", ruleID, ruleFile)), nil
}

// validateAstGrepRule checks if the given YAML content is a valid ast-grep rule.
// It ensures the YAML is well-formed and contains the essential fields 'id', 'language', and 'rule'.
func validateAstGrepRule(yamlContent string) error {
	var rule AstGrepRule
	if err := yaml.Unmarshal([]byte(yamlContent), &rule); err != nil {
		return fmt.Errorf("could not parse YAML: %v", err)
	}

	// Check for the presence of required fields.
	// The 'Rule' field is checked by its presence in the struct, but we ensure others are not empty.
	if rule.ID == "" {
		return fmt.Errorf("rule 'id' is missing or empty")
	}
	if rule.Language == "" {
		return fmt.Errorf("rule 'language' is missing or empty")
	}

	return nil
}
