package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/hackafterdark/context-sherpa/pkg/inference"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	scip "github.com/sourcegraph/scip/bindings/go/scip"
	"google.golang.org/protobuf/proto"
	"gopkg.in/yaml.v3"
)

// HubLock represents the metadata stored in the hub.lock file
type HubLock struct {
	PID       int       `json:"pid"`
	Port      int       `json:"port"`
	StartTime time.Time `json:"startTime"`
}

// GetHubLockPath returns the platform-specific path to the hub.lock file
func GetHubLockPath() string {
	var baseDir string
	if runtime.GOOS == "windows" {
		baseDir = os.Getenv("LOCALAPPDATA")
	} else {
		home, _ := os.UserHomeDir()
		baseDir = filepath.Join(home, ".local", "share") // Standard for Linux/macOS
	}
	dir := filepath.Join(baseDir, "context-sherpa")
	_ = os.MkdirAll(dir, 0755)
	return filepath.Join(dir, "hub.lock")
}

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

// workspaceRootOverride stores the custom workspace root directory when specified via command-line argument
var workspaceRootOverride string

// astGrepPathOverride stores the custom ast-grep binary path when specified via command-line argument
var astGrepPathOverride string

// Logging configuration
var (
	verboseLogging bool
	logFile        *os.File
	customLogger   *log.Logger
)

// getCommunityRulesRepoURL returns the community rules repository URL (can be overridden in tests)
func getCommunityRulesRepoURL() string {
	return communityRulesRepo
}

// Start initializes and starts the MCP server.
func Start(workspaceRoot string, verbose bool, logFilePath string, astGrepPath string, clientName string) {
	if workspaceRoot == "" {
		// Attempt to auto-discover workspace root from executable location
		if exePath, err := os.Executable(); err == nil {
			if root, err := findWorkspaceRoot(filepath.Dir(exePath)); err == nil {
				workspaceRoot = root
				verboseLog("Auto-discovered workspace root: %s", workspaceRoot)
			}
		}
	}

	if workspaceRoot != "" {
		workspaceRootOverride = workspaceRoot
	}
	if astGrepPath != "" {
		astGrepPathOverride = astGrepPath
	}

	// Initialize logging system with the resolved workspace root
	initLogging(verbose, logFilePath, workspaceRoot)

	// Detect client if not provided
	if clientName == "" {
		clientName = detectClientName()
	}

	// Create a new MCP server
	s := server.NewMCPServer(
		"context-sherpa 🚀",
		"1.0.0",
		server.WithToolCapabilities(false),
	)

	// Semantic Reasoning Tools
	s.AddTool(mcp.NewTool("list_local_models",
		mcp.WithDescription("List available local SLMs and their status."),
	), listLocalModelsHandler)

	s.AddTool(mcp.NewTool("switch_local_model",
		mcp.WithDescription("Changes the active model in the Hub."),
		mcp.WithString("modelId", mcp.Required(), mcp.Description("ID of the model to activate.")),
	), switchLocalModelHandler)

	s.AddTool(mcp.NewTool("ask_little_brain",
		mcp.WithDescription("Sends a prompt to the local SLM for private, low-latency semantic tasks."),
		mcp.WithString("prompt", mcp.Required(), mcp.Description("The question or instruction for the SLM.")),
		mcp.WithString("modelId", mcp.Description("Optional model ID to use. Defaults to currently active.")),
		mcp.WithNumber("max_tokens", mcp.Description("Optional maximum number of tokens to generate. Defaults to 512.")),
		mcp.WithNumber("temperature", mcp.Description("Optional sampling temperature (0.0 to 1.0). Defaults to 0.1.")),
	), askLittleBrainHandler)

	// --- Workspace Initialization ---
	// Attempt to resolve workspace root early for local state and registration
	resolvedRoot, err := findWorkspaceRoot("")
	if err == nil {
		// 1. Initialize local state (.context-sherpa/ folder)
		if err := initLocalState(resolvedRoot); err != nil {
			customLogger.Printf("Warning: Failed to initialize local state: %v\n", err)
		}

		// 2. Register with Hub (Master Hub Ping)
		go registerWithHub(resolvedRoot, clientName)
	} else {
		customLogger.Printf("Warning: Could not resolve workspace root on startup: %v\n", err)
	}
	// --- End Workspace Initialization ---

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
			mcp.Description("Path to a specific sgconfig.yml file to use for the scan. If omitted, it defaults to the workspace root sgconfig.yml."),
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
			mcp.Description("Path to specific sgconfig.yml configuration file. If omitted, uses 'sgconfig.yml' in workspace root. Example: 'custom/sgconfig.yml'."),
		),
		mcp.WithString("language",
			mcp.Description("Programming language filter for directory scans. Supported: 'go', 'python', 'javascript', 'typescript', 'rust', 'java', 'cpp', 'c'. If specified, only files with matching extensions are scanned."),
		),
	)

	// Add get_symbol_map tool
	getSymbolMapTool := mcp.NewTool("get_symbol_map",
		mcp.WithDescription("Get a map of definitions and references for a specific symbol using SCIP code intelligence."),
		mcp.WithString("symbolName",
			mcp.Required(),
			mcp.Description("The name of the symbol to look up (e.g., 'AuthService')."),
		),
		mcp.WithString("language",
			mcp.Description("Optional language hint (e.g., 'go', 'typescript'). Defaults to workspace language detection."),
		),
		mcp.WithString("workspaceRoot",
			mcp.Description("Optional workspace root directory to index. If omitted, it will try to auto-detect the workspace root."),
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
		mcp.WithDescription("Remove a specific ast-grep rule file from the local workspace's rule directory."),
		mcp.WithString("rule_id",
			mcp.Required(),
			mcp.Description("The unique ID of the rule to be removed (e.g., 'no-sql-injection'). This should match the filename without the .yml extension."),
		),
	)

	// Add initialize_ast_grep tool
	initializeAstGrepTool := mcp.NewTool("initialize_ast_grep",
		mcp.WithDescription("Sets up the current workspace for ast-grep by creating a default `sgconfig.yml` file and a `rules/` directory. This is a required first step before adding or importing local rules."),
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
		mcp.WithDescription("Download a community rule and add it to the local workspace"),
		mcp.WithString("rule_id",
			mcp.Required(),
			mcp.Description("Unique identifier of the rule to import"),
		),
	)

	// Add list_symbols_in_file tool
	listSymbolsInFileTool := mcp.NewTool("list_symbols_in_file",
		mcp.WithDescription("Lists all classes, functions, and variables defined in a specific file using the symbolic index."),
		mcp.WithString("file_path",
			mcp.Required(),
			mcp.Description("File path relative to workspace root (e.g., 'pkg/mcp/server.go')."),
		),
		mcp.WithString("workspaceRoot",
			mcp.Description("Optional workspace root directory to index. If omitted, it will try to auto-detect the workspace root."),
		),
	)

	// Add search_definitions tool
	searchDefinitionsTool := mcp.NewTool("search_definitions",
		mcp.WithDescription("Searches for a symbol definition across the entire project using SCIP code intelligence."),
		mcp.WithString("query",
			mcp.Required(),
			mcp.Description("The symbol name to search for (e.g., 'App')."),
		),
		mcp.WithString("workspaceRoot",
			mcp.Description("Optional workspace root directory to index. If omitted, it will try to auto-detect the workspace root."),
		),
	)

	// Add initialize_scip tool
	initializeScipTool := mcp.NewTool("initialize_scip",
		mcp.WithDescription("Indexes the workspace for SCIP-based navigation."),
		mcp.WithString("workspaceRoot",
			mcp.Description("Optional workspace root directory to index. If omitted, it will try to auto-detect the workspace root."),
		),
		mcp.WithString("language",
			mcp.Description("Programming language of the project (e.g., 'go', 'typescript', 'python'). If omitted, will attempt auto-detection."),
		),
	)

	// Add tool handlers
	s.AddTool(scanCodeTool, scanCodeHandler)
	s.AddTool(scanPathTool, scanPathHandler)
	s.AddTool(getSymbolMapTool, getSymbolMapHandler)
	s.AddTool(listSymbolsInFileTool, listSymbolsInFileHandler)
	s.AddTool(searchDefinitionsTool, searchDefinitionsHandler)
	s.AddTool(initializeScipTool, initializeScipHandler)
	s.AddTool(addOrUpdateRuleTool, addOrUpdateRuleHandler)
	s.AddTool(removeRuleTool, removeRuleHandler)
	s.AddTool(initializeAstGrepTool, initializeAstGrepHandler)
	s.AddTool(searchCommunityRulesTool, searchCommunityRulesHandler)
	s.AddTool(getCommunityRuleDetailsTool, getCommunityRuleDetailsHandler)
	s.AddTool(importCommunityRuleTool, importCommunityRuleHandler)

	// Test ast-grep binary and log version information
	sgPath, err := findAstGrepBinary(astGrepPathOverride)
	if err != nil {
		customLogger.Printf("Failed to find ast-grep binary: %v\n", err)
		// Don't exit here - let the MCP tools handle the error when actually used
	} else {
		// Log ast-grep version for debugging and verification
		versionCmd := exec.Command(sgPath, "--version")
		if versionOutput, err := versionCmd.Output(); err == nil {
			customLogger.Printf("Using ast-grep: %s", strings.TrimSpace(string(versionOutput)))
		} else {
			customLogger.Printf("Warning: Could not get ast-grep version: %v", err)
		}
	}

	customLogger.Println("Starting MCP server...")

	// Start the stdio server
	if err := server.ServeStdio(s); err != nil {
		customLogger.Printf("Server error: %v\n", err)
	}
}

// initLocalState ensures the .context-sherpa directory exists in the workspace root
func initLocalState(workspaceRoot string) error {
	sherpaDir := filepath.Join(workspaceRoot, ".context-sherpa")
	if err := os.MkdirAll(sherpaDir, 0755); err != nil {
		return fmt.Errorf("failed to create .context-sherpa directory: %w", err)
	}

	// Check if we should initialize the database
	// dbPath := filepath.Join(sherpaDir, "sherpa.db")
	// For now, we just ensure the directory. Actual DB init can happen when needed.

	return nil
}

// registerWithHub pings the Master Hub to register this workspace
func registerWithHub(workspaceRoot string, clientName string) {
	// 1. Discover the Hub via lock file
	lockPath := GetHubLockPath()
	var hubPort int = 9000 // Default fallback

	// Retry loop
	for {
		data, err := os.ReadFile(lockPath)
		if err == nil {
			var lock HubLock
			if err := json.Unmarshal(data, &lock); err == nil {
				// Check if process is still alive (optional but recommended)
				if _, err := os.FindProcess(lock.PID); err == nil {
					// On Windows, FindProcess always succeeds, so we might need a more robust check later
					// But for now, trust the lock file
					hubPort = lock.Port
				}
			}
		}

		hubURL := fmt.Sprintf("http://127.0.0.1:%d/workspaces", hubPort)
		verboseLog("Attempting to register workspace with Hub at %s...", hubURL)

		payload := map[string]interface{}{
			"pid":    os.Getpid(),
			"root":   workspaceRoot,
			"client": clientName,
			"state":  "active",
		}

		jsonData, _ := json.Marshal(payload)
		req, err := http.NewRequest(http.MethodPut, hubURL, strings.NewReader(string(jsonData)))
		if err == nil {
			req.Header.Set("Content-Type", "application/json")
			client := &http.Client{Timeout: 2 * time.Second}
			resp, err := client.Do(req)
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusNoContent || resp.StatusCode == http.StatusOK {
					verboseLog("Heartbeat: Successfully registered workspace with Hub: %s", workspaceRoot)
					// Heartbeat interval: 30 seconds
					time.Sleep(30 * time.Second)
					continue
				}
				verboseLog("Hub returned error status: %d", resp.StatusCode)
			} else {
				verboseLog("Hub registration ping failed: %v", err)
			}
		} else {
			verboseLog("Failed to create registration request: %v", err)
		}

		verboseLog("Retrying registration in 5 seconds...")
		time.Sleep(5 * time.Second)
	}
}

func getSymbolMapHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	symbolName, err := req.RequireString("symbolName")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("symbolName is required: %v", err)), nil
	}

	workspaceRootArg := ""
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if wr, ok := args["workspaceRoot"].(string); ok && wr != "" {
			workspaceRootArg = wr
		}
	}

	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v", err)), nil
	}

	indexes, err := loadSCIPIndexes(workspaceRoot)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to load SCIP indexes: %v", err)), nil
	}
	if len(indexes) == 0 {
		return mcp.NewToolResultError("SCIP index not found. Please index the workspace first via the Dashboard."), nil
	}

	// Search for symbol across all indexes
	result := map[string]interface{}{
		"symbol":     symbolName,
		"definition": nil,
		"references": []map[string]interface{}{},
	}

	for _, index := range indexes {
		for _, doc := range index.Documents {
			rel := filepath.ToSlash(doc.RelativePath)
			for _, occ := range doc.Occurrences {
				if isSymbolMatch(occ.Symbol, symbolName) {
					loc := map[string]interface{}{
						"file": rel,
						"line": occ.Range[0] + 1,
					}
					isDef := occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0
					if isDef && result["definition"] == nil {
						result["definition"] = loc
					} else {
						result["references"] = append(result["references"].([]map[string]interface{}), loc)
					}
				}
			}
		}
	}

	jsonRes, _ := json.MarshalIndent(result, "", "  ")
	return mcp.NewToolResultText(string(jsonRes)), nil
}

// detectClientName attempts to identify the MCP client
func detectClientName() string {
	// 1. Check for explicit environment variable
	if client := os.Getenv("CONTEXT_SHERPA_CLIENT"); client != "" {
		return client
	}

	// 2. Check for IDE-specific variables
	if os.Getenv("VSCODE_PID") != "" || os.Getenv("VSCODE_GIT_IPC_HANDLE") != "" {
		return "vscode"
	}
	if os.Getenv("INTELLIJ_PID") != "" {
		return "intellij"
	}
	if os.Getenv("TERM_PROGRAM") == "vscode" {
		return "vscode"
	}

	// 3. Fallback to parent process name
	ppid := os.Getppid()
	if ppid > 0 {
		var cmd *exec.Cmd
		if runtime.GOOS == "windows" {
			// On Windows, tasklist or wmic can be used. tasklist is more common.
			cmd = exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", ppid), "/FO", "CSV", "/NH")
		} else {
			cmd = exec.Command("ps", "-p", fmt.Sprintf("%d", ppid), "-o", "comm=")
		}

		if output, err := cmd.Output(); err == nil {
			name := strings.TrimSpace(string(output))
			if runtime.GOOS == "windows" {
				// CSV format: "exe name","pid","session name","session#","mem"
				parts := strings.Split(name, ",")
				if len(parts) > 0 {
					name = strings.Trim(parts[0], "\"")
				}
			}
			if name != "" {
				return name
			}
		}
	}

	return "unknown"
}

func listSymbolsInFileHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	filePath, err := req.RequireString("file_path")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("file_path is required: %v", err)), nil
	}

	workspaceRootArg := ""
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if wr, ok := args["workspaceRoot"].(string); ok && wr != "" {
			workspaceRootArg = wr
		}
	}

	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v", err)), nil
	}

	indexes, err := loadSCIPIndexes(workspaceRoot)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to load SCIP indexes: %v", err)), nil
	}
	if len(indexes) == 0 {
		return mcp.NewToolResultError("SCIP index not found. Please index the workspace first via the Dashboard."), nil
	}
	verboseLog("listSymbolsInFileHandler: loaded %d indexes", len(indexes))

	// Normalize input path to match SCIP's internal format (use forward slashes consistently)
	inputPath := filepath.ToSlash(filePath)
	verboseLog("listSymbolsInFileHandler: searching for '%s' (inputPath), original: '%s'", inputPath, filePath)

	var symbols []map[string]interface{}
	for idxNum, index := range indexes {
		verboseLog("Searching index %d, documents: %d", idxNum, len(index.Documents))
		for _, doc := range index.Documents {
			rel := filepath.ToSlash(doc.RelativePath)
			if rel == inputPath {
				verboseLog("listSymbolsInFileHandler: matched file '%s' in index %d", rel, idxNum)
				for _, occ := range doc.Occurrences {
					// Only include definitions (Role 1)
					isDef := occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0

					if isDef {
						symbolInfo := map[string]interface{}{
							"symbol": occ.Symbol,
							"line":   occ.Range[0] + 1,
						}
						symbols = append(symbols, symbolInfo)
					}
				}
				break
			}
		}
	}

	if len(symbols) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("No symbols found in file: %s", filePath)), nil
	}

	jsonRes, _ := json.MarshalIndent(symbols, "", "  ")
	return mcp.NewToolResultText(string(jsonRes)), nil
}

func searchDefinitionsHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query, err := req.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("query is required: %v", err)), nil
	}

	workspaceRootArg := ""
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if wr, ok := args["workspaceRoot"].(string); ok && wr != "" {
			workspaceRootArg = wr
		}
	}

	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v", err)), nil
	}

	indexes, err := loadSCIPIndexes(workspaceRoot)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to load SCIP indexes: %v", err)), nil
	}
	if len(indexes) == 0 {
		return mcp.NewToolResultError("SCIP index not found. Please index the workspace first via the Dashboard."), nil
	}

	var definitions []map[string]interface{}
	for _, index := range indexes {
		for _, doc := range index.Documents {
			rel := filepath.ToSlash(doc.RelativePath)
			for _, occ := range doc.Occurrences {
				if isSymbolMatch(occ.Symbol, query) {
					isDef := occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0
					if isDef {
						definitions = append(definitions, map[string]interface{}{
							"symbol": occ.Symbol,
							"file":   rel,
							"line":   occ.Range[0] + 1,
						})
					}
				}
			}
		}
	}

	if len(definitions) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("No definitions found for query: %s (Loaded %d indexes)", query, len(indexes))), nil
	}

	jsonRes, _ := json.MarshalIndent(definitions, "", "  ")
	return mcp.NewToolResultText(string(jsonRes)), nil
}

func initializeScipHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	workspaceRootArg := ""
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if wr, ok := args["workspaceRoot"].(string); ok && wr != "" {
			workspaceRootArg = wr
		}
	}

	workspaceRoot := ""
	if workspaceRootArg == "" {
		wr, err := findWorkspaceRoot("")
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v", err)), nil
		}
		workspaceRoot = wr
	} else {
		// Normalize explicit path
		if abs, err := filepath.Abs(workspaceRootArg); err == nil {
			workspaceRoot = abs
		} else {
			workspaceRoot = workspaceRootArg
		}
		if runtime.GOOS == "windows" && len(workspaceRoot) > 1 && workspaceRoot[1] == ':' {
			workspaceRoot = strings.ToUpper(string(workspaceRoot[0])) + workspaceRoot[1:]
		}
	}

	language := ""
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if l, ok := args["language"].(string); ok && l != "" {
			language = l
		}
	}

	// Auto-detect language if not provided
	if language == "" {
		if _, err := os.Stat(filepath.Join(workspaceRoot, "go.mod")); err == nil {
			language = "go"
		} else if _, err := os.Stat(filepath.Join(workspaceRoot, "package.json")); err == nil {
			language = "typescript"
		} else if _, err := os.Stat(filepath.Join(workspaceRoot, "requirements.txt")); err == nil {
			language = "python"
		} else if _, err := os.Stat(filepath.Join(workspaceRoot, "setup.py")); err == nil {
			language = "python"
		}
	}

	if language == "" {
		return mcp.NewToolResultError("Could not auto-detect project language. Please specify 'language' parameter."), nil
	}

	if err := indexWorkspace(workspaceRoot, language); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Indexing failed: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Workspace indexed successfully for %s.", language)), nil
}

func indexWorkspace(workspaceRoot string, language string) error {
	// 0. Use canonical absolute path
	if abs, err := filepath.Abs(workspaceRoot); err == nil {
		workspaceRoot = abs
	}
	// Normalization: use uppercase drive letter on Windows for consistency
	if runtime.GOOS == "windows" && len(workspaceRoot) > 1 && workspaceRoot[1] == ':' {
		workspaceRoot = strings.ToUpper(string(workspaceRoot[0])) + workspaceRoot[1:]
	}
	verboseLog("indexWorkspace: entry for %s at %s", language, workspaceRoot)

	// Ensure .context-sherpa directory exists
	if err := initLocalState(workspaceRoot); err != nil {
		return err
	}

	// 1. Detect language if not provided
	if language == "" {
		if _, err := os.Stat(filepath.Join(workspaceRoot, "go.mod")); err == nil {
			language = "go"
		} else if _, err := os.Stat(filepath.Join(workspaceRoot, "package.json")); err == nil {
			language = "typescript"
		} else if _, err := os.Stat(filepath.Join(workspaceRoot, "requirements.txt")); err == nil || (func() bool { _, e := os.Stat(filepath.Join(workspaceRoot, "pyproject.toml")); return e == nil }()) {
			language = "python"
		} else {
			return fmt.Errorf("could not auto-detect language for workspace: %s", workspaceRoot)
		}
	}

	var cmd *exec.Cmd

	var homeDir string
	if runtime.GOOS == "windows" {
		homeDir = os.Getenv("LOCALAPPDATA")
		if homeDir == "" {
			homeDir, _ = os.UserHomeDir()
		}
	} else {
		homeDir, _ = os.UserHomeDir()
	}

	binDir := filepath.Join(homeDir, "context-sherpa", "bin")
	if runtime.GOOS != "windows" {
		binDir = filepath.Join(homeDir, ".context-sherpa", "bin")
	}

	if language == "go" {
		// Use scip-go. Try to find it in PATH or context-sherpa/bin
		indexerName := "scip-go"
		if runtime.GOOS == "windows" {
			indexerName += ".exe"
		}
		indexerPath := indexerName
		localBin := filepath.Join(binDir, indexerName)

		if _, err := os.Stat(localBin); err == nil {
			indexerPath = localBin
		}

		absOutput := filepath.Join(workspaceRoot, ".context-sherpa", fmt.Sprintf("index-%s.scip", language))
		cmd = exec.Command(indexerPath, "--project-root", workspaceRoot, "--repository-root", workspaceRoot, "--output", absOutput)
		cmd.Dir = workspaceRoot
	} else {
		// Try to find managed indexer for other languages
		indexerName := "scip-" + language

		// Priority paths
		var pathsToTry []string
		if runtime.GOOS == "windows" {
			// Local bin might have .exe (e.g. if we compiled it)
			pathsToTry = append(pathsToTry, filepath.Join(binDir, indexerName+".exe"))
			// NPM bin usually has .cmd or .ps1
			pathsToTry = append(pathsToTry, filepath.Join(binDir, "node_modules", ".bin", indexerName+".cmd"))
			pathsToTry = append(pathsToTry, filepath.Join(binDir, "node_modules", ".bin", indexerName+".ps1"))
		} else {
			pathsToTry = append(pathsToTry, filepath.Join(binDir, indexerName))
			pathsToTry = append(pathsToTry, filepath.Join(binDir, "node_modules", ".bin", indexerName))
		}

		indexPath := ""
		for _, p := range pathsToTry {
			if _, err := os.Stat(p); err == nil {
				indexPath = p
				break
			}
		}

		if indexPath == "" {
			if path, err := exec.LookPath(indexerName); err == nil {
				indexPath = path
			}
		}

		if indexPath == "" {
			return fmt.Errorf("indexer for %s not found. Please install it via the Context-Sherpa Dashboard", language)
		}

		absOutput := filepath.Join(workspaceRoot, ".context-sherpa", fmt.Sprintf("index-%s.scip", language))

		// On Windows, if we are running a .cmd or .ps1, we might need to invoke it via cmd /c
		if runtime.GOOS == "windows" && (strings.HasSuffix(indexPath, ".cmd") || strings.HasSuffix(indexPath, ".ps1")) {
			verboseLog("Running indexer via cmd /c: %s index --output %s", indexPath, absOutput)
			cmd = exec.Command("cmd", "/c", indexPath, "index", "--output", absOutput)
		} else {
			verboseLog("Running indexer: %s index --output %s", indexPath, absOutput)
			cmd = exec.Command(indexPath, "index", "--output", absOutput)
		}
		cmd.Dir = workspaceRoot

	}

	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("indexing failed: %v\nOutput: %s", err, string(output))
	}

	return nil
}

// isSymbolMatch checks if a SCIP symbol string represents the given symbol name.
// It handles various SCIP naming conventions (suffixes like #, ., (), etc.)
func isSymbolMatch(scipSymbol string, symbolName string) bool {
	if scipSymbol == symbolName {
		return true
	}

	// SCIP symbols are structured. We care about the "base" name at the end.
	// scip-go often encloses the symbol in backticks: `SymbolName`

	// Check for backticked version
	backticked := "`" + symbolName + "`"
	if strings.Contains(scipSymbol, backticked) {
		// Ensure it's not part of a larger word
		idx := strings.Index(scipSymbol, backticked)
		// Check character before and after
		if idx > 0 {
			prev := scipSymbol[idx-1]
			if prev != '/' && prev != ' ' && prev != '.' && prev != '#' {
				// Part of a larger word?
			} else {
				return true
			}
		} else {
			return true
		}
	}

	// Common suffixes:
	// - Go Types: TypeName#
	// - Go Methods: TypeName#MethodName().
	// - Go Functions: FuncName().
	// - TS/Py: .../SymbolName

	// Check for suffixes
	suffixes := []string{"#", ".", "().", "()", ":"}
	for _, s := range suffixes {
		if strings.HasSuffix(scipSymbol, symbolName+s) {
			// Ensure it's preceded by a separator to avoid partial matches (e.g., "MyApp" matching "App")
			prefix := strings.TrimSuffix(scipSymbol, symbolName+s)
			if prefix == "" || strings.HasSuffix(prefix, "/") || strings.HasSuffix(prefix, " ") || strings.HasSuffix(prefix, ".") || strings.HasSuffix(prefix, "#") || strings.HasSuffix(prefix, ":") {
				return true
			}
		}
	}

	// Check if it ends exactly with the symbol name preceded by a separator
	separators := []string{"/", " ", ".", "#", ":"}
	for _, s := range separators {
		if strings.HasSuffix(scipSymbol, s+symbolName) {
			return true
		}
	}

	return false
}

func scanCodeHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var workspaceRoot string

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
	verboseLog("scanCodeHandler: Using sgconfig file: %s", sgconfigStr)
	// --- END DEBUG LOGGING ---

	// Find the workspace root where sgconfig.yml is located
	workspaceRoot, err = findWorkspaceRoot("")
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Resolve sgconfig path relative to workspace root
	resolvedSgconfigPath := resolvePathRelativeToWorkspaceRoot(sgconfigStr, workspaceRoot)

	// Check if the configuration file exists at the resolved path
	if _, err := os.Stat(resolvedSgconfigPath); os.IsNotExist(err) {
		return mcp.NewToolResultText(fmt.Sprintf("Error: Configuration file '%s' not found at resolved path '%s'. Please run the 'initialize_ast_grep' tool first to set up the workspace.", sgconfigStr, resolvedSgconfigPath)), nil
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

	sgPath, err := findAstGrepBinary(astGrepPathOverride)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error finding ast-grep binary: %v", err)), nil
	}

	cmd := exec.Command(sgPath, "scan", "--config", resolvedSgconfigPath, tmpfile.Name(), "--json")
	cmd.Dir = workspaceRoot // Run ast-grep from the workspace root
	output, err := cmd.Output()
	if err != nil {
		// ast-grep exits with non-zero status code if issues are found.
		// We still want to parse the output.
	}

	return mcp.NewToolResultText(string(output)), nil
}

// scanPathHandler handles the scan_path tool
func scanPathHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var workspaceRoot string

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
	verboseLog("scanPathHandler: Using sgconfig file: %s", sgconfigStr)
	verboseLog("scanPathHandler: Scanning path: %s", path)
	if languageFilter != "" {
		verboseLog("scanPathHandler: Language filter: %s", languageFilter)
	}
	// --- END DEBUG LOGGING ---

	// Find the workspace root where sgconfig.yml is located
	workspaceRoot, err = findWorkspaceRoot(path)

	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	// Resolve sgconfig path relative to workspace root
	resolvedSgconfigPath := resolvePathRelativeToWorkspaceRoot(sgconfigStr, workspaceRoot)

	// Check if the configuration file exists at the resolved path
	if _, err := os.Stat(resolvedSgconfigPath); os.IsNotExist(err) {
		return mcp.NewToolResultText(fmt.Sprintf("Error: Configuration file '%s' not found at resolved path '%s'. Please run the 'initialize_ast_grep' tool first to set up the workspace.", sgconfigStr, resolvedSgconfigPath)), nil
	}

	sgPath, err := findAstGrepBinary(astGrepPathOverride)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error finding ast-grep binary: %v", err)), nil
	}

	// Discover files to scan
	files, err := discoverFiles(path, languageFilter, workspaceRoot)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error discovering files: %v", err)), nil
	}

	if len(files) == 0 {
		return mcp.NewToolResultText("[]"), nil // Return empty JSON array for no files
	}

	// --- DEBUG LOGGING ---
	verboseLog("scan_file: Found %d files to scan", len(files))
	// --- END DEBUG LOGGING ---

	// Filter files by size (1MB limit) - FIRST ITERATION FEATURE
	var validFiles []string
	var skippedFiles []string
	for _, file := range files {
		fileInfo, err := os.Stat(file)
		if err != nil {
			verboseLog("scan_file: Warning - could not stat file %s: %v", file, err)
			continue
		}

		if fileInfo.Size() > 1024*1024 { // 1MB limit
			skippedFiles = append(skippedFiles, file)
			verboseLog("scan_file: Skipping file %s (size: %d bytes > 1MB limit)", file, fileInfo.Size())
			continue
		}

		validFiles = append(validFiles, file)
	}

	// --- DEBUG LOGGING ---
	verboseLog("scan_file: %d files valid for scanning, %d files skipped (over 1MB)", len(validFiles), len(skippedFiles))
	// --- END DEBUG LOGGING ---

	if len(validFiles) == 0 {
		return mcp.NewToolResultText("[]"), nil // Return empty JSON array for no valid files
	}

	// Scan files in batches
	allOutput, err := scanFileBatch(validFiles, resolvedSgconfigPath, workspaceRoot, sgPath)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error scanning files: %v", err)), nil
	}

	return mcp.NewToolResultText(allOutput), nil
}

// discoverFiles discovers files to scan based on the path pattern
func discoverFiles(path, languageFilter, workspaceRoot string) ([]string, error) {
	var files []string

	// Check if path is a direct file
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		// Resolve file path relative to workspace root for consistency
		resolvedPath := resolvePathRelativeToWorkspaceRoot(path, workspaceRoot)

		// Apply language filter if specified
		if languageFilter != "" && !matchesLanguage(resolvedPath, languageFilter) {
			return files, nil // Return empty slice if file doesn't match language filter
		}
		return []string{resolvedPath}, nil
	}

	// Handle directory scanning (when path is "." or a directory)
	if path == "." {
		// Resolve "." relative to workspace root
		searchRoot := workspaceRoot
		if searchRoot == "" {
			searchRoot = "."
		}

		err := filepath.Walk(searchRoot, func(currentPath string, info os.FileInfo, err error) error {
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
		// For directory check, we need to resolve the path first to handle workspace root override
		resolvedPath := resolvePathRelativeToWorkspaceRoot(path, workspaceRoot)

		// Verify the resolved path is actually a directory
		if resolvedInfo, err := os.Stat(resolvedPath); err != nil || !resolvedInfo.IsDir() {
			// If resolved path doesn't exist or isn't a directory, fall back to original path
			resolvedPath = path
		}

		err := filepath.Walk(resolvedPath, func(currentPath string, info os.FileInfo, err error) error {
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

	// Handle glob patterns - walk workspace root directory and match patterns
	searchRoot := workspaceRoot
	if searchRoot == "" {
		searchRoot = "."
	}

	err := filepath.Walk(searchRoot, func(currentPath string, info os.FileInfo, err error) error {
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
func scanFileBatch(files []string, sgconfigStr, workspaceRoot, sgPath string) (string, error) {
	if len(files) == 0 {
		return "[]", nil
	}

	// For now, scan all files in a single batch
	// TODO: Implement batching for very large file lists
	args := []string{"scan", "--config", sgconfigStr}
	args = append(args, files...)
	args = append(args, "--json")

	cmd := exec.Command(sgPath, args...)
	cmd.Dir = workspaceRoot

	output, err := cmd.Output()
	if err != nil {
		// ast-grep exits with non-zero status code if issues are found.
		// We still want to parse the output.
		verboseLog("scan_file: ast-grep command exited with error: %v", err)
	}

	// Log the actual ast-grep command output when verbose logging is enabled
	verboseLog("scan_file: ast-grep command output: %s", string(output))

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
			return mcp.NewToolResultText(fmt.Sprintf("Error: %s. Please run the 'initialize_ast_grep' tool first to set up the workspace.", err.Error())), nil
		}
		return mcp.NewToolResultError(err.Error()), nil
	}

	ruleFile := fmt.Sprintf("%s/%s.yml", ruleDir, ruleID)
	if err := os.WriteFile(ruleFile, []byte(ruleYAML), 0644); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error writing rule file: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Rule '%s' was added or updated successfully.", ruleID)), nil
}

// findWorkspaceRoot finds the workspace root by searching for sgconfig.yml
// or .git in the requested directory and its parent directories.
func findWorkspaceRoot(startPath string) (string, error) {
	var dir string

	if workspaceRootOverride != "" {
		dir = workspaceRootOverride
	} else if startPath != "" {
		dir = startPath
	} else {
		dir, _ = os.Getwd()
	}

	// Normalize path (especially drive letter on Windows)
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	if runtime.GOOS == "windows" && len(dir) > 1 && dir[1] == ':' {
		dir = strings.ToUpper(string(dir[0])) + dir[1:]
	}

	// 0. Search for .context-sherpa (our own metadata folder)
	tempDir := dir
	for {
		sherpaPath := filepath.Join(tempDir, ".context-sherpa")
		verboseLog("Checking for SCIP index at: %s", sherpaPath)
		if info, err := os.Stat(sherpaPath); err == nil && info.IsDir() {
			verboseLog("Found workspace root with .context-sherpa: %s", tempDir)
			return tempDir, nil
		}

		parentDir := filepath.Dir(tempDir)
		if parentDir == tempDir {
			break
		}
		tempDir = parentDir
	}

	// 1. Search for sgconfig.yml (ast-grep workspace)
	tempDir = dir
	for {
		configPath := filepath.Join(tempDir, "sgconfig.yml")
		if _, err := os.Stat(configPath); err == nil {
			return tempDir, nil
		}

		parentDir := filepath.Dir(tempDir)
		if parentDir == tempDir {
			break
		}
		tempDir = parentDir
	}

	// 2. Search for git root
	tempDir = dir
	// Try git rev-parse first
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	cmd.Dir = dir
	if gitOutput, err := cmd.Output(); err == nil {
		gitRoot := strings.TrimSpace(string(gitOutput))
		if gitRoot != "" {
			return gitRoot, nil
		}
	}

	// Fallback to manual .git discovery
	for {
		gitPath := filepath.Join(tempDir, ".git")
		if _, err := os.Stat(gitPath); err == nil {
			return tempDir, nil
		}

		parentDir := filepath.Dir(tempDir)
		if parentDir == tempDir {
			break
		}
		tempDir = parentDir
	}

	return "", fmt.Errorf("could not resolve workspace root (no sgconfig.yml or .git found). Please run 'ast-grep new' or initialize a git repository first")
}

func findAstGrepBinary(astGrepPath string) (string, error) {
	// 1. User explicitly specified path (highest priority)
	if astGrepPath != "" {
		if _, err := os.Stat(astGrepPath); err == nil {
			verboseLog("Using user-specified ast-grep path: %s", astGrepPath)
			return astGrepPath, nil
		}
		return "", fmt.Errorf("ast-grep not found at specified path: %s", astGrepPath)
	}

	// 2. System PATH (standard location)
	if path, err := exec.LookPath("ast-grep"); err == nil {
		verboseLog("Found ast-grep in PATH: %s", path)
		return path, nil
	}

	if runtime.GOOS == "windows" {
		if path, err := exec.LookPath("ast-grep.exe"); err == nil {
			verboseLog("Found ast-grep.exe in PATH: %s", path)
			return path, nil
		}
	}

	// 3. Custom Context-Sherpa installation directory
	var homeDir string
	var err error
	if runtime.GOOS == "windows" {
		homeDir = os.Getenv("LOCALAPPDATA")
		if homeDir == "" {
			homeDir, err = os.UserHomeDir()
		}
	} else {
		homeDir, err = os.UserHomeDir()
	}

	if err == nil {
		binDir := filepath.Join(homeDir, "context-sherpa", "bin")
		if runtime.GOOS != "windows" {
			binDir = filepath.Join(homeDir, ".context-sherpa", "bin")
		}

		binName := "ast-grep"
		if runtime.GOOS == "windows" {
			binName = "ast-grep.exe"
		}

		customPath := filepath.Join(binDir, binName)
		if _, err := os.Stat(customPath); err == nil {
			verboseLog("Found ast-grep in custom context-sherpa bin dir: %s", customPath)
			return customPath, nil
		}
	}

	// 4. Clear error - explain MCP server limitation
	return "", fmt.Errorf(`ast-grep not found in PATH or in the global Context-Sherpa directory.

As an MCP server communicating via stdio, I cannot:
- Detect where your editor/IDE is running from
- Access your current working directory
- Find binaries in workspace-specific locations

Please ensure ast-grep is available in one of these ways:

1. Use the Context-Sherpa Desktop Dashboard to install it (recommended).

2. Install in system PATH (see https://ast-grep.github.io/guide/quick-start.html):
   # Choose one of these installation methods:
   brew install ast-grep                    # macOS/Linux
   cargo install ast-grep --locked         # Rust
   npm i @ast-grep/cli -g                  # Node.js
   pip install ast-grep-cli                # Python
   sudo port install ast-grep              # MacPorts

3. Specify explicit path:
   context-sherpa --astGrepPath="/path/to/ast-grep"`)
}

// getSherpaModelsDir returns the platform-specific path to the models directory
func getSherpaModelsDir() (string, error) {
	var homeDir string
	var err error
	if runtime.GOOS == "windows" {
		homeDir = os.Getenv("LOCALAPPDATA")
		if homeDir == "" {
			homeDir, err = os.UserHomeDir()
		}
	} else {
		homeDir, err = os.UserHomeDir()
	}

	if err != nil {
		return "", err
	}

	modelsDir := filepath.Join(homeDir, "context-sherpa", "models")
	if runtime.GOOS != "windows" {
		modelsDir = filepath.Join(homeDir, ".context-sherpa", "models")
	}
	_ = os.MkdirAll(modelsDir, 0755)
	return modelsDir, nil
}

// getRuleDir determines the directory where rules should be stored by searching
// for sgconfig.yml in the current and parent directories.
func getRuleDir() (string, error) {
	var dir string
	var err error

	if workspaceRootOverride != "" {
		// Use the specified workspace root as starting point
		verboseLog("Using custom workspace root override: %s", workspaceRootOverride)
		dir = workspaceRootOverride
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

	return "", fmt.Errorf("sgconfig.yml not found. Please run 'ast-grep new' to initialize an ast-grep workspace first")
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
			return mcp.NewToolResultText(fmt.Sprintf("Error: %s. Please run the 'initialize_ast_grep' tool first to set up the workspace.", err.Error())), nil
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

// initializeAstGrepHandler handles the initialize_ast_grep tool
func initializeAstGrepHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	var workspaceRoot string
	var err error

	if workspaceRootOverride != "" {
		workspaceRoot = workspaceRootOverride
	} else {
		workspaceRoot, err = os.Getwd()
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Error getting current directory: %v", err)), nil
		}
	}

	// Create rules directory
	rulesDir := filepath.Join(workspaceRoot, "rules")
	if err := os.MkdirAll(rulesDir, 0755); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error creating rules directory: %v", err)), nil
	}

	// Create default sgconfig.yml
	sgconfigPath := filepath.Join(workspaceRoot, "sgconfig.yml")
	if _, err := os.Stat(sgconfigPath); os.IsNotExist(err) {
		config := SgConfig{
			RuleDirs: []string{"rules"},
		}
		data, err := yaml.Marshal(config)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Error creating sgconfig.yml: %v", err)), nil
		}
		if err := os.WriteFile(sgconfigPath, data, 0644); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Error writing sgconfig.yml: %v", err)), nil
		}
	}

	return mcp.NewToolResultText("ast-grep has been initialized successfully. You can now add rules to the 'rules' directory."), nil
}

// fetchCommunityRuleIndex fetches and caches the community rule index
func fetchCommunityRuleIndex() (*CommunityRuleIndex, error) {
	// Check if we have a valid cached index
	if communityRuleCache != nil && time.Since(cacheTimestamp) < cacheTTL {
		verboseLog("Using cached community rule index")
		return communityRuleCache, nil
	}

	verboseLog("Fetching community rule index from repository")

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

	verboseLog("Successfully loaded %d community rules", len(index.Rules))
	return &index, nil
}

// initLogging initializes the logging system with support for verbose logging and log files
func initLogging(verbose bool, logFilePath string, workspaceRoot string) {
	verboseLogging = verbose

	var writers []io.Writer

	// MCP protocol requires strictly writing ONLY JSON-RPC to stdout.
	// All logging MUST go to stderr.
	writers = append(writers, os.Stderr)

	// If log file is specified, append to it. Otherwise use a default in workspace root if possible.
	if logFilePath == "" && workspaceRoot != "" {
		logFilePath = filepath.Join(workspaceRoot, "mcp_debug.log")
	}

	if logFilePath != "" {
		var err error
		logFile, err = os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			log.Printf("Warning: Failed to open log file %s: %v", logFilePath, err)
		} else {
			writers = append(writers, logFile)
		}
	}

	// Create multi-writer for logging
	customLogger = log.New(io.MultiWriter(writers...), "", log.LstdFlags)

	if verbose {
		customLogger.Println("Verbose logging enabled")
		if logFilePath != "" {
			customLogger.Printf("Logging to file: %s", logFilePath)
		}
	}
}

// verboseLog logs a message only if verbose logging is enabled
func verboseLog(format string, v ...interface{}) {
	if verboseLogging && customLogger != nil {
		customLogger.Printf(format, v...)
	}
}

// resolvePathRelativeToWorkspaceRoot resolves a user-provided path relative to the workspace root.
// If the path is already absolute, it returns it as is.
// If workspaceRoot is empty, it treats the path as relative to current directory.
func resolvePathRelativeToWorkspaceRoot(path, workspaceRoot string) string {
	if filepath.IsAbs(path) {
		return path
	}

	if workspaceRoot == "" {
		return path
	}

	resolvedPath := filepath.Join(workspaceRoot, path)
	// Log resolution for debugging
	verboseLog("resolvePathRelativeToWorkspaceRoot: '%s' resolved to '%s' (workspaceRoot: '%s')", path, resolvedPath, workspaceRoot)
	return resolvedPath
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
			return mcp.NewToolResultText(fmt.Sprintf("Error: %s. Please run the 'initialize_ast_grep' tool first to set up the workspace.", err.Error())), nil
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

func listLocalModelsHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	modelsDir, err := getSherpaModelsDir()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error finding models directory: %v", err)), nil
	}

	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error reading models directory: %v", err)), nil
	}

	var sb strings.Builder
	sb.WriteString("Available Local Models (GGUF):\n")
	found := false
	for _, entry := range entries {
		if !entry.IsDir() {
			sb.WriteString(fmt.Sprintf("- %s\n", entry.Name()))
			found = true
		}
	}

	if !found {
		sb.WriteString("(No models found. Please download them from the Dashboard.)\n")
	}

	return mcp.NewToolResultText(sb.String()), nil
}

func switchLocalModelHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	// In one-shot mode, we don't 'switch' models in the backend.
	// The client just provides the modelId in the inference request.
	modelID, _ := request.RequireString("modelId")
	return mcp.NewToolResultText(fmt.Sprintf("Model selection for one-shot tasks updated: %s. Use this ID in your next ask_little_brain call.", modelID)), nil
}

func askLittleBrainHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	prompt, err := request.RequireString("prompt")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("prompt is required: %v", err)), nil
	}

	modelID := ""
	maxTokens := 0
	temperature := float32(0)

	if args, ok := request.Params.Arguments.(map[string]interface{}); ok {
		if m, ok := args["modelId"].(string); ok {
			modelID = m
		}
		if mt, ok := args["max_tokens"].(float64); ok {
			maxTokens = int(mt)
		}
		if temp, ok := args["temperature"].(float64); ok {
			temperature = float32(temp)
		}
	}

	modelsDir, err := getSherpaModelsDir()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error finding models directory: %v", err)), nil
	}

	svc := inference.NewInferenceService(modelsDir)
	res, err := svc.Execute(ctx, inference.InferenceRequest{
		ModelID:     modelID,
		Prompt:      prompt,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	})

	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Inference failed: %v", err)), nil
	}

	return mcp.NewToolResultText(res.Text), nil
}

// loadSCIPIndexes loads all index-*.scip (and index.scip) files found in .context-sherpa directories
// within the workspaceRoot or its first-level subdirectories.
func loadSCIPIndexes(workspaceRoot string) ([]*scip.Index, error) {
	var indexes []*scip.Index
	var searchDirs []string

	// 1. Check root .context-sherpa
	rootSherpa := filepath.Join(workspaceRoot, ".context-sherpa")
	if _, err := os.Stat(rootSherpa); err == nil {
		searchDirs = append(searchDirs, rootSherpa)
	}

	// 2. Check first-level subdirectories for .context-sherpa
	entries, err := os.ReadDir(workspaceRoot)
	if err == nil {
		for _, entry := range entries {
			if entry.IsDir() && !strings.HasPrefix(entry.Name(), ".") {
				subSherpa := filepath.Join(workspaceRoot, entry.Name(), ".context-sherpa")
				if _, err := os.Stat(subSherpa); err == nil {
					searchDirs = append(searchDirs, subSherpa)
				}
			}
		}
	}

	for _, sherpaDir := range searchDirs {
		verboseLog("Searching for indexes in: %s", sherpaDir)
		files, err := os.ReadDir(sherpaDir)
		if err != nil {
			continue
		}

		for _, file := range files {
			if !file.IsDir() && (strings.HasPrefix(file.Name(), "index-") && strings.HasSuffix(file.Name(), ".scip") || file.Name() == "index.scip") {
				scipPath := filepath.Join(sherpaDir, file.Name())
				verboseLog("Loading SCIP index: %s", scipPath)
				data, err := os.ReadFile(scipPath)
				if err != nil {
					verboseLog("Warning: failed to read index %s: %v", scipPath, err)
					continue
				}

				var index scip.Index
				if err := proto.Unmarshal(data, &index); err != nil {
					verboseLog("Warning: failed to parse index %s: %v", scipPath, err)
					continue
				}
				indexes = append(indexes, &index)
			}
		}
	}

	return indexes, nil
}
