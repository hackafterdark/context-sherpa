package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/hackafterdark/context-sherpa/pkg/inference"
	"github.com/hackafterdark/context-sherpa/pkg/sysutils"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	scip "github.com/sourcegraph/scip/bindings/go/scip"
	"google.golang.org/protobuf/proto"
	"gopkg.in/yaml.v3"
)

// HubLock represents the metadata stored in the hub.lock file
type HubLock struct {
	PID       int    `json:"pid"`
	Port      int    `json:"port"`
	StartTime string `json:"startTime"`
}

// GetHubLockPath returns the platform-specific path to the hub.lock file
func GetHubLockPath() string {
	var baseDir string
	if runtime.GOOS == "windows" {
		baseDir = os.Getenv("LOCALAPPDATA")
		if baseDir == "" {
			home, _ := os.UserHomeDir()
			baseDir = home
		}
	} else {
		baseDir, _ = os.UserHomeDir()
	}

	dir := filepath.Join(baseDir, "context-sherpa")
	if runtime.GOOS != "windows" {
		dir = filepath.Join(baseDir, ".context-sherpa")
	}

	_ = os.MkdirAll(dir, 0755)
	return filepath.Join(dir, "hub.lock")
}

// SgConfig represents the structure of sgconfig.yml
type SgConfig struct {
	ID       string   `yaml:"id"`
	Language string   `yaml:"language"`
	RuleDirs []string `yaml:"ruleDirs"`
}

// UserPreferences represents persistent user settings
type UserPreferences struct {
	Theme             string `json:"theme"`
	WindowWidth       int    `json:"windowWidth"`
	WindowHeight      int    `json:"windowHeight"`
	WindowX           int    `json:"windowX"`
	WindowY           int    `json:"windowY"`
	IsMaximized       bool   `json:"isMaximized"`
	InferenceProvider string `json:"inferenceProvider"`
	InferenceURL      string `json:"inferenceURL"`
	InferenceModel    string `json:"inferenceModel"`
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

// sessionWorkspaceRoot stores the auto-discovered workspace root for the current session
var sessionWorkspaceRoot string

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

// normalizeDriveLetter ensures that Windows drive letters are consistently uppercase.
func normalizeDriveLetter(path string) string {
	if runtime.GOOS == "windows" && len(path) > 1 && path[1] == ':' {
		return strings.ToUpper(string(path[0])) + path[1:]
	}
	return path
}

// ensureForwardSlashes replaces all backslashes with forward slashes for cross-platform consistency.
func ensureForwardSlashes(path string) string {
	return strings.ReplaceAll(path, "\\", "/")
}

// toScipPath transforms an input path into a SCIP-compatible relative path.
// It ensures that relative input paths are resolved against the workspaceRoot.
func toScipPath(workspaceRoot, inputPath string) string {
	// Normalize inputPath to use forward slashes for consistency across platforms
	inputPath = ensureForwardSlashes(inputPath)

	// 1. Force Root to Absolute, Cleaned, and Normalized
	absRoot, err := filepath.Abs(workspaceRoot)
	if err == nil {
		absRoot = normalizeDriveLetter(filepath.Clean(absRoot))
	} else {
		absRoot = normalizeDriveLetter(filepath.Clean(workspaceRoot))
	}

	// 2. Resolve inputPath relative to workspaceRoot if it's not absolute
	fullPath := inputPath
	if !filepath.IsAbs(inputPath) {
		fullPath = filepath.Join(absRoot, inputPath)
	}
	fullPath = normalizeDriveLetter(filepath.Clean(fullPath))

	// 3. Force to Absolute and Normalize for comparison
	absPath, err := filepath.Abs(fullPath)
	if err != nil {
		return ensureForwardSlashes(inputPath)
	}
	absPath = normalizeDriveLetter(absPath)

	// 4. Relativize
	rel, err := filepath.Rel(absRoot, absPath)
	res := ""
	if err != nil {
		res = ensureForwardSlashes(inputPath)
	} else {
		res = ensureForwardSlashes(rel)
	}

	return res
}


// Start initializes and starts the MCP server.
func Start(workspaceRoot string, verbose bool, logFilePath string, astGrepPath string, clientName string) {
	if workspaceRoot == "" {
		// Attempt to auto-discover workspace root starting from current working directory
		if root, err := findWorkspaceRoot("", ""); err == nil {
			workspaceRoot = root
			if customLogger != nil {
				customLogger.Printf("Auto-discovered workspace root: %s", workspaceRoot)
			}
		}
	} else {
		// Only override if explicitly passed via argument
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


	s.AddTool(mcp.NewTool("query_local_reasoning",
		mcp.WithDescription("(The Fallback) A catch-all tool for asking any open-ended semantic question about a code snippet that doesn't fit a specific template."),
		mcp.WithString("prompt", mcp.Required(), mcp.Description("The question or instruction for the local SLM to reason about.")),
		mcp.WithString("modelId", mcp.Description("Optional model ID to use. Defaults to currently active.")),
		mcp.WithNumber("max_tokens", mcp.Description("Optional maximum number of tokens to generate. Defaults to 512.")),
		mcp.WithNumber("temperature", mcp.Description("Optional sampling temperature (0.0 to 1.0). Defaults to 0.1.")),
	), queryLocalReasoningHandler)

	s.AddTool(mcp.NewTool("list_local_models",
		mcp.WithDescription("List available models from the configured local inference engine (Ollama/LM Studio)."),
	), listLocalModelsHandler)

	s.AddTool(mcp.NewTool("switch_local_model",
		mcp.WithDescription("Changes the preferred model in the Hub settings."),
		mcp.WithString("modelId", mcp.Required(), mcp.Description("ID of the model to set as default.")),
	), switchLocalModelHandler)

	s.AddTool(mcp.NewTool("pull_inference_model",
		mcp.WithDescription("Requests the local inference engine (Ollama/LM Studio) to download a new model."),
		mcp.WithString("modelId", mcp.Required(), mcp.Description("ID of the model to pull (e.g. 'qwen2.5:0.5b' or full GGUF path).")),
	), pullInferenceModelHandler)

	s.AddTool(mcp.NewTool("classify_repo_intent",
		mcp.WithDescription("(The Router) Determines which Sherpa tool (Symbolic, Structural, or Semantic) is best suited for a user's high-level query."),
		mcp.WithString("query", mcp.Required(), mcp.Description("The high-level user query to classify.")),
	), classifyRepoIntentHandler)

	s.AddTool(mcp.NewTool("summarize_code_intent",
		mcp.WithDescription("Distills raw code into a 3-sentence functional summary (Inputs, Outputs, Side-effects)."),
		mcp.WithString("code", mcp.Required(), mcp.Description("The raw code string to summarize.")),
	), summarizeCodeIntentHandler)

	s.AddTool(mcp.NewTool("generate_structural_pattern",
		mcp.WithDescription("Translates natural language into a valid ast-grep S-expression or pattern."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Natural language description of the structural pattern to generate.")),
	), generateStructuralPatternHandler)

	s.AddTool(mcp.NewTool("analyze_impact_triage",
		mcp.WithDescription("Scans a list of SCIP references to identify which call sites are most likely to be affected by a change."),
		mcp.WithString("references", mcp.Required(), mcp.Description("The list of SCIP references to triage by risk level.")),
	), analyzeImpactTriageHandler)

	s.AddTool(mcp.NewTool("check_rule_compliance",
		mcp.WithDescription("Validates a code snippet against the project-specific rules/ directory."),
		mcp.WithString("code", mcp.Required(), mcp.Description("The code snippet to validate.")),
		mcp.WithString("rule_id", mcp.Required(), mcp.Description("The specific Rule ID to verify compliance against.")),
	), checkRuleComplianceHandler)

	// --- Workspace Initialization ---
	// Attempt to resolve workspace root early for local state and registration
	resolvedRoot, err := findWorkspaceRoot("", "")
	if err == nil {
		// SAFETY: NEVER register a workspace if it's a marker-less system folder.
		// findWorkspaceRoot already checks this, but we reinforce it here.
		if !isSystemDir(resolvedRoot) {
			// 1. Initialize local state (.context-sherpa/ folder)
			if err := initLocalState(resolvedRoot); err != nil {
				customLogger.Printf("Warning: Failed to initialize local state: %v\n", err)
			}

			// 2. Register with Hub (Master Hub Ping)
			go registerWithHub(resolvedRoot, clientName)
		} else {
			customLogger.Printf("Warning: Workspace root '%s' ignored (system/application folder without markers)\n", resolvedRoot)
		}
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



	// Add list_symbols_in_file tool
	listSymbolsInFileTool := mcp.NewTool("list_symbols_in_file",
		mcp.WithDescription("Lists symbols in a file with enriched metadata. Limited to 50 definitions with truncated docstrings. Can be 'distilled' into a semantic summary."),
		mcp.WithString("file_path",
			mcp.Required(),
			mcp.Description("File path relative to workspace root (e.g., 'pkg/mcp/server.go')."),
		),
		mcp.WithBoolean("distill",
			mcp.Description("If true, uses a local SLM to provide a categorized 'Table of Contents' summary."),
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

	// Add ast_grep_scan tool
	astGrepScanTool := mcp.NewTool("ast_grep_scan",
		mcp.WithDescription("Perform a structural search across the codebase using ast-grep patterns. Use this to find specific code shapes (e.g., all functions with a specific decorator) without the noise of text-based grep."),
		mcp.WithString("pattern",
			mcp.Required(),
			mcp.Description("The ast-grep pattern (e.g., func ($$$) $NAME($$$) { $$$ })."),
		),
		mcp.WithString("path",
			mcp.Description("The directory or file to scan (defaults to project root)."),
		),
		mcp.WithString("language",
			mcp.Description("Language hint (e.g., go, typescript)."),
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
	s.AddTool(astGrepScanTool, astGrepScanHandler)

	// Test ast-grep binary and log version information
	sgPath, err := findAstGrepBinary(astGrepPathOverride)
	if err != nil {
		customLogger.Printf("Failed to find ast-grep binary: %v\n", err)
		// Don't exit here - let the MCP tools handle the error when actually used
	} else {
		// Log ast-grep version for debugging and verification
		versionCmd := sysutils.SilentCommand(sgPath, "--version")
		if versionOutput, err := versionCmd.Output(); err == nil {
			customLogger.Printf("Using ast-grep: %s", strings.TrimSpace(string(versionOutput)))
		} else {
			customLogger.Printf("Warning: Could not get ast-grep version: %v", err)
		}
	}

	customLogger.Println("Starting MCP server...")

	// Wrap stdin with a SniffingReader to capture the rootUri from the handshake
	// non-blockingly during the session.
	sniffingStdin := &SniffingReader{r: os.Stdin}

	// Manual transport setup to allow custom stdin (sniffing)
	stdioSvr := server.NewStdioServer(s)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Signal handling for graceful shutdown (standard MCP pattern)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sigChan
		cancel()
	}()

	// Use our sniffing stdin
	if err := stdioSvr.Listen(ctx, sniffingStdin, os.Stdout); err != nil {
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
	// SAFETY: Absolutely block registration of system folders.
	if isSystemDir(workspaceRoot) {
		verboseLog("registerWithHub: blocking registration of system directory: %s", workspaceRoot)
		return
	}

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

	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg, "")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v. Hint: Check if the .context-sherpa folder exists in your project root.", err)), nil
	}

	indexes, err := loadSCIPIndexes(workspaceRoot)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to load SCIP indexes: %v. Hint: Check if the .context-sherpa folder exists in your project root.", err)), nil
	}
	if len(indexes) == 0 {
		return mcp.NewToolResultError("SCIP index not found. Please index the workspace first via the Dashboard. Hint: Check if the .context-sherpa folder exists in your project root."), nil
	}

	// Search for symbol across all indexes
	result := map[string]interface{}{
		"symbol":     symbolName,
		"definition": nil,
		"references": []map[string]interface{}{},
	}

	for _, index := range indexes {
		for _, doc := range index.Documents {
			rel := ensureForwardSlashes(doc.RelativePath)
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
			cmd = sysutils.SilentCommand("tasklist", "/FI", fmt.Sprintf("PID eq %d", ppid), "/FO", "CSV", "/NH")
		} else {
			cmd = sysutils.SilentCommand("ps", "-p", fmt.Sprintf("%d", ppid), "-o", "comm=")
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

	distill := false
	workspaceRootArg := ""
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if d, ok := args["distill"].(bool); ok {
			distill = d
		}
		if wr, ok := args["workspaceRoot"].(string); ok && wr != "" {
			workspaceRootArg = wr
		}
	}

	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg, filePath)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v. Hint: Check if the .context-sherpa folder exists in your project root.", err)), nil
	}

	indexes, err := loadSCIPIndexes(workspaceRoot)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to load SCIP indexes: %v. Hint: Check if the .context-sherpa folder exists in your project root.", err)), nil
	}
	if len(indexes) == 0 {
		return mcp.NewToolResultError("SCIP index not found. Please index the workspace first via the Dashboard. Hint: Check if the .context-sherpa folder exists in your project root."), nil
	}
	verboseLog("listSymbolsInFileHandler: loaded %d indexes", len(indexes))

	// Normalize input path to match SCIP's internal format
	inputPath := toScipPath(workspaceRoot, filePath)
	verboseLog("listSymbolsInFileHandler: searching for '%s' (inputPath), original: '%s'", inputPath, filePath)

	var symbols []map[string]interface{}
	symbolCount := 0
	const maxSymbols = 50

	for idxNum, index := range indexes {
		verboseLog("Searching index %d, documents: %d", idxNum, len(index.Documents))
		for _, doc := range index.Documents {
			rel := ensureForwardSlashes(doc.RelativePath)

			if rel == inputPath {
				verboseLog("listSymbolsInFileHandler: matched file '%s' in index %d", rel, idxNum)

				// Map symbols from both ExternalSymbols and Document symbols for lookup
				symbolMap := make(map[string]*scip.SymbolInformation)
				for _, sym := range index.ExternalSymbols {
					symbolMap[sym.Symbol] = sym
				}
				for _, sym := range doc.Symbols {
					symbolMap[sym.Symbol] = sym
				}

				for _, occ := range doc.Occurrences {
					// Stop if we hit the symbol budget
					if symbolCount >= maxSymbols {
						break
					}

					// Only include definitions (Role 1)
					isDef := occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0

					if isDef {
						displayName := ""
						kind := ""

						parsed, err := scip.ParseSymbol(occ.Symbol)
						if err == nil && len(parsed.Descriptors) > 0 {
							// The last descriptor is the leaf (Method, Type, Term, etc.)
							leaf := parsed.Descriptors[len(parsed.Descriptors)-1]

							// Skip Namespaces (Packages) as definitions
							if leaf.Suffix == scip.Descriptor_Namespace {
								continue
							}
							displayName = leaf.Name

							// Determine kind from SCIP descriptor
							switch leaf.Suffix {
							case scip.Descriptor_Type:
								kind = "Struct/Type"
							case scip.Descriptor_Term:
								kind = "Variable/Field"
							case scip.Descriptor_Method:
								kind = "Method"
							case scip.Descriptor_Macro:
								kind = "Macro"
							case scip.Descriptor_Parameter:
								kind = "Parameter"
							}
						}

						// Fallback if parsing failed or didn't yield a name
						if displayName == "" {
							displayName = occ.Symbol
							// Handle scip-go symbols which have space-separated parts
							if idx := strings.LastIndex(displayName, " "); idx != -1 {
								displayName = displayName[idx+1:]
							}
							if idx := strings.LastIndex(displayName, "/"); idx != -1 {
								displayName = displayName[idx+1:]
							}
							// Strip SCIP suffixes
							displayName = strings.TrimSuffix(displayName, ".")
							displayName = strings.TrimSuffix(displayName, "#")
							displayName = strings.TrimSuffix(displayName, "()")
							displayName = strings.TrimSuffix(displayName, ":")
							if strings.HasPrefix(displayName, "`") && strings.HasSuffix(displayName, "`") {
								displayName = displayName[1 : len(displayName)-1]
							}
						}

						symbolInfo := map[string]interface{}{
							"name":   displayName,
							"symbol": occ.Symbol,
							"line":   occ.Range[0] + 1,
						}

						// Metadata enrichment if details available in index
						if sym, ok := symbolMap[occ.Symbol]; ok {
							if sym.Kind != scip.SymbolInformation_UnspecifiedKind {
								kind = sym.Kind.String()
							}

							// Documentation Truncation: 120 chars, max 2 lines
							if len(sym.Documentation) > 0 {
								doc := strings.Join(sym.Documentation, "\n")
								lines := strings.Split(doc, "\n")
								maxLines := 2
								if len(lines) < maxLines {
									maxLines = len(lines)
								}
								docSummary := strings.Join(lines[:maxLines], "\n")
								if len(docSummary) > 120 {
									docSummary = docSummary[:117] + "..."
								}
								symbolInfo["documentation"] = docSummary
							}

							// Signature
							if sym.SignatureDocumentation != nil && sym.SignatureDocumentation.Language != "" {
								symbolInfo["signature"] = sym.SignatureDocumentation.Text
							} else if len(sym.Documentation) > 0 {
								// Fallback: Check if first line of doc looks like a signature
								firstLine := strings.TrimSpace(strings.Split(sym.Documentation[0], "\n")[0])
								if strings.HasPrefix(firstLine, "func ") || strings.HasPrefix(firstLine, "type ") || strings.HasPrefix(firstLine, "var ") || strings.HasPrefix(firstLine, "const ") {
									symbolInfo["signature"] = firstLine
								}
							}
						}

						// Final fallback for kind if still empty
						if kind == "" {
							if strings.HasSuffix(occ.Symbol, "().") || strings.HasSuffix(occ.Symbol, "()") {
								kind = "Function/Method"
							} else if strings.HasSuffix(occ.Symbol, "#") {
								kind = "Type/Struct"
							}
						}
						symbolInfo["kind"] = kind

						symbols = append(symbols, symbolInfo)
						symbolCount++
					}
				}
				break
			}
		}
	}

	if len(symbols) == 0 {
		return mcp.NewToolResultText(fmt.Sprintf("No symbols found in file: %s", filePath)), nil
	}

	if distill {
		// Prompt Compression: Condensed Tab-Separated format
		var sb strings.Builder
		sb.WriteString("KIND | NAME | SIGNATURE | DOC\n")
		for _, s := range symbols {
			kind := s["kind"].(string)
			name := s["name"].(string)
			sig := ""
			if v, ok := s["signature"].(string); ok {
				sig = v
			}
			doc := ""
			if v, ok := s["documentation"].(string); ok {
				doc = v
			}
			sb.WriteString(fmt.Sprintf("%s | %s | %s | %s\n", kind, name, sig, strings.ReplaceAll(doc, "\n", " ")))
		}

		prompt := fmt.Sprintf("Analyze the following list of symbols found in file '%s'. Provide a semantic 'Table of Contents' by grouping symbols into logical categories (e.g., API Handlers, State Management, Utility Functions). Include a 1-paragraph overview of the file's primary responsibility.\n\nSymbols:\n%s", filePath, sb.String())
		return runSLM(ctx, req, prompt)
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

	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg, "")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v. Hint: Check if the .context-sherpa folder exists in your project root.", err)), nil
	}

	indexes, err := loadSCIPIndexes(workspaceRoot)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to load SCIP indexes: %v. Hint: Check if the .context-sherpa folder exists in your project root.", err)), nil
	}
	if len(indexes) == 0 {
		return mcp.NewToolResultError("SCIP index not found. Please index the workspace first via the Dashboard. Hint: Check if the .context-sherpa folder exists in your project root."), nil
	}

	var definitions []map[string]interface{}
	for _, index := range indexes {
		for _, doc := range index.Documents {
			rel := ensureForwardSlashes(doc.RelativePath)
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

	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg, "")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v. Hint: Check if the .context-sherpa folder exists in your project root.", err)), nil
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

	if err := IndexWorkspace(workspaceRoot, language); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Indexing failed: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Workspace indexed successfully for %s.", language)), nil
}

// IndexWorkspace indexes the workspace for the given language.
func IndexWorkspace(workspaceRoot string, language string) error {
	// 0. Use canonical absolute path, Cleaned and Uppercased Drive
	if abs, err := filepath.Abs(workspaceRoot); err == nil {
		workspaceRoot = filepath.Clean(abs)
	} else {
		workspaceRoot = filepath.Clean(workspaceRoot)
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

	absOutput := filepath.Join(workspaceRoot, ".context-sherpa", fmt.Sprintf("index-%s.scip", language))
	indexerArgs := []string{}
	indexPath := ""

	if language == "go" {
		indexerName := "scip-go"
		if runtime.GOOS == "windows" {
			indexerName += ".exe"
		}
		indexerPath := indexerName
		localBin := filepath.Join(binDir, indexerName)

		if _, err := os.Stat(localBin); err == nil {
			indexerPath = localBin
		}
		indexPath = indexerPath
		indexerArgs = []string{"--project-root", workspaceRoot, "--repository-root", workspaceRoot, "--output", absOutput}
	} else {
		indexerName := "scip-" + language
		var pathsToTry []string
		if runtime.GOOS == "windows" {
			pathsToTry = append(pathsToTry, filepath.Join(binDir, indexerName+".exe"))
			pathsToTry = append(pathsToTry, filepath.Join(binDir, "node_modules", ".bin", indexerName+".cmd"))
			pathsToTry = append(pathsToTry, filepath.Join(binDir, "node_modules", ".bin", indexerName+".ps1"))
		} else {
			pathsToTry = append(pathsToTry, filepath.Join(binDir, indexerName))
			pathsToTry = append(pathsToTry, filepath.Join(binDir, "node_modules", ".bin", indexerName))
		}

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
		indexerArgs = []string{"index", "--output", absOutput}
	}

	if runtime.GOOS == "windows" {
		ext := strings.ToLower(filepath.Ext(indexPath))
		if ext == ".ps1" {
			verboseLog("Running indexer via powershell: %s %s", indexPath, strings.Join(indexerArgs, " "))
			fullArgs := append([]string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", indexPath}, indexerArgs...)
			cmd = sysutils.SilentCommand("powershell", fullArgs...)
		} else if ext == ".cmd" || ext == ".bat" {
			verboseLog("Running indexer via cmd /c: %s %s", indexPath, strings.Join(indexerArgs, " "))
			fullArgs := append([]string{"/c", indexPath}, indexerArgs...)
			cmd = sysutils.SilentCommand("cmd", fullArgs...)
		} else {
			verboseLog("Running indexer: %s %s", indexPath, strings.Join(indexerArgs, " "))
			// DEBUG: log to stderr
			fmt.Fprintf(os.Stderr, "DEBUG: indexing: dir=%q path=%q args=%q\n", workspaceRoot, indexPath, strings.Join(indexerArgs, " "))
			cmd = sysutils.SilentCommand(indexPath, indexerArgs...)
		}
	} else {
		verboseLog("Running indexer: %s %s", indexPath, strings.Join(indexerArgs, " "))
		// DEBUG: log to stderr
		fmt.Fprintf(os.Stderr, "DEBUG: indexing: dir=%q path=%q args=%q\n", workspaceRoot, indexPath, strings.Join(indexerArgs, " "))
		cmd = sysutils.SilentCommand(indexPath, indexerArgs...)
	}
	cmd.Dir = workspaceRoot
	verboseLog("IndexWorkspace: running command: %s %s (in %s)", indexPath, strings.Join(indexerArgs, " "), workspaceRoot)

	if output, err := cmd.CombinedOutput(); err != nil {
		gomodPath := filepath.Join(workspaceRoot, "go.mod")
		hasGomod := "exists"
		if _, serr := os.Stat(gomodPath); serr != nil {
			hasGomod = "MISSING"
		}
		return fmt.Errorf("indexing failed: %v\nCWD: %q\ngo.mod expected at: %q (%s)\nOutput: %s", err, workspaceRoot, gomodPath, hasGomod, string(output))
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
	workspaceRoot, err = findWorkspaceRoot("", "")
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

	cmd := sysutils.SilentCommand(sgPath, "scan", "--config", resolvedSgconfigPath, tmpfile.Name(), "--json")
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
	workspaceRoot, err = findWorkspaceRoot(path, path)

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

	cmd := sysutils.SilentCommand(sgPath, args...)
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

func astGrepScanHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	pattern, err := req.RequireString("pattern")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("pattern is required: %v", err)), nil
	}

	pathArg := ""
	langArg := ""
	workspaceRootArg := ""
	if args, ok := req.Params.Arguments.(map[string]interface{}); ok {
		if p, ok := args["path"].(string); ok && p != "" {
			pathArg = p
		}
		if l, ok := args["language"].(string); ok && l != "" {
			langArg = l
		}
		if wr, ok := args["workspaceRoot"].(string); ok && wr != "" {
			workspaceRootArg = wr
		}
	}

	// We pass empty startPath to find the project root, using pathArg as a hint for discovery
	workspaceRoot, err := findWorkspaceRoot(workspaceRootArg, pathArg)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to resolve workspace root: %v", err)), nil
	}

	sgPath, err := findAstGrepBinary(astGrepPathOverride)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("ast-grep binary not found: %v", err)), nil
	}

	runScan := func(p string) (string, error) {
		args := []string{"run", "--pattern", p, "--json"}
		if langArg != "" {
			args = append(args, "--lang", langArg)
		}
		scanPath := resolvePathRelativeToWorkspaceRoot(pathArg, workspaceRoot)
		args = append(args, scanPath)

		cmd := sysutils.SilentCommand(sgPath, args...)
		cmd.Dir = workspaceRoot
		output, err := cmd.CombinedOutput()
		if err != nil {
			if len(output) == 0 {
				return "", err
			}
		}
		return string(output), nil
	}

	// 1. Primary Pass
	output, err := runScan(pattern)

	// 2. Multi-Stage Fallback
	patternUsed := pattern
	outputTrimmed := strings.TrimSpace(output)
	if (err != nil || outputTrimmed == "[]" || outputTrimmed == "") && !strings.Contains(pattern, "{") {
		fallbacks := expandToFallbacks(pattern)
		for _, fallback := range fallbacks {
			if fallback == pattern {
				continue
			}
			fallbackOutput, fallbackErr := runScan(fallback)
			fallbackOutputStr := strings.TrimSpace(fallbackOutput)
			if fallbackErr == nil && fallbackOutputStr != "[]" && fallbackOutputStr != "" {
				output = fallbackOutputStr
				patternUsed = fallback
				if verboseLogging && customLogger != nil {
					customLogger.Printf("ast_grep_scan: primary pattern yielded no results, moving to fallback: %s", fallback)
				}
				break
			}
		}

		// Tier 3: Structural YAML Probe
		outputTrimmed = strings.TrimSpace(output)
		if (outputTrimmed == "[]" || outputTrimmed == "") && strings.Contains(pattern, ".") {
			methodName := extractMethodName(pattern)
			if methodName != "" {
				probeLang := langArg
				if probeLang == "" {
					probeLang = "go"
				}

				yamlRule := generateStructuralProbe(methodName, probeLang)

				// Create temporary file
				tmpDir := os.TempDir()
				tmpFile, tmpErr := os.CreateTemp(tmpDir, "sgprobe-*.yml")
				if tmpErr == nil {
					defer os.Remove(tmpFile.Name())
					_, _ = tmpFile.WriteString(yamlRule)
					_ = tmpFile.Sync()
					_ = tmpFile.Close()

					if verboseLogging && customLogger != nil {
						customLogger.Printf("ast_grep_scan: Escalating to Tier 3 Structural Probe: %s", methodName)
						customLogger.Printf("ast_grep_scan: Generated Rule:\n%s", yamlRule)
					}

					// Run scan with ad-hoc rule
					args := []string{"scan", "-r", tmpFile.Name(), "--json"}
					scanPath := resolvePathRelativeToWorkspaceRoot(pathArg, workspaceRoot)
					args = append(args, scanPath)

					cmd := sysutils.SilentCommand(sgPath, args...)
					cmd.Dir = workspaceRoot
					probeOutput, probeErr := cmd.CombinedOutput()
					probeOutputStr := strings.TrimSpace(string(probeOutput))

					if verboseLogging && customLogger != nil {
						customLogger.Printf("ast_grep_scan: Tier 3 probe output (len=%d): %s", len(probeOutputStr), probeOutputStr)
					}

					if probeErr == nil && probeOutputStr != "[]" && probeOutputStr != "" {
						output = probeOutputStr
						patternUsed = "Structural Probe: " + methodName
						output = injectExpansionHint(output, "Structural Probe: [inside: call_expression]")
					}
				}
			}
		}
	}

	if strings.TrimSpace(output) == "" {
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("ast-grep execution failed: %v", err)), nil
		}
		output = "[]"
	}

	// If fallback was used or we want to provide metadata, wrap the result
	// Note: We return raw JSON from ast-grep, so we can either append a metadata object
	// or just return it as is if it's already a valid JSON array.
	// For simplicity and since LLMs expect the array, we'll return it but maybe add a log.
	if patternUsed != pattern && customLogger != nil {
		customLogger.Printf("ast_grep_scan: primary pattern yielded no results, used fallback: %s", patternUsed)
	}

	return mcp.NewToolResultText(output), nil
}

// expandToFallbacks generates a set of fallback patterns to increase match probability for nested calls.
// It prioritizes structural wildcards ($OBJ.Method) over greedy wildcards ($$$.Method).
func expandToFallbacks(pattern string) []string {
	var results []string

	dotIdx := strings.LastIndex(pattern, ".")
	if dotIdx <= 0 {
		return results
	}

	// Find the start of the receiver.
	startIdx := 0
	for i := dotIdx - 1; i >= 0; i-- {
		c := pattern[i]
		if c == ' ' || c == '(' || c == ',' || c == '[' || c == '{' || c == '\t' || c == '\n' {
			startIdx = i + 1
			break
		}
	}

	// Stage 1: Single-node wildcard receiver (most likely to work in Go for a.db.Query)
	// Example: "a.db.Query($$$)" -> "$OBJ.Query($$$)"
	results = append(results, pattern[:startIdx]+"$OBJ"+pattern[dotIdx:])

	// Stage 2: Greedy wildcard receiver
	// Example: "a.db.Query($$$)" -> "$$$.Query($$$)"
	results = append(results, pattern[:startIdx]+"$$$"+pattern[dotIdx:])

	// Stage 3: Direct method match (shallow search)
	// Example: "a.db.Query($$$)" -> "Query($$$)"
	results = append(results, pattern[dotIdx+1:])

	return results
}

// extractMethodName extracts the core method name from a pattern.
// Example: "a.db.Query($$$)" -> "Query"
func extractMethodName(pattern string) string {
	base := pattern
	if idx := strings.Index(pattern, "("); idx != -1 {
		base = pattern[:idx]
	}
	base = strings.TrimSpace(base)

	if idx := strings.LastIndex(base, "."); idx != -1 {
		return base[idx+1:]
	}

	return base
}

// generateStructuralProbe synthesizes a YAML rule that matches a method call regardless of receiver.
func generateStructuralProbe(methodName, language string) string {
	return fmt.Sprintf(`id: structural-fallback-probe
language: %s
rule:
  any:
    - pattern: %s($$$)
    - kind: call_expression
      has:
        field: function
        regex: \.%s$
`, language, methodName, methodName)
}

// injectExpansionHint adds a hint to the JSON results to indicate why they were found.
func injectExpansionHint(jsonStr string, hint string) string {
	var results []interface{}
	if err := json.Unmarshal([]byte(jsonStr), &results); err != nil {
		return jsonStr
	}

	for i := range results {
		if obj, ok := results[i].(map[string]interface{}); ok {
			obj["expansion_hint"] = hint
		}
	}

	bytes, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return jsonStr
	}
	return string(bytes)
}

// findWorkspaceRoot finds the workspace root by searching for sgconfig.yml
// or .git in the requested directory and its parent directories.
func findWorkspaceRoot(startPath string, hintPath string) (string, error) {
	var dir string

	if startPath != "" {
		dir = startPath
		if customLogger != nil {
			customLogger.Printf("findWorkspaceRoot: using startPath: %s", dir)
		}
	} else if sessionWorkspaceRoot != "" {
		if customLogger != nil {
			customLogger.Printf("findWorkspaceRoot: using cached session root: %s", sessionWorkspaceRoot)
		}
		return sessionWorkspaceRoot, nil
	} else if hintPath != "" {
		dir = hintPath
		if customLogger != nil {
			customLogger.Printf("findWorkspaceRoot: using hintPath: %s", dir)
		}
	} else if workspaceRootOverride != "" {
		dir = workspaceRootOverride
		if customLogger != nil {
			customLogger.Printf("findWorkspaceRoot: using workspaceRootOverride: %s", dir)
		}
	} else if envRoot := os.Getenv("MCP_WORKSPACE_ROOT"); envRoot != "" {
		dir = envRoot
		if customLogger != nil {
			customLogger.Printf("findWorkspaceRoot: using MCP_WORKSPACE_ROOT env: %s", dir)
		}
	} else if envRoot := os.Getenv("PROJECT_ROOT"); envRoot != "" {
		dir = envRoot
		if customLogger != nil {
			customLogger.Printf("findWorkspaceRoot: using PROJECT_ROOT env: %s", dir)
		}
	} else {
		dir, _ = os.Getwd()
		if customLogger != nil {
			customLogger.Printf("findWorkspaceRoot: using process CWD: %s", dir)
		}
	}

	// Normalize path
	if abs, err := filepath.Abs(dir); err == nil {
		dir = normalizeDriveLetter(filepath.Clean(abs))
	} else {
		dir = normalizeDriveLetter(filepath.Clean(dir))
	}

	// 0. If we ALREADY have a sessionWorkspaceRoot and the dir is relative or within it, use it.
	if sessionWorkspaceRoot != "" {
		if !filepath.IsAbs(dir) || strings.HasPrefix(strings.ToLower(ensureForwardSlashes(dir)), strings.ToLower(ensureForwardSlashes(sessionWorkspaceRoot))) {
			if customLogger != nil {
				customLogger.Printf("findWorkspaceRoot: using existing session root for relative/nested path: %s", sessionWorkspaceRoot)
			}
			return sessionWorkspaceRoot, nil
		}
	}

	// 0. If the resolved dir is a System/Application folder without markers,
	// we should probably NOT trust it as a workspace root.
	isGuaranteedRoot := false
	if startPath != "" {
		// If the user explicitly passed this, we trust it more, but still check if it's a "known bad" CWD
		isGuaranteedRoot = true
	}

	// Higher-priority markers (Sherpa-specific or explicit configs)
	prioMarkers := []string{".context-sherpa", "sgconfig.yml"}
	if root, err := findRootByMarkers(dir, prioMarkers); err == nil {
		// Cache if safe
		if !isSystemDir(root) && sessionWorkspaceRoot == "" {
			sessionWorkspaceRoot = root
			if customLogger != nil {
				customLogger.Printf("findWorkspaceRoot: cached discovered root: %s", root)
			}
		}
		return root, nil
	}

	// Language/Project markers
	langMarkers := []string{
		"go.mod",           // Go
		"package.json",     // JS/TS
		"pyproject.toml",   // Python (modern)
		"requirements.txt", // Python (classic)
		"setup.py",         // Python (classic)
		"Cargo.toml",       // Rust
		"composer.json",    // PHP
		"Gemfile",          // Ruby
		"Makefile",         // Generic Build
	}
	if root, err := findRootByMarkers(dir, langMarkers); err == nil {
		// Cache if safe
		if !isSystemDir(root) && sessionWorkspaceRoot == "" {
			sessionWorkspaceRoot = root
			if customLogger != nil {
				customLogger.Printf("findWorkspaceRoot: cached discovered root: %s", root)
			}
		}
		return root, nil
	}

	// VCS Markers (VCS is often a reliable root)
	vcsMarkers := []string{".git", ".hg", ".svn"}

	// Try git rev-parse first as a fast-path for git
	cmd := sysutils.SilentCommand("git", "rev-parse", "--show-toplevel")
	cmd.Dir = dir
	if gitOutput, err := cmd.Output(); err == nil {
		gitRoot := strings.TrimSpace(string(gitOutput))
		if gitRoot != "" {
			// Basic normalization for git output
			if abs, err := filepath.Abs(gitRoot); err == nil {
				gitRoot = filepath.Clean(abs)
			}
			if runtime.GOOS == "windows" && len(gitRoot) > 1 && gitRoot[1] == ':' {
				gitRoot = strings.ToUpper(string(gitRoot[0])) + gitRoot[1:]
			}
			// Cache if safe
			if !isSystemDir(gitRoot) && sessionWorkspaceRoot == "" {
				sessionWorkspaceRoot = gitRoot
				if customLogger != nil {
					customLogger.Printf("findWorkspaceRoot: cached discovered git root: %s", gitRoot)
				}
			}
			return gitRoot, nil
		}
	}

	// Fallback to manual VCS discovery
	if root, err := findRootByMarkers(dir, vcsMarkers); err == nil {
		// Cache if safe
		if !isSystemDir(root) && sessionWorkspaceRoot == "" {
			sessionWorkspaceRoot = root
			if customLogger != nil {
				customLogger.Printf("findWorkspaceRoot: cached discovered vcs root: %s", root)
			}
		}
		return root, nil
	}

	// Last resort fallback: Use the requested/discovered dir (but ONLY if it's not a system dir)
	if !isSystemDir(dir) {
		verboseLog("No workspace anchors found. Falling back to: %s", dir)
		// Cache if safe
		if sessionWorkspaceRoot == "" {
			sessionWorkspaceRoot = dir
			verboseLog("findWorkspaceRoot: cached fallback root: %s", dir)
		}
		return dir, nil
	}

	// If it IS a system dir and we got here, it means NO markers were found.
	// We refuse to treat a marker-less system folder as a workspace root.
	if isGuaranteedRoot {
		return "", fmt.Errorf("the specified path '%s' looks like a system/application directory and contains no project markers (.git, go.mod, package.json, etc.). Registration blocked for safety.", dir)
	}

	return "", fmt.Errorf("could not discover workspace root. Please provide an explicit 'workspaceRoot' argument or ensure you are running from within a project directory")
}

// findRootByMarkers searches upwards from startDir for any of the given markers.
func findRootByMarkers(startDir string, markers []string) (string, error) {
	tempDir := startDir
	for {
		for _, marker := range markers {
			markerPath := filepath.Join(tempDir, marker)
			if _, err := os.Stat(markerPath); err == nil {
				verboseLog("Found workspace root via marker '%s' at: %s", marker, tempDir)
				return tempDir, nil
			}
		}

		parentDir := filepath.Dir(tempDir)
		if parentDir == tempDir {
			break
		}
		tempDir = parentDir
	}
	return "", fmt.Errorf("no markers found")
}

// isSystemDir returns true if the path looks like a system or application installation directory.
func isSystemDir(path string) bool {
	p := ensureForwardSlashes(path)
	pLower := strings.ToLower(p)

	switch runtime.GOOS {
	case "windows":
		windowsSystemPaths := []string{
			"appdata/local/programs",
			"program files",
			"windows/system32",
			"windows/winsxs",
		}
		for _, pat := range windowsSystemPaths {
			if strings.Contains(pLower, pat) {
				// Special case: if it contains "antigravity", it's likely the app installation folder
				if strings.Contains(pLower, "antigravity") {
					return true
				}
			}
		}
	case "darwin":
		macSystemPaths := []string{
			"/Applications",
			"/Library",
			"/System",
			"/usr/bin",
			"/bin",
			"/sbin",
		}
		for _, pat := range macSystemPaths {
			if strings.HasPrefix(p, pat) {
				return true
			}
		}
	case "linux":
		linuxSystemPaths := []string{
			"/usr/bin",
			"/usr/local/bin",
			"/bin",
			"/sbin",
			"/var/lib",
			"/snap",
		}
		for _, pat := range linuxSystemPaths {
			if strings.HasPrefix(p, pat) {
				return true
			}
		}
	}

	return false
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

// RemoveLocalRule removes a specific rule from a configuration's rule directory.
func RemoveLocalRule(configPath string, ruleID string) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("error reading config file: %v", err)
	}

	var config SgConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("error parsing config file: %v", err)
	}

	if len(config.RuleDirs) == 0 {
		return fmt.Errorf("no ruleDirs specified in config: %s", configPath)
	}

	configDir := filepath.Dir(configPath)
	// Try each rule directory until we find and remove the rule
	for _, rd := range config.RuleDirs {
		ruleFile := filepath.Join(configDir, rd, ruleID+".yml")
		if _, err := os.Stat(ruleFile); err == nil {
			return os.Remove(ruleFile)
		}
	}

	return fmt.Errorf("rule '%s' not found in any rule directories of %s", ruleID, configPath)
}

// InitializeAstGrepConfig sets up a new ast-grep configuration in the target directory.
func InitializeAstGrepConfig(directory string, language string) error {
	// Create rules directory
	rulesDir := filepath.Join(directory, "rules")
	if err := os.MkdirAll(rulesDir, 0755); err != nil {
		return fmt.Errorf("error creating rules directory: %v", err)
	}

	// Create default sgconfig.yml
	sgconfigPath := filepath.Join(directory, "sgconfig.yml")
	if _, err := os.Stat(sgconfigPath); os.IsNotExist(err) {
		config := SgConfig{
			ID:       filepath.Base(directory) + "-config",
			Language: language,
			RuleDirs: []string{"rules"},
		}
		data, err := yaml.Marshal(config)
		if err != nil {
			return fmt.Errorf("error creating sgconfig.yml: %v", err)
		}
		if err := os.WriteFile(sgconfigPath, data, 0644); err != nil {
			return fmt.Errorf("error writing sgconfig.yml: %v", err)
		}
	} else {
		return fmt.Errorf("sgconfig.yml already exists in %s", directory)
	}

	return nil
}

// GetWorkspaceConfigs recursively searches for all sgconfig.yml files in the workspace.
func GetWorkspaceConfigs(workspaceRoot string) ([]map[string]interface{}, error) {
	var configs []map[string]interface{}

	err := filepath.WalkDir(workspaceRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() {
			// Skip common ignored directories
			name := d.Name()
			if name == ".git" || name == "node_modules" || name == "vendor" || name == ".context-sherpa" {
				return filepath.SkipDir
			}
			return nil
		}

		if d.Name() == "sgconfig.yml" {
			data, err := os.ReadFile(path)
			if err != nil {
				return nil // Skip files we can't read
			}

			var config SgConfig
			if err := yaml.Unmarshal(data, &config); err != nil {
				return nil // Skip malformed configs
			}

			// Add metadata for frontend
			configs = append(configs, map[string]interface{}{
				"id":        config.ID,
				"language":  config.Language,
				"path":      path,
				"directory": filepath.Dir(path),
				"ruleDirs":  config.RuleDirs,
			})
		}

		return nil
	})

	return configs, err
}

// FetchCommunityRuleIndex fetches and caches the community rule index
func FetchCommunityRuleIndex() (*CommunityRuleIndex, error) {
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

// FetchCommunityRuleContent fetches the YAML content for a community rule
func FetchCommunityRuleContent(ruleID string) (*CommunityRule, string, error) {
	// Fetch the community rule index
	index, err := FetchCommunityRuleIndex()
	if err != nil {
		return nil, "", fmt.Errorf("failed to fetch community rules: %v", err)
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
		return nil, "", fmt.Errorf("rule '%s' not found in community repository", ruleID)
	}

	// Fetch the actual rule YAML content
	ruleURL := fmt.Sprintf("https://raw.githubusercontent.com/hackafterdark/context-sherpa-community-rules/main/%s", foundRule.Path)
	resp, err := http.Get(ruleURL)
	if err != nil {
		return foundRule, "", fmt.Errorf("failed to fetch rule content: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return foundRule, "", fmt.Errorf("failed to fetch rule content: HTTP %d", resp.StatusCode)
	}

	// Read the YAML content
	var buf []byte
	chunkSize := 1024
	buffer := make([]byte, chunkSize)
	for {
		n, err := resp.Body.Read(buffer)
		if n > 0 {
			buf = append(buf, buffer[:n]...)
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return foundRule, "", fmt.Errorf("failed to read rule content: %v", err)
		}
	}

	return foundRule, string(buf), nil
}

// ValidateAstGrepRule checks if the given YAML content is a valid ast-grep rule.
// It ensures the YAML is well-formed and contains the essential fields 'id', 'language', and 'rule'.
func ValidateAstGrepRule(yamlContent string) error {
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


func runSLM(ctx context.Context, request mcp.CallToolRequest, prompt string) (*mcp.CallToolResult, error) {
	modelID := ""
	maxTokens := 512
	temperature := float32(0.1)

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

	// Read preferences directly from file to get the latest settings
	// (MCP server runs in a separate process from the Hub)
	prefs, err := loadPreferencesManually()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error loading preferences: %v", err)), nil
	}

	provider, _, err := getInferenceProvider()
	if err != nil {
		return nil, err
	}

	if provider == nil {
		return mcp.NewToolResultError("No inference provider configured. Please set up Ollama or LM Studio in the Hub settings."), nil
	}

	// Use preferred model if client didn't specify one
	finalModelID := modelID
	if finalModelID == "" {
		finalModelID = prefs.InferenceModel
	}

	svc := inference.NewInferenceService(provider)
	res, err := svc.Execute(ctx, inference.InferenceRequest{
		ModelID:     finalModelID,
		Prompt:      prompt,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	})

	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Inference failed: %v", err)), nil
	}

	return mcp.NewToolResultText(res.Text), nil
}

// loadPreferencesManually reads the preferences.json file directly.
// This is necessary because the MCP server is a separate binary.
func loadPreferencesManually() (UserPreferences, error) {
	var baseDir string
	if runtime.GOOS == "windows" {
		baseDir = os.Getenv("LOCALAPPDATA")
		if baseDir == "" {
			home, _ := os.UserHomeDir()
			baseDir = home
		}
	} else {
		baseDir, _ = os.UserHomeDir()
	}

	dir := filepath.Join(baseDir, "context-sherpa")
	if runtime.GOOS != "windows" {
		dir = filepath.Join(baseDir, ".context-sherpa")
	}

	prefsPath := filepath.Join(dir, "preferences.json")
	data, err := os.ReadFile(prefsPath)
	if err != nil {
		return UserPreferences{}, err
	}

	var prefs UserPreferences
	if err := json.Unmarshal(data, &prefs); err != nil {
		return UserPreferences{}, err
	}

	return prefs, nil
}

func listLocalModelsHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	provider, _, err := getInferenceProvider()
	if err != nil {
		return nil, err
	}
	if provider == nil {
		return mcp.NewToolResultError("Inference is disabled or not configured."), nil
	}

	models, err := provider.ListModels(ctx)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to list models: %v", err)), nil
	}

	if len(models) == 0 {
		return mcp.NewToolResultText("No models found in the local inference engine."), nil
	}

	var sb strings.Builder
	sb.WriteString("Available Models:\n")
	for _, m := range models {
		sb.WriteString(fmt.Sprintf("- %s\n", m))
	}
	return mcp.NewToolResultText(sb.String()), nil
}

func switchLocalModelHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	modelID, err := request.RequireString("modelId")
	if err != nil {
		return mcp.NewToolResultError("modelId is required"), nil
	}

	prefs, err := loadPreferencesManually()
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error loading preferences: %v", err)), nil
	}

	prefs.InferenceModel = modelID
	if err := savePreferencesManually(prefs); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Error saving preferences: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Successfully updated default model to: %s", modelID)), nil
}

func pullInferenceModelHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	modelID, err := request.RequireString("modelId")
	if err != nil {
		return mcp.NewToolResultError("modelId is required"), nil
	}

	provider, _, err := getInferenceProvider()
	if err != nil {
		return nil, err
	}
	if provider == nil {
		return mcp.NewToolResultError("Inference is disabled or not configured."), nil
	}

	if err := provider.PullModel(ctx, modelID); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Failed to pull model: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("Successfully initiated pull for model: %s. This may take some time depending on the model size.", modelID)), nil
}

func queryLocalReasoningHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	prompt, err := request.RequireString("prompt")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("prompt is required: %v", err)), nil
	}
	return runSLM(ctx, request, prompt)
}

func getInferenceProvider() (inference.InferenceProvider, UserPreferences, error) {
	prefs, err := loadPreferencesManually()
	if err != nil {
		return nil, UserPreferences{}, fmt.Errorf("error loading preferences: %v", err)
	}

	var provider inference.InferenceProvider
	switch prefs.InferenceProvider {
	case "ollama":
		url := prefs.InferenceURL
		if url == "" {
			url = "http://localhost:11434"
		}
		provider = inference.NewOllamaProvider(url)
	case "openai":
		url := prefs.InferenceURL
		if url == "" {
			url = "http://localhost:1234/v1"
		}
		provider = inference.NewOpenAIProvider(url)
	case "lmstudio":
		url := prefs.InferenceURL
		if url == "" {
			url = "http://localhost:1234/api/v1"
		}
		provider = inference.NewLMStudioProvider(url)
	case "disabled":
		return nil, prefs, nil
	default:
		return nil, prefs, nil
	}
	return provider, prefs, nil
}

func savePreferencesManually(prefs UserPreferences) error {
	var baseDir string
	if runtime.GOOS == "windows" {
		baseDir = os.Getenv("LOCALAPPDATA")
		if baseDir == "" {
			home, _ := os.UserHomeDir()
			baseDir = home
		}
	} else {
		baseDir, _ = os.UserHomeDir()
	}

	dir := filepath.Join(baseDir, "context-sherpa")
	if runtime.GOOS != "windows" {
		dir = filepath.Join(baseDir, ".context-sherpa")
	}

	prefsPath := filepath.Join(dir, "preferences.json")
	data, err := json.MarshalIndent(prefs, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(prefsPath, data, 0644)
}

func classifyRepoIntentHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query, err := request.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("query is required: %v", err)), nil
	}
	prompt := fmt.Sprintf("Is the user asking for a symbol, a structural pattern, or logic analysis?\nQuery: %s", query)
	return runSLM(ctx, request, prompt)
}

func summarizeCodeIntentHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	code, err := request.RequireString("code")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("code is required: %v", err)), nil
	}
	prompt := fmt.Sprintf("Provide a high-density functional distillation (Inputs, Outputs, Side-effects) in 3 sentences for the following code:\n%s", code)
	return runSLM(ctx, request, prompt)
}

func generateStructuralPatternHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query, err := request.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("query is required: %v", err)), nil
	}
	prompt := fmt.Sprintf("Translate the following natural language into a valid ast-grep S-expression or pattern. Use few-shot examples of structural syntax if known.\nQuery: %s", query)
	return runSLM(ctx, request, prompt)
}

func analyzeImpactTriageHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	references, err := request.RequireString("references")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("references is required: %v", err)), nil
	}
	prompt := fmt.Sprintf("Triage these references by risk level. Identify which call sites are most likely to be affected by a change:\n%s", references)
	return runSLM(ctx, request, prompt)
}

func checkRuleComplianceHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	code, err := request.RequireString("code")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("code is required: %v", err)), nil
	}
	ruleID, err := request.RequireString("rule_id")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("rule_id is required: %v", err)), nil
	}
	prompt := fmt.Sprintf("Verify compliance with rule [%s] for the following code snippet:\n%s", ruleID, code)
	return runSLM(ctx, request, prompt)
}

// loadSCIPIndexes loads all index-*.scip (and index.scip) files found in .context-sherpa directories
// recursively within the workspaceRoot.
func loadSCIPIndexes(workspaceRoot string) ([]*scip.Index, error) {
	var indexes []*scip.Index

	// Normalize workspace root for offset calculation
	absRoot, err := filepath.Abs(workspaceRoot)
	if err == nil {
		if runtime.GOOS == "windows" && len(absRoot) > 1 && absRoot[1] == ':' {
			absRoot = strings.ToUpper(string(absRoot[0])) + absRoot[1:]
		}
	} else {
		absRoot = workspaceRoot
	}

	err = filepath.WalkDir(absRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Only look for .context-sherpa directories
		if d.IsDir() && d.Name() == ".context-sherpa" {
			// Calculate offset from workspace root
			relDir, err := filepath.Rel(absRoot, filepath.Dir(path))
			if err != nil {
				relDir = "."
			}
			if relDir == "." {
				relDir = ""
			}

			verboseLog("Searching for indexes in: %s (rel: '%s')", path, relDir)
			files, err := os.ReadDir(path)
			if err != nil {
				return nil // Continue walking
			}

			for _, file := range files {
				if !file.IsDir() && (strings.HasPrefix(file.Name(), "index-") && strings.HasSuffix(file.Name(), ".scip") || file.Name() == "index.scip") {
					scipPath := filepath.Join(path, file.Name())
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

					// Prefix document paths if this index is in a subdirectory
					if relDir != "" {
						prefix := ensureForwardSlashes(relDir) + "/"
						for _, doc := range index.Documents {
							if !filepath.IsAbs(doc.RelativePath) {
								doc.RelativePath = prefix + ensureForwardSlashes(doc.RelativePath)
							}
						}
					} else {
						// Ensure document paths are consistently slashed even for root index
						for _, doc := range index.Documents {
							doc.RelativePath = ensureForwardSlashes(doc.RelativePath)
						}
					}

					indexes = append(indexes, &index)
				}
			}
			// We can potentially skip walking INTO .context-sherpa itself
			return filepath.SkipDir
		}

		// Skip common ignored directories to speed up walking
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == "node_modules" || name == "vendor" || name == ".gemini" || name == ".roo" {
				return filepath.SkipDir
			}
		}

		return nil
	})

	if err != nil {
		verboseLog("Error walking workspace for SCIP indexes: %v", err)
	}

	return indexes, nil
}

// SniffingReader wraps an io.Reader and looks for the MCP initialize rootUri
type SniffingReader struct {
	r       io.Reader
	buf     []byte
	sniffed bool
}

func (sr *SniffingReader) Read(p []byte) (int, error) {
	n, err := sr.r.Read(p)
	if n > 0 && !sr.sniffed {
		if customLogger != nil {
			customLogger.Printf("SniffingReader: Seen %d bytes. Current buffer length: %d", n, len(sr.buf)+n)
		}
		// Collect up to 16KB for sniffing
		if len(sr.buf) < 16384 {
			sr.buf = append(sr.buf, p[:n]...)
			if customLogger != nil {
				customLogger.Printf("SniffingReader: Buffer content: %s", string(sr.buf))
			}
			if bytes.Contains(sr.buf, []byte("\"rootUri\"")) || bytes.Contains(sr.buf, []byte("\"rootPath\"")) {
				// Found it! Extract
				sr.extractRoot(sr.buf)
				sr.sniffed = true
				sr.buf = nil // Free memory
			}
		} else {
			if customLogger != nil {
				customLogger.Printf("SniffingReader: Sniffing limit reached, giving up. Buffer size: %d", len(sr.buf))
			}
			sr.sniffed = true // Give up sniffing
			sr.buf = nil
		}
	}
	return n, err
}

func (sr *SniffingReader) extractRoot(captured []byte) {
	// Try to find rootUri or rootPath or even uri in params
	re := regexp.MustCompile(`"(rootUri|rootPath|uri)"\s*:\s*"([^"]+)"`)
	match := re.FindSubmatch(captured)
	if len(match) > 1 {
		uri := string(match[2])
		if customLogger != nil {
			customLogger.Printf("SniffingReader: Found matching field %s with value %s", string(match[1]), uri)
		}
		
		path := strings.TrimPrefix(uri, "file://")
		// Windows: file:///C:/path -> /C:/path
		if strings.HasPrefix(path, "/") && len(path) > 2 && path[2] == ':' {
			path = path[1:]
		}
		path = filepath.FromSlash(path)

		// Normalize Windows drive letters
		if runtime.GOOS == "windows" && len(path) > 1 && path[1] == ':' {
			path = strings.ToUpper(string(path[0])) + path[1:]
		}

		sessionWorkspaceRoot = path
		if customLogger != nil {
			customLogger.Printf("SniffingReader: Successfully intercepted handshake! Found rootUri: %s", path)
		}
	} else {
		// Log a bit of the buffer for debugging
		snipLen := 512
		if len(captured) < snipLen {
			snipLen = len(captured)
		}
		if customLogger != nil {
			customLogger.Printf("SniffingReader: No match in buffer snippet: %s", string(captured[:snipLen]))
		}
	}
}
