package main

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/hackafterdark/context-sherpa/pkg/database"
	"github.com/hackafterdark/context-sherpa/pkg/inference"
	"github.com/hackafterdark/context-sherpa/pkg/mcp"
	"github.com/hackafterdark/context-sherpa/pkg/sysutils"
	scip "github.com/sourcegraph/scip/bindings/go/scip"
	wailsRuntime "github.com/wailsapp/wails/v2/pkg/runtime"
	"google.golang.org/protobuf/proto"
	"gopkg.in/yaml.v3"
)

// Workspace represents a detected workspace from a Node
type Workspace struct {
	PID       int    `json:"pid"`
	Root      string `json:"root"`
	Client    string `json:"client"`
	State     string `json:"state"`
	LastSeen  string `json:"lastSeen"`
	IsManaged bool   `json:"isManaged"`
}

// UserPreferences represents persistent user settings
type UserPreferences struct {
	Theme             string `json:"theme"`
	WindowWidth       int    `json:"windowWidth"`
	WindowHeight      int    `json:"windowHeight"`
	WindowX           int    `json:"windowX"`
	WindowY           int    `json:"windowY"`
	IsMaximized       bool   `json:"isMaximized"`
	InferenceProvider string `json:"inferenceProvider"` // "ollama" or "openai"
	InferenceURL      string `json:"inferenceURL"`
	InferenceModel    string `json:"inferenceModel"`
}

// App struct
type App struct {
	ctx        context.Context
	workspaces []Workspace
	isHub      bool
	downloader *inference.Downloader
	inference  *inference.InferenceService
	db         *database.DB
}

// MarkdownEntry represents a markdown file with optional front-matter metadata
type MarkdownEntry struct {
	Path        string            `json:"path"`
	FrontMatter map[string]string `json:"frontMatter"`
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts.
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	a.workspaces = make([]Workspace, 0)

	// Restore window position if not maximized
	prefs := a.GetPreferences()
	if !prefs.IsMaximized && (prefs.WindowX != 0 || prefs.WindowY != 0) {
		wailsRuntime.WindowSetPosition(a.ctx, prefs.WindowX, prefs.WindowY)
	}

	// Ensure we have a clean start by checking lock liveness
	if a.tryAcquireHubLock() {
		a.isHub = true
		fmt.Println("Hub: Successfully acquired hub.lock. Starting as Master Hub.")

		// Initialize Inference services
		configDir, _ := getSherpaConfigDir()
		modelsDir := filepath.Join(configDir, "models")
		a.downloader = inference.NewDownloader(modelsDir)

		// Set up provider based on preferences
		var provider inference.InferenceProvider
		switch prefs.InferenceProvider {
		case "ollama":
			provider = inference.NewOllamaProvider(prefs.InferenceURL)
		case "openai":
			provider = inference.NewOpenAIProvider(prefs.InferenceURL)
		case "disabled":
			// Explicitly disabled
			provider = nil
		default:
			// No provider configured yet or invalid
			provider = nil
		}
		a.inference = inference.NewInferenceService(provider)

		// Initialize Hub Database
		a.initDatabase()

		// Start the Hub's registration server on a background goroutine
		go a.startHubServer()
		go a.startSweeper()
	} else {
		fmt.Println("Hub: Another instance is already Master Hub. Starting as Node viewer.")
		go a.startViewerPoller()
	}
}

func isProcessRunning(pid int) bool {
	if pid <= 0 {
		return false
	}

	// platform specific check
	if runtime.GOOS == "windows" {
		cmd := sysutils.SilentCommand("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH")
		out, err := cmd.Output()
		if err != nil {
			return false
		}
		return strings.Contains(string(out), fmt.Sprintf("%d", pid))
	}

	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	// On Unix, Signal(0) checks for existence
	err = p.Signal(syscall.Signal(0))
	return err == nil
}

func (a *App) tryAcquireHubLock() bool {
	lockPath := mcp.GetHubLockPath()

	// 1. Try to read existing lock
	if data, err := os.ReadFile(lockPath); err == nil {
		var lock mcp.HubLock
		if err := json.Unmarshal(data, &lock); err == nil {
			// Check if process still exists
			if isProcessRunning(lock.PID) {
				// Hub is truly already running
				return false
			}
			fmt.Printf("Hub: Stale lock found (PID %d not running). Overwriting...\n", lock.PID)
		}
	}

	// 2. Try to take the lock
	lock := mcp.HubLock{
		PID:       os.Getpid(),
		Port:      9000,
		StartTime: time.Now().Format(time.RFC3339),
	}
	data, _ := json.MarshalIndent(lock, "", "  ")

	// Create directory if it doesn't exist (GetHubLockPath does this, but being safe)
	_ = os.MkdirAll(filepath.Dir(lockPath), 0755)

	err := os.WriteFile(lockPath, data, 0644)
	return err == nil
}

// normalizePath ensures paths are absolute and have consistent casing for drive letters on Windows
func (a *App) normalizePath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	if runtime.GOOS == "windows" && len(abs) > 1 && abs[1] == ':' {
		// Uppercase drive letter for consistency
		abs = strings.ToUpper(string(abs[0])) + abs[1:]
	}
	return filepath.Clean(abs)
}

func (a *App) initDatabase() {
	configDir, err := getSherpaConfigDir()
	if err != nil {
		fmt.Printf("Hub: Failed to get config dir: %v\n", err)
		return
	}

	dbPath := filepath.Join(configDir, "hub.db")
	a.db, err = database.InitDB(dbPath)
	if err != nil {
		fmt.Printf("Hub: Failed to initialize hub.db: %v\n", err)
		return
	}

	// Create workspaces table
	_, err = a.db.Exec(`
		CREATE TABLE IF NOT EXISTS workspaces (
			root TEXT PRIMARY KEY,
			client TEXT,
			last_seen DATETIME,
			is_managed BOOLEAN DEFAULT 0
		)
	`)
	if err != nil {
		fmt.Printf("Hub: Failed to create workspaces table: %v\n", err)
	}

	// Load existing workspaces into memory
	rows, err := a.db.Query("SELECT root, client, last_seen, is_managed FROM workspaces")
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var ws Workspace
			var lastSeen string
			var isManaged int
			if err := rows.Scan(&ws.Root, &ws.Client, &lastSeen, &isManaged); err == nil {
				ws.Root = a.normalizePath(ws.Root)
				ws.LastSeen = lastSeen
				ws.IsManaged = isManaged == 1
				ws.State = "offline"
				a.workspaces = append(a.workspaces, ws)
			}
		}
	}
}


// RegisterWorkspace manually adds a workspace directory to the Hub's persistent list
func (a *App) RegisterWorkspace(path string) error {
	if !a.isHub {
		return fmt.Errorf("only the Master Hub can register workspaces")
	}

	// 1. Verify path exists
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("invalid path: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("path is not a directory")
	}

	// 2. Canonicalize path
	absPath := a.normalizePath(path)

	// 3. Initialize local state (.context-sherpa folder)
	sherpaDir := filepath.Join(absPath, ".context-sherpa")
	if err := os.MkdirAll(sherpaDir, 0755); err != nil {
		return fmt.Errorf("failed to create .context-sherpa dir: %w", err)
	}

	// 4. Persist to database
	if a.db != nil {
		_, err := a.db.Exec(`
			INSERT INTO workspaces (root, client, last_seen, is_managed)
			VALUES (?, ?, ?, ?)
			ON CONFLICT(root) DO UPDATE SET last_seen = excluded.last_seen, is_managed = 1
		`, absPath, "manual", time.Now().Format(time.RFC3339), 1)
		if err != nil {
			return fmt.Errorf("database error: %w", err)
		}
	}

	// 5. Update in-memory list
	found := false
	for i, ws := range a.workspaces {
		if ws.Root == absPath {
			a.workspaces[i].LastSeen = time.Now().Format(time.RFC3339)
			a.workspaces[i].IsManaged = true
			found = true
			break
		}
	}
	if !found {
		a.workspaces = append(a.workspaces, Workspace{
			Root:      absPath,
			Client:    "manual",
			State:     "offline",
			LastSeen:  time.Now().Format(time.RFC3339),
			IsManaged: true,
		})
	}

	// 6. Notify UI
	wailsRuntime.EventsEmit(a.ctx, "workspace-updated", a.workspaces)

	return nil
}

// ReadMarkdown loads the raw text for MDXEditor.
func (a *App) ReadMarkdown(path string) (string, error) {
	fmt.Printf("Hub: ReadMarkdown requested for path: %s\n", path)
	if !a.isPathInWorkspace(path) {
		fmt.Printf("Hub: ReadMarkdown access denied for path: %s\n", path)
		return "", fmt.Errorf("access denied: path is outside of registered workspaces")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("Hub: ReadMarkdown error for %s: %v\n", path, err)
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	// Safety: Strip null bytes to prevent bridge truncation
	content := strings.ReplaceAll(string(data), "\x00", "")

	fmt.Printf("Hub: ReadMarkdown success, read %d bytes from %s\n", len(data), path)
	return content, nil
}

// WriteMarkdown commits the edits to the workspace.
func (a *App) WriteMarkdown(path string, content string) error {
	fmt.Printf("Hub: WriteMarkdown requested for path: %s\n", path)
	if !a.isPathInWorkspace(path) {
		return fmt.Errorf("access denied: path is outside of registered workspaces")
	}

	// Audit: Ensure path is within workspace boundaries
	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// DiscoverMarkdownFiles recursively scans a workspace for .md files and extracts front-matter.
func (a *App) DiscoverMarkdownFiles(root string) ([]MarkdownEntry, error) {
	if !a.isPathInWorkspace(root) {
		return nil, fmt.Errorf("access denied: path is outside of registered workspaces")
	}

	var results []MarkdownEntry
	// Robust skip list for common non-content directories
	skipDirs := map[string]bool{
		".git":            true,
		".svn":            true,
		".hg":             true,
		".bzr":            true,
		"_darcs":          true,
		".context-sherpa": true,
		".ssh":            true,
		".aws":            true,
		".kube":           true,
		".env":            true,
		"node_modules":    true,
		"vendor":          true,
		"__pycache__":     true,
		".idea":           true,
		".vscode":         true,
		".history":        true,
	}

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			base := filepath.Base(path)
			if skipDirs[base] {
				return filepath.SkipDir
			}
			return nil
		}

		if strings.ToLower(filepath.Ext(path)) == ".md" {
			entry := MarkdownEntry{Path: path}

			// Try to extract front-matter
			if fm, err := a.extractFrontMatter(path); err == nil {
				entry.FrontMatter = fm
			}

			results = append(results, entry)
		}
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to scan for markdown files: %w", err)
	}

	return results, nil
}

// extractFrontMatter reads the beginning of a file and attempts to parse YAML front-matter
func (a *App) extractFrontMatter(path string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read first 4KB (usually enough for front-matter)
	buf := make([]byte, 4096)
	n, err := file.Read(buf)
	if err != nil && err != io.EOF {
		return nil, err
	}
	content := string(buf[:n])

	// Check if it starts with ---
	if !strings.HasPrefix(content, "---\n") && !strings.HasPrefix(content, "---\r\n") {
		return nil, fmt.Errorf("no front-matter prefix")
	}

	// Find the closing ---
	// Start searching after the first ---
	startSearch := 4
	if strings.HasPrefix(content, "---\r\n") {
		startSearch = 5
	}

	endIdx := strings.Index(content[startSearch:], "---")
	if endIdx == -1 {
		return nil, fmt.Errorf("no front-matter closer")
	}

	yamlContent := content[startSearch : startSearch+endIdx]
	var fm map[string]string
	if err := yaml.Unmarshal([]byte(yamlContent), &fm); err != nil {
		return nil, err
	}

	return fm, nil
}

// isPathInWorkspace checks if the given path is within any registered workspace.
func (a *App) isPathInWorkspace(path string) bool {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return false
	}

	// On Windows, drive letters and paths are case-insensitive
	isWindows := runtime.GOOS == "windows"
	if isWindows {
		absPath = strings.ToLower(absPath)
	}

	for _, ws := range a.workspaces {
		wsAbs, err := filepath.Abs(ws.Root)
		if err != nil {
			continue
		}

		if isWindows {
			wsAbs = strings.ToLower(wsAbs)
		}

		// Ensure we're comparing clean directory boundaries
		wsAbs = filepath.Clean(wsAbs)
		if !strings.HasSuffix(wsAbs, string(filepath.Separator)) {
			wsAbs += string(filepath.Separator)
		}

		// Check if absPath is equal to wsAbs (root file) or a child
		if absPath == strings.TrimSuffix(wsAbs, string(filepath.Separator)) || strings.HasPrefix(absPath, wsAbs) {
			return true
		}
	}

	return false
}

// IndexTarget represents a directory and language that should be indexed
type IndexTarget struct {
	Path string
	Lang string
}

// RunIndexingTask triggers the SCIP indexer for a workspace and streams log output.
// It discovers language-specific roots and runs indexers from those locations.
func (a *App) RunIndexingTask(workspacePath string) error {
	if !a.isHub {
		return fmt.Errorf("only the Master Hub can run indexing tasks")
	}

	targets := a.discoverIndexTargets(workspacePath)
	a.emitIndexingLog(workspacePath, fmt.Sprintf("Discovered %d indexing targets.", len(targets)))
	if len(targets) == 0 {
		a.emitIndexingLog(workspacePath, "No indexable code files found (.go, .ts, .py, etc.).")
		return fmt.Errorf("no indexable targets found")
	}

	go func() {
		successCount := 0
		for _, target := range targets {
			a.emitIndexingLog(workspacePath, fmt.Sprintf("Processing %s in %s...", target.Lang, target.Path))

			// Resolve indexer tool
			status := a.GetScipIndexerStatus(target.Lang)
			if !status["installed"].(bool) {
				a.emitIndexingLog(workspacePath, fmt.Sprintf("Error: Indexer for %s not installed. Please visit Settings to install it.", target.Lang))
				continue
			}
			toolPath := status["path"].(string)

			// Prepare command and output path
			// We use relative paths for --output to be safer on Windows
			scipFilename := fmt.Sprintf("index-%s.scip", target.Lang)
			scipRelPath := filepath.Join(".context-sherpa", scipFilename)
			scipAbsPath := filepath.Join(target.Path, scipRelPath)

			_ = os.MkdirAll(filepath.Join(target.Path, ".context-sherpa"), 0755)

			var cmd *exec.Cmd
			indexerArgs := []string{}
			if target.Lang == "go" {
				indexerArgs = []string{"--project-root", ".", "--repository-root", ".", "--output", scipRelPath}
			} else {
				// For scip-typescript, scip-python, etc.
				indexerArgs = []string{"index", "--output", scipRelPath}
			}

			if runtime.GOOS == "windows" {
				ext := strings.ToLower(filepath.Ext(toolPath))
				if ext == ".ps1" {
					// Use powershell for .ps1 files
					fullArgs := append([]string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", toolPath}, indexerArgs...)
					cmd = sysutils.SilentCommand("powershell", fullArgs...)
				} else if ext == ".cmd" || ext == ".bat" {
					// Use cmd /c for .cmd and .bat files
					fullArgs := append([]string{"/c", toolPath}, indexerArgs...)
					cmd = sysutils.SilentCommand("cmd", fullArgs...)
				} else {
					cmd = sysutils.SilentCommand(toolPath, indexerArgs...)
				}
			} else {
				cmd = sysutils.SilentCommand(toolPath, indexerArgs...)
			}
			cmd.Dir = target.Path

			// Log the actual command being run
			actualCmd := cmd.Path
			if len(cmd.Args) > 1 {
				actualCmd += " " + strings.Join(cmd.Args[1:], " ")
			}
			a.emitIndexingLog(workspacePath, fmt.Sprintf("Running: %s", actualCmd))

			// Stream logs
			stdout, _ := cmd.StdoutPipe()
			stderr, _ := cmd.StderrPipe()

			go a.streamLogsToUI(workspacePath, stdout)
			go a.streamLogsToUI(workspacePath, stderr)

			if err := cmd.Run(); err != nil {
				a.emitIndexingLog(workspacePath, fmt.Sprintf("Indexing %s failed in %s: %v", target.Lang, target.Path, err))
			} else {
				// Verify file existence
				if _, err := os.Stat(scipAbsPath); err == nil {
					a.emitIndexingLog(workspacePath, fmt.Sprintf("Indexing %s complete! %s created at %s", target.Lang, scipFilename, scipAbsPath))
					successCount++
				} else {
					a.emitIndexingLog(workspacePath, fmt.Sprintf("Indexing %s finished with success code, but %s was not found at expected path: %s", target.Lang, scipFilename, scipAbsPath))
				}
			}
		}

		wailsRuntime.EventsEmit(a.ctx, "indexing-finished", map[string]interface{}{
			"root":    workspacePath,
			"success": successCount > 0,
			"count":   successCount,
		})
	}()

	return nil
}

type langSignal struct {
	path     string
	strength int // 2 = Strong (config file), 1 = Weak (code file/hint)
}

func (a *App) discoverIndexTargets(root string) []IndexTarget {
	// 1. Walk tree and collect all directories with any language signal
	allSignals := make(map[string]map[string]int) // path -> lang -> strength

	isExcluded := func(path string) bool {
		base := filepath.Base(path)
		return base == "node_modules" || base == ".git" || base == ".context-sherpa" || base == "vendor"
	}

	queue := []string{root}
	for len(queue) > 0 {
		path := queue[0]
		queue = queue[1:]

		if isExcluded(path) {
			continue
		}

		files, err := os.ReadDir(path)
		if err != nil {
			continue
		}

		signals := make(map[string]int)
		for _, f := range files {
			if f.IsDir() {
				queue = append(queue, filepath.Join(path, f.Name()))
				continue
			}
			name := f.Name()
			ext := filepath.Ext(name)

			// Go Signals
			if name == "go.mod" {
				signals["go"] = 2
			} else if ext == ".go" && signals["go"] < 1 {
				signals["go"] = 1
			}

			// TypeScript/JS Signals
			if name == "tsconfig.json" {
				signals["typescript"] = 2
			} else if (name == "package.json" || ext == ".ts" || ext == ".tsx" || ext == ".js" || ext == ".jsx") && signals["typescript"] < 1 {
				signals["typescript"] = 1
			}

			// Python Signals
			if name == "pyproject.toml" || name == "requirements.txt" {
				signals["python"] = 2
			} else if ext == ".py" && signals["python"] < 1 {
				signals["python"] = 1
			}
		}

		if len(signals) > 0 {
			allSignals[path] = signals
		}
	}

	// 2. Filter targets per language
	var targets []IndexTarget
	langs := []string{"go", "typescript", "python"}

	for _, lang := range langs {
		// Identify candidate paths for this language
		var candidates []langSignal
		for p, signals := range allSignals {
			if strength, ok := signals[lang]; ok {
				candidates = append(candidates, langSignal{path: p, strength: strength})
			}
		}

		// Pruning logic:
		// A candidate path P is kept for language L if:
		// - NO ancestor of P has a Strong signal (Strength=2) for L.
		// - AND (If P is Weak (Strength=1), NO descendant of P has a Strong signal for L).
		for _, candidate := range candidates {
			keep := true

			// Check ancestors
			parent := filepath.Dir(candidate.path)
			for {
				if signals, ok := allSignals[parent]; ok {
					if signals[lang] == 2 {
						keep = false
						break
					}
				}
				if parent == root || parent == filepath.Dir(parent) {
					break
				}
				parent = filepath.Dir(parent)
			}

			if !keep {
				continue
			}

			// If Weak, check descendants for Strong signals
			if candidate.strength == 1 {
				for p, signals := range allSignals {
					if signals[lang] == 2 && strings.HasPrefix(p, candidate.path+string(filepath.Separator)) {
						keep = false
						break
					}
				}
			}

			if keep {
				targets = append(targets, IndexTarget{Path: candidate.path, Lang: lang})
			}
		}
	}

	// 3. Final pruning of targets for the same language (keep only highest ancestor among selected targets)
	var finalTargets []IndexTarget
	for _, t := range targets {
		isSub := false
		for _, other := range targets {
			if t.Lang == other.Lang && t.Path != other.Path && strings.HasPrefix(t.Path, other.Path+string(filepath.Separator)) {
				isSub = true
				break
			}
		}
		if !isSub {
			finalTargets = append(finalTargets, t)
		}
	}

	return finalTargets
}

func (a *App) detectLanguage(root string) string {
	targets := a.discoverIndexTargets(root)
	if len(targets) > 0 {
		return targets[0].Lang
	}
	return ""
}

func (a *App) streamLogsToUI(root string, r io.Reader) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		a.emitIndexingLog(root, scanner.Text())
	}
}

func (a *App) emitIndexingLog(root string, message string) {
	wailsRuntime.EventsEmit(a.ctx, "indexing-log", map[string]string{
		"root":    root,
		"message": message,
	})
}

// PickDirectory opens a native directory picker and returns the selected path
func (a *App) PickDirectory() (string, error) {
	return wailsRuntime.OpenDirectoryDialog(a.ctx, wailsRuntime.OpenDialogOptions{
		Title: "Select Workspace Directory",
	})
}

// BeforeClose is called when the application is about to close.
// It returns true to prevent closing, or false to allow it.
func (a *App) BeforeClose(ctx context.Context) bool {
	// Save window state before shutdown
	width, height := wailsRuntime.WindowGetSize(ctx)
	x, y := wailsRuntime.WindowGetPosition(ctx)
	isMaximized := wailsRuntime.WindowIsMaximised(ctx)

	// Avoid saving zero sizes
	if width > 0 && height > 0 {
		prefs := a.GetPreferences()
		prefs.WindowWidth = width
		prefs.WindowHeight = height
		prefs.WindowX = x
		prefs.WindowY = y
		prefs.IsMaximized = isMaximized

		if err := a.SavePreferences(prefs); err != nil {
			fmt.Printf("Hub: Failed to save window preferences: %v\n", err)
		}
	}

	return false // allow close
}

func (a *App) Shutdown(ctx context.Context) {
	if a.isHub {
		lockPath := mcp.GetHubLockPath()
		_ = os.Remove(lockPath)
		fmt.Println("Hub: Released hub.lock")
	}
}

func (a *App) startHubServer() {
	http.HandleFunc("/workspaces", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(a.workspaces)
			return
		}
		if r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var ws Workspace
		if err := json.NewDecoder(r.Body).Decode(&ws); err != nil {
			http.Error(w, "Invalid payload", http.StatusBadRequest)
			return
		}

		ws.Root = a.normalizePath(ws.Root)

		// Check if we already have this workspace (match by Root only)
		found := false
		for i, existing := range a.workspaces {
			// Case-insensitive comparison on Windows
			match := false
			if runtime.GOOS == "windows" {
				match = strings.EqualFold(existing.Root, ws.Root)
			} else {
				match = (existing.Root == ws.Root)
			}

			if match {
				// Update existing entry
				a.workspaces[i].PID = ws.PID
				a.workspaces[i].Client = ws.Client
				a.workspaces[i].LastSeen = time.Now().Format(time.RFC3339)
				a.workspaces[i].State = "active"
				// Keep current IsManaged flag
				
				ws = a.workspaces[i] // Use updated existing for DB persist
				found = true
				break
			}
		}

		if !found {
			ws.LastSeen = time.Now().Format(time.RFC3339)
			ws.State = "active"
			a.workspaces = append(a.workspaces, ws)
		}

		// Persist to database

		if a.db != nil {
			_, err := a.db.Exec(`
				INSERT INTO workspaces (root, client, last_seen, is_managed)
				VALUES (?, ?, ?, 0)
				ON CONFLICT(root) DO UPDATE SET
					client = excluded.client,
					last_seen = excluded.last_seen
			`, ws.Root, ws.Client, ws.LastSeen)
			if err != nil {
				fmt.Printf("Hub: Failed to persist workspace to DB: %v\n", err)
			}
		}

		fmt.Printf("Hub: Workspace registered/updated: %s (PID: %d, Client: %s)\n", ws.Root, ws.PID, ws.Client)

		// Emit event to frontend for real-time updates
		wailsRuntime.EventsEmit(a.ctx, "workspace-updated", a.workspaces)

		w.WriteHeader(http.StatusNoContent)
	})

	http.HandleFunc("/api/v1/inference", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req inference.InferenceRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid payload", http.StatusBadRequest)
			return
		}
		res, err := a.inference.Execute(r.Context(), req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(res)
	})

	http.HandleFunc("/api/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		models, err := a.ListLocalModels()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models)
	})

	http.HandleFunc("/api/v1/models/load", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			ModelID string `json:"modelId"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid payload", http.StatusBadRequest)
			return
		}

		// Trigger model load by calling Execute with empty prompt (or a dedicated Load method if we add it)
		// For now, Execute handles loading if modelID is provided.
		_, err := a.inference.Execute(r.Context(), inference.InferenceRequest{
			ModelID: req.ModelID,
			Prompt:  "", // Empty prompt just triggers load/switch
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusNoContent)
	})

	fmt.Println("Hub: Workspace registration server listening on http://localhost:9000")
	if err := http.ListenAndServe(":9000", nil); err != nil {
		fmt.Printf("Hub: Server failed: %v\n", err)
	}
}

// GetAstGrepStatus checks if ast-grep is installed and returns its version
func (a *App) GetAstGrepStatus() map[string]interface{} {
	result := map[string]interface{}{
		"installed": false,
		"version":   "",
		"path":      "",
	}

	// Determine install path based on OS
	osStr := runtime.GOOS
	var homeDir string
	var err error
	if osStr == "windows" {
		homeDir = os.Getenv("LOCALAPPDATA")
		if homeDir == "" {
			homeDir, err = os.UserHomeDir()
		}
	} else {
		homeDir, err = os.UserHomeDir()
	}

	if err != nil {
		return result
	}

	binDir := filepath.Join(homeDir, "context-sherpa", "bin")
	if osStr != "windows" {
		binDir = filepath.Join(homeDir, ".context-sherpa", "bin")
	}

	binName := "ast-grep"
	if osStr == "windows" {
		binName = "ast-grep.exe"
	}
	targetPath := filepath.Join(binDir, binName)

	if _, err := os.Stat(targetPath); err == nil {
		result["installed"] = true
		result["path"] = targetPath

		cmd := sysutils.SilentCommand(targetPath, "--version")
		if output, err := cmd.CombinedOutput(); err == nil {
			// Take the first line and trim
			v := strings.Split(strings.TrimSpace(string(output)), "\n")[0]
			result["version"] = v
		}
	} else {
		// Fallback to checking system PATH
		if path, err := exec.LookPath(binName); err == nil {
			result["installed"] = true
			result["path"] = path

			cmd := sysutils.SilentCommand(path, "--version")
			if output, err := cmd.CombinedOutput(); err == nil {
				v := strings.Split(strings.TrimSpace(string(output)), "\n")[0]
				result["version"] = v
			}
		}
	}

	return result
}

// getSherpaBinDir returns the platform-specific path to the Context-Sherpa bin directory
func getSherpaBinDir() (string, error) {
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

	binDir := filepath.Join(homeDir, "context-sherpa", "bin")
	if runtime.GOOS != "windows" {
		binDir = filepath.Join(homeDir, ".context-sherpa", "bin")
	}
	return binDir, nil
}

// getSherpaConfigDir returns the platform-specific path to the global Context-Sherpa config directory
func getSherpaConfigDir() (string, error) {
	var baseDir string
	if runtime.GOOS == "windows" {
		baseDir = os.Getenv("LOCALAPPDATA")
		if baseDir == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			baseDir = home
		}
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		baseDir = home
	}

	dir := filepath.Join(baseDir, "context-sherpa")
	if runtime.GOOS != "windows" {
		dir = filepath.Join(baseDir, ".context-sherpa")
	}

	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

// GetPreferences returns the stored user preferences
func (a *App) GetPreferences() UserPreferences {
	prefs := UserPreferences{
		Theme: "dracula", // Default
	}

	configDir, err := getSherpaConfigDir()
	if err != nil {
		return prefs
	}

	prefsPath := filepath.Join(configDir, "preferences.json")
	if data, err := os.ReadFile(prefsPath); err == nil {
		json.Unmarshal(data, &prefs)
	}

	return prefs
}

// SavePreferences saves user preferences to the global config directory
func (a *App) SavePreferences(prefs UserPreferences) error {
	configDir, err := getSherpaConfigDir()
	if err != nil {
		return err
	}

	prefsPath := filepath.Join(configDir, "preferences.json")
	data, err := json.MarshalIndent(prefs, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(prefsPath, data, 0644)
}

// TestInferenceConnection verifies that an inference provider is reachable
func (a *App) TestInferenceConnection(providerType string, url string) (string, error) {
	var provider inference.InferenceProvider
	switch providerType {
	case "ollama":
		provider = inference.NewOllamaProvider(url)
	case "openai":
		provider = inference.NewOpenAIProvider(url)
	case "lmstudio":
		provider = inference.NewLMStudioProvider(url)
	case "disabled":
		return "Disabled", nil
	default:
		return "", fmt.Errorf("invalid provider type: %s", providerType)
	}

	return provider.TestConnection(a.ctx)
}

// GetInferenceModels returns a list of available models from the configured provider
func (a *App) GetInferenceModels() ([]string, error) {
	prefs := a.GetPreferences()
	if prefs.InferenceProvider == "" || prefs.InferenceProvider == "disabled" {
		return []string{}, nil
	}

	var provider inference.InferenceProvider
	switch prefs.InferenceProvider {
	case "ollama":
		provider = inference.NewOllamaProvider(prefs.InferenceURL)
	case "openai":
		provider = inference.NewOpenAIProvider(prefs.InferenceURL)
	case "lmstudio":
		provider = inference.NewLMStudioProvider(prefs.InferenceURL)
	default:
		return nil, fmt.Errorf("invalid provider type")
	}

	return provider.ListModels(a.ctx)
}

// PullInferenceModel requests the provider to pull a model
func (a *App) PullInferenceModel(modelID string) error {
	prefs := a.GetPreferences()
	var provider inference.InferenceProvider

	switch prefs.InferenceProvider {
	case "ollama":
		provider = inference.NewOllamaProvider(prefs.InferenceURL)
	case "lmstudio":
		provider = inference.NewLMStudioProvider(prefs.InferenceURL)
	default:
		return fmt.Errorf("model pulling is not supported for provider: %s", prefs.InferenceProvider)
	}

	return provider.PullModel(a.ctx, modelID)
}
func (a *App) OpenConfigDir() error {
	path, err := getSherpaConfigDir()
	if err != nil {
		return err
	}

	// Ensure it exists
	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}

	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = sysutils.SilentCommand("explorer", path)
	case "darwin":
		cmd = sysutils.SilentCommand("open", path)
	default: // linux
		cmd = sysutils.SilentCommand("xdg-open", path)
	}

	return cmd.Start()
}

// OpenBinDir opens the Context-Sherpa bin directory in the OS file explorer
func (a *App) OpenBinDir() error {
	path, err := getSherpaBinDir()
	if err != nil {
		return err
	}

	// Ensure it exists
	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}

	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = sysutils.SilentCommand("explorer", path)
	case "darwin":
		cmd = sysutils.SilentCommand("open", path)
	default: // linux
		cmd = sysutils.SilentCommand("xdg-open", path)
	}

	return cmd.Start()
}

// GetScipIndexerStatus checks if a SCIP indexer for a specific language is installed
func (a *App) GetScipIndexerStatus(language string) map[string]interface{} {
	result := map[string]interface{}{
		"installed": false,
		"version":   "",
		"path":      "",
	}

	binName := "scip-" + language
	if language == "go" {
		binName = "scip-go"
	}

	binDir, err := getSherpaBinDir()
	if err != nil {
		return result
	}

	// Priority extensions on Windows
	extensions := []string{""}
	if runtime.GOOS == "windows" {
		extensions = []string{".exe", ".cmd", ".ps1", ".bat"}
	}

	// Places to look:
	// 1. In sherpa bin directory
	// 2. In sherpa bin/node_modules/.bin
	// 3. In system PATH

	checkPath := func(p string) bool {
		for _, ext := range extensions {
			fullPath := p + ext
			if _, err := os.Stat(fullPath); err == nil {
				result["installed"] = true
				result["path"] = fullPath
				return true
			}
		}
		return false
	}

	// 1 & 2: Local sherpa bin
	if !checkPath(filepath.Join(binDir, binName)) {
		checkPath(filepath.Join(binDir, "node_modules", ".bin", binName))
	}

	// 3: System PATH
	if !result["installed"].(bool) {
		for _, ext := range extensions {
			if path, err := exec.LookPath(binName + ext); err == nil {
				result["installed"] = true
				result["path"] = path
				break
			}
		}
	}

	if result["installed"].(bool) {
		// Try to get version
		cmd := sysutils.SilentCommand(result["path"].(string), "--version")
		if output, err := cmd.CombinedOutput(); err == nil {
			// Use the first line and trim
			v := strings.Split(strings.TrimSpace(string(output)), "\n")[0]
			result["version"] = v
		}
	}

	return result
}

// InstallScipIndexer downloads and installs a SCIP indexer for a specific language
func (a *App) InstallScipIndexer(language string) (string, error) {
	binDir, err := getSherpaBinDir()
	if err != nil {
		return "", fmt.Errorf("failed to get bin directory: %w", err)
	}

	if err := os.MkdirAll(binDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create bin directory: %w", err)
	}

	if language == "typescript" || language == "python" {
		npmPkg := "@sourcegraph/scip-" + language
		// Check for npm
		if _, err := exec.LookPath("npm"); err != nil {
			return "", fmt.Errorf("npm is required to install %s. Please install Node.js/npm first.", npmPkg)
		}

		// Install locally to bin directory
		cmd := sysutils.SilentCommand("npm", "install", "--prefix", binDir, npmPkg)
		if output, err := cmd.CombinedOutput(); err != nil {
			return "", fmt.Errorf("npm install failed: %v\nOutput: %s", err, string(output))
		}
		return fmt.Sprintf("Successfully installed %s to %s", npmPkg, binDir), nil
	}

	// For now, focusing on scip-go as the primary "native" example
	if language != "go" {
		return "", fmt.Errorf("automatic installation for %s is not yet implemented. Please install it manually in ~/.context-sherpa/bin/", language)
	}

	// scip-go release logic
	version := "0.1.26"
	arch := runtime.GOARCH
	osStr := runtime.GOOS

	var mappedArch, mappedOS string
	switch arch {
	case "amd64":
		mappedArch = "amd64"
	case "arm64":
		mappedArch = "arm64"
	default:
		return "", fmt.Errorf("unsupported architecture: %s", arch)
	}

	switch osStr {
	case "windows":
		mappedOS = "windows"
	case "darwin":
		mappedOS = "darwin"
	case "linux":
		mappedOS = "linux"
	default:
		return "", fmt.Errorf("unsupported OS: %s", osStr)
	}

	// Example: scip-go_0.1.26_windows_amd64.tar.gz
	filename := fmt.Sprintf("scip-go_%s_%s_%s.tar.gz", version, mappedOS, mappedArch)
	downloadURL := fmt.Sprintf("https://github.com/sourcegraph/scip-go/releases/download/v%s/%s", version, filename)

	binDir, err = getSherpaBinDir()
	if err != nil {
		return "", fmt.Errorf("failed to get bin directory: %w", err)
	}

	if err := os.MkdirAll(binDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create bin directory: %w", err)
	}

	// Download
	resp, err := http.Get(downloadURL)
	if err != nil {
		return "", fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download scip-go: HTTP %d", resp.StatusCode)
	}

	// Save to temp
	tmpFile, err := os.CreateTemp("", "scip-go-dl-*.tar.gz")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		return "", fmt.Errorf("failed to save download: %w", err)
	}

	// Extract .tar.gz
	// Since scip-go's tar.gz contains the binary at the root (usually)
	binName := "scip-go"
	if osStr == "windows" {
		binName = "scip-go.exe"
	}

	if err := extractTarGz(tmpFile.Name(), binDir, binName); err != nil {
		return "", fmt.Errorf("failed to extract: %w", err)
	}

	return fmt.Sprintf("Successfully installed scip-go to %s", binDir), nil
}

// InstallAstGrep downloads and extracts the latest ast-grep binary to local app data
func (a *App) InstallAstGrep() (string, error) {
	arch := runtime.GOARCH
	osStr := runtime.GOOS

	// Map architecture and OS to GitHub release filenames
	var mappedArch, mappedOS string
	switch arch {
	case "amd64":
		mappedArch = "x86_64"
	case "arm64":
		mappedArch = "aarch64"
	default:
		return "", fmt.Errorf("unsupported architecture: %s", arch)
	}

	switch osStr {
	case "windows":
		mappedOS = "pc-windows-msvc"
	case "darwin":
		mappedOS = "apple-darwin"
	case "linux":
		mappedOS = "unknown-linux-gnu"
	default:
		return "", fmt.Errorf("unsupported OS: %s", osStr)
	}

	filename := fmt.Sprintf("app-%s-%s.zip", mappedArch, mappedOS)
	downloadURL := fmt.Sprintf("https://github.com/ast-grep/ast-grep/releases/latest/download/%s", filename)

	binDir, err := getSherpaBinDir()
	if err != nil {
		return "", fmt.Errorf("failed to get bin directory: %w", err)
	}

	if err := os.MkdirAll(binDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create bin directory: %w", err)
	}

	// Target binary path
	binName := "ast-grep"
	if osStr == "windows" {
		binName = "ast-grep.exe"
	}
	targetPath := filepath.Join(binDir, binName)

	// Download file
	resp, err := http.Get(downloadURL)
	if err != nil {
		return "", fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download ast-grep: HTTP %d", resp.StatusCode)
	}

	// ast-grep always uses .zip as verified on GitHub
	tmpFile, err := os.CreateTemp("", "ast-grep-dl-*.zip")
	if err != nil {
		return "", fmt.Errorf("failed to create temp archive: %w", err)
	}
	tmpName := tmpFile.Name()
	defer os.Remove(tmpName)

	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		tmpFile.Close()
		return "", fmt.Errorf("failed to save download: %w", err)
	}
	tmpFile.Close()

	// Extract
	err = extractZip(tmpName, targetPath, binName)
	if err != nil {
		return "", fmt.Errorf("failed to extract: %w", err)
	}

	// Make executable
	if osStr != "windows" {
		if err := os.Chmod(targetPath, 0755); err != nil {
			return "", fmt.Errorf("failed to make executable: %w", err)
		}
	}

	return fmt.Sprintf("Latest ast-grep installed successfully to %s", targetPath), nil
}

// DeleteAstGrep removes the ast-grep binary from the local toolchain
func (a *App) DeleteAstGrep() error {
	binDir, err := getSherpaBinDir()
	if err != nil {
		return err
	}
	binName := "ast-grep"
	if runtime.GOOS == "windows" {
		binName = "ast-grep.exe"
	}
	targetPath := filepath.Join(binDir, binName)
	if _, err := os.Stat(targetPath); err == nil {
		return os.Remove(targetPath)
	}
	return nil
}

// DeleteScipIndexer uninstalls a SCIP indexer for a specific language
func (a *App) DeleteScipIndexer(language string) error {
	binDir, err := getSherpaBinDir()
	if err != nil {
		return err
	}

	// For NPM-based indexers, try to uninstall properly
	if language == "typescript" || language == "python" {
		npmPkg := "@sourcegraph/scip-" + language
		if _, err := exec.LookPath("npm"); err == nil {
			// Uninstall from our local prefix
			cmd := sysutils.SilentCommand("npm", "uninstall", "--prefix", binDir, npmPkg)
			cmd.Run()
		}
	}

	// Direct binary cleanup (Go or leftover NPM shims)
	binName := "scip-" + language
	if language == "go" {
		binName = "scip-go"
	}

	extensions := []string{""}
	if runtime.GOOS == "windows" {
		extensions = []string{".exe", ".cmd", ".ps1", ".bat"}
	}

	for _, ext := range extensions {
		// 1. Root bin dir
		targetPath := filepath.Join(binDir, binName+ext)
		if _, err := os.Stat(targetPath); err == nil {
			os.Remove(targetPath)
		}
		// 2. node_modules/.bin
		npmBinPath := filepath.Join(binDir, "node_modules", ".bin", binName+ext)
		if _, err := os.Stat(npmBinPath); err == nil {
			os.Remove(npmBinPath)
		}
	}

	return nil
}

// extractAllZip extracts every file in a zip archive to the target directory
func extractAllZip(zipPath, targetDir string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		fpath := filepath.Join(targetDir, f.Name)

		if f.FileInfo().IsDir() {
			os.MkdirAll(fpath, os.ModePerm)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
			return err
		}

		outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}

		rc, err := f.Open()
		if err != nil {
			outFile.Close()
			return err
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()

		if err != nil {
			return err
		}
	}
	return nil
}

// GetWorkspaces returns the list of registered workspaces
func (a *App) GetWorkspaces() []Workspace {
	if a.isHub {
		return a.workspaces
	}

	// If not Hub, try to fetch from the Master Hub
	lockPath := mcp.GetHubLockPath()
	data, err := os.ReadFile(lockPath)
	if err != nil {
		return a.workspaces
	}

	var lock mcp.HubLock
	if err := json.Unmarshal(data, &lock); err != nil {
		return a.workspaces
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/workspaces", lock.Port)
	client := &http.Client{Timeout: 1 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return a.workspaces
	}
	defer resp.Body.Close()

	var ws []Workspace
	if err := json.NewDecoder(resp.Body).Decode(&ws); err == nil {
		a.workspaces = ws
	}
	return a.workspaces
}

// OpenWorkspace opens the workspace root directory in the default file browser.
func (a *App) OpenWorkspace(root string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = sysutils.SilentCommand("explorer", root)
	case "darwin":
		cmd = sysutils.SilentCommand("open", root)
	default: // linux and others
		cmd = sysutils.SilentCommand("xdg-open", root)
	}
	return cmd.Run()
}

// Helper to extract a single file from a zip archive
func extractZip(zipPath, targetPath, targetFileName string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		if filepath.Base(f.Name) == targetFileName {
			rc, err := f.Open()
			if err != nil {
				return err
			}
			defer rc.Close()

			outFile, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
			if err != nil {
				return err
			}
			defer outFile.Close()

			if _, err := io.Copy(outFile, rc); err != nil {
				return err
			}
			return nil
		}
	}
	return fmt.Errorf("file %s not found in zip archive", targetFileName)
}

// Helper to extract a single file from a tar.gz archive
func extractTarGz(tarPath, targetPath, targetFileName string) error {
	// Note: In a production environment, you'd use archive/tar and compress/gzip.
	// For simplicity in this agentic task, I'll implement a basic extractor block here.
	// However, many SCIP tools follow this format.

	// Implementation using standard libraries
	f, err := os.Open(tarPath)
	if err != nil {
		return err
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// Look for the binary
		if filepath.Base(header.Name) == targetFileName {
			targetFile := filepath.Join(targetPath, targetFileName)
			outFile, err := os.OpenFile(targetFile, os.O_CREATE|os.O_RDWR|os.O_TRUNC, os.FileMode(header.Mode))
			if err != nil {
				return err
			}
			defer outFile.Close()

			if _, err := io.Copy(outFile, tr); err != nil {
				return err
			}
			return nil
		}
	}

	return fmt.Errorf("could not find %s in archive", targetFileName)
}

func (a *App) startViewerPoller() {
	ticker := time.NewTicker(5 * time.Second)
	for range ticker.C {
		if a.ctx == nil {
			continue
		}
		// Trigger a GetWorkspaces call to sync from Master Hub
		oldLen := len(a.workspaces)
		ws := a.GetWorkspaces()
		if len(ws) != oldLen {
			// If length changed, emit event to frontend
			wailsRuntime.EventsEmit(a.ctx, "workspace-updated", ws)
		}
	}
}

// startSweeper runs on the Master Hub to mark stale nodes as offline
func (a *App) startSweeper() {
	ticker := time.NewTicker(15 * time.Second)
	for range ticker.C {
		if a.ctx == nil {
			continue
		}
		updated := false
		newWorkspaces := make([]Workspace, 0)

		for _, ws := range a.workspaces {
			lastSeenTime, _ := time.Parse(time.RFC3339, ws.LastSeen)

			// Stage 2: Remove dead dynamic nodes after 2 minutes of silence
			if !ws.IsManaged && time.Since(lastSeenTime) > 120*time.Second {
				fmt.Printf("Hub: Sweeper removing dead dynamic workspace: %s (PID: %d)\n", ws.Root, ws.PID)
				updated = true
				continue
			}

			// Stage 1: Mark as offline after 60s
			if ws.State != "offline" && time.Since(lastSeenTime) > 60*time.Second {
				ws.State = "offline"
				fmt.Printf("Hub: Sweeper marking workspace as offline: %s (PID: %d)\n", ws.Root, ws.PID)
				updated = true
			}

			newWorkspaces = append(newWorkspaces, ws)
		}

		if updated {
			a.workspaces = newWorkspaces
			wailsRuntime.EventsEmit(a.ctx, "workspace-updated", a.workspaces)
		}

		// Also update the hub.lock heartbeat so others know we're alive
		a.updateHubLock()
	}
}

func (a *App) updateHubLock() {
	lockPath := mcp.GetHubLockPath()
	lock := mcp.HubLock{
		PID:       os.Getpid(),
		Port:      9000,
		StartTime: time.Now().Format(time.RFC3339), // Using StartTime as LastHeartbeat for simplicity
	}
	data, _ := json.MarshalIndent(lock, "", "  ")
	_ = os.WriteFile(lockPath, data, 0644)
}

// ListLocalModels returns a list of models in the local storage
func (a *App) ListLocalModels() ([]inference.ModelInfo, error) {
	configDir, err := getSherpaConfigDir()
	if err != nil {
		return nil, err
	}
	modelsDir := filepath.Join(configDir, "models")

	files, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, nil // Return empty list if dir doesn't exist
	}

	var models []inference.ModelInfo
	for _, f := range files {
		if !f.IsDir() && f.Name() != "runner.wasm" && f.Name() != "default.gguf" && !filepath.HasPrefix(f.Name(), ".") {
			info, _ := f.Info()
			models = append(models, inference.ModelInfo{
				ID:         f.Name(),
				Name:       f.Name(),
				Size:       info.Size(),
				Downloaded: true,
				Path:       filepath.Join(modelsDir, f.Name()),
			})
		}
	}
	return models, nil
}

// DeleteModel removes a downloaded model file from disk
func (a *App) DeleteModel(modelID string) error {
	configDir, err := getSherpaConfigDir()
	if err != nil {
		return err
	}
	modelPath := filepath.Join(configDir, "models", modelID)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil // Already gone
	}

	return os.Remove(modelPath)
}

// DownloadModel starts a background download of a model
func (a *App) DownloadModel(modelID string, url string) error {
	if !a.isHub {
		return fmt.Errorf("only the Master Hub can download models")
	}
	go func() {
		err := a.downloader.DownloadModel(context.Background(), modelID, url)
		if err != nil {
			fmt.Printf("Hub: Download failed for %s: %v\n", modelID, err)
			wailsRuntime.EventsEmit(a.ctx, "model-download-failed", map[string]string{
				"modelId": modelID,
				"error":   err.Error(),
			})
		} else {
			fmt.Printf("Hub: Download complete for %s\n", modelID)
			wailsRuntime.EventsEmit(a.ctx, "model-download-complete", modelID)
		}
	}()
	return nil
}

// GetDownloadProgress returns the progress of a model download
func (a *App) GetDownloadProgress(modelID string) float64 {
	if a.downloader == nil {
		return 0
	}
	p, exists := a.downloader.GetProgress(modelID)
	if !exists {
		// Check if it already exists on disk
		configDir, _ := getSherpaConfigDir()
		path := filepath.Join(configDir, "models", modelID)
		if _, err := os.Stat(path); err == nil {
			return 1.0
		}
		return 0
	}
	return p
}

// RunInference executes a prompt using a local model
func (a *App) RunInference(modelID string, prompt string) (string, error) {
	req := inference.InferenceRequest{
		ModelID: modelID,
		Prompt:  prompt,
	}

	if !a.isHub {
		// Forward to the Master Hub
		lockPath := mcp.GetHubLockPath()
		data, err := os.ReadFile(lockPath)
		if err != nil {
			return "", fmt.Errorf("could not find Master Hub: %w", err)
		}
		var lock mcp.HubLock
		if err := json.Unmarshal(data, &lock); err != nil {
			return "", fmt.Errorf("invalid hub.lock: %w", err)
		}

		url := fmt.Sprintf("http://127.0.0.1:%d/api/v1/inference", lock.Port)
		jsonData, _ := json.Marshal(req)
		resp, err := http.Post(url, "application/json", strings.NewReader(string(jsonData)))
		if err != nil {
			return "", fmt.Errorf("failed to forward inference to Hub: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("hub returned error: %d", resp.StatusCode)
		}

		var res inference.InferenceResponse
		if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
			return "", fmt.Errorf("failed to decode hub response: %w", err)
		}
		return res.Text, nil
	}

	res, err := a.inference.Execute(context.Background(), req)
	if err != nil {
		return "", err
	}
	return res.Text, nil
}

// SearchForIndexes recursively finds .scip files in the workspace
func (a *App) SearchForIndexes(rootPath string) ([]string, error) {
	var scipFiles []string
	err := filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip errors
		}
		if !info.IsDir() && (strings.HasSuffix(path, ".scip")) {
			scipFiles = append(scipFiles, path)
		}
		return nil
	})
	return scipFiles, err
}

// RegenerateIndex triggers a regeneration of a specific SCIP index.
func (a *App) RegenerateIndex(workspaceRoot string, scipPath string) error {
	// Derive actual project root from the .scip file path.
	// Index files are stored in {projectRoot}/.context-sherpa/index-{lang}.scip
	// So we go up 2 levels: index file -> .context-sherpa/ -> projectRoot/
	projectRoot := filepath.Dir(filepath.Dir(scipPath))

	// Infer language from scip filename (e.g., index-go.scip -> go)
	filename := filepath.Base(scipPath)
	language := ""
	if strings.HasPrefix(filename, "index-") {
		language = strings.TrimPrefix(filename, "index-")
		language = strings.TrimSuffix(language, ".scip")
	}

	return mcp.IndexWorkspace(projectRoot, language)
}

// GetFileContent reads the content of a file from a workspace.
func (a *App) GetFileContent(workspaceRoot string, relativePath string) (string, error) {
	// Canonicalize paths to prevent directory traversal
	absRoot, _ := filepath.Abs(workspaceRoot)
	joined := filepath.Join(absRoot, relativePath)
	absPath, _ := filepath.Abs(joined)

	// Security: Ensure the resolved path is still within the workspace root
	if !strings.HasPrefix(absPath, absRoot) {
		return "", fmt.Errorf("security: access denied to path outside workspace root")
	}

	content, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}
	return string(content), nil
}

// GraphNode represents an ECharts graph node
type GraphNode struct {
	ID        string   `json:"id"`
	Name      string   `json:"name"`
	Value     int      `json:"value"`
	Category  int      `json:"category"`
	Path      string   `json:"path"`
	Kind      string   `json:"kind"`
	Docstring string   `json:"docstring"`
	Members   []Member `json:"members"`
	Loc       int      `json:"loc"`
	StartLine int      `json:"startLine"`
	EndLine   int      `json:"endLine"`
	Parent    string   `json:"parent,omitempty"`
}

// Member represents a sub-component of a node (e.g., field or method)
type Member struct {
	Name   string `json:"name"`
	Kind   string `json:"kind"`
	Symbol string `json:"symbol"` // Full SCIP symbol for navigation
}

// GraphLink represents an ECharts graph link
type GraphLink struct {
	Source string `json:"source"`
	Target string `json:"target"`
	Label  string `json:"label"`
}

// GraphCategory represents a graph category (e.g., folder or kind)
type GraphCategory struct {
	Name string `json:"name"`
}

// CyElement represents a Cytoscape.js element (node or edge)
type CyElement struct {
	Group string `json:"group"`
	Data  any    `json:"data"`
}

// GraphData representing the response for Cytoscape.js
type GraphData struct {
	Elements   []CyElement     `json:"elements"`
	Categories []GraphCategory `json:"categories"`
	Language   string          `json:"language"`
}

// GetGraphData transforms SCIP data into ECharts JSON with "Hotpath" sizing and spatial clustering
func (a *App) GetGraphData(scipPath string) (*GraphData, error) {
	data, err := os.ReadFile(scipPath)
	if err != nil {
		return nil, err
	}

	var index scip.Index
	if err := proto.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("failed to parse SCIP index: %w", err)
	}

	nodes := make([]GraphNode, 0)
	links := make([]GraphLink, 0)
	seenNodes := make(map[string]bool)
	seenLinks := make(map[string]bool)

	// Custom Categories for Layout/Clustering (Folders)
	// But Legend will be fixed to Code Kinds.
	folderToCatID := make(map[string]int)
	categories := make([]GraphCategory, 0)

	symbolToNodeID := make(map[string]string)
	symbolToInfo := make(map[string]*scip.SymbolInformation)
	refCount := make(map[string]int)
	outboundCalls := make(map[string]int)

	addNode := func(n GraphNode) {
		if n.ID != "" && !seenNodes[n.ID] {
			nodes = append(nodes, n)
			seenNodes[n.ID] = true
		}
	}

	addLink := func(l GraphLink) {
		if l.Source == "" || l.Target == "" || l.Source == l.Target {
			return
		}
		key := fmt.Sprintf("%s-%s-%s", l.Source, l.Target, l.Label)
		if !seenLinks[key] {
			links = append(links, l)
			seenLinks[key] = true
		}
	}

	// 1. Pre-map symbol information and identify directories
	for _, doc := range index.Documents {
		dir := filepath.Dir(doc.RelativePath)
		if dir == "." {
			dir = "root"
		}
		if _, exists := folderToCatID[dir]; !exists {
			folderToCatID[dir] = len(categories)
			categories = append(categories, GraphCategory{Name: dir})
		}
		for _, si := range doc.Symbols {
			symbolToInfo[si.Symbol] = si
		}
	}
	for _, si := range index.ExternalSymbols {
		symbolToInfo[si.Symbol] = si
	}

	// 2. Pass 1: Count references and outbound calls
	for _, doc := range index.Documents {
		var currentScope string
		for _, occ := range doc.Occurrences {
			if strings.Contains(occ.Symbol, "local") {
				continue
			}
			isDef := occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0
			if isDef {
				currentScope = occ.Symbol
				continue
			}
			// It's a reference
			refCount[occ.Symbol]++
			if currentScope != "" {
				outboundCalls[currentScope]++
			}
		}
	}

	// 3. Pass 2: Create Symbol Nodes (No physical folder/file nodes)
	for _, doc := range index.Documents {
		dir := filepath.Dir(doc.RelativePath)
		catID := folderToCatID[dir]

		for _, occ := range doc.Occurrences {
			isDef := occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0
			if isDef {
				// Filter noise: skip local symbols
				if strings.Contains(occ.Symbol, "local") {
					continue
				}

				parsed, err := scip.ParseSymbol(occ.Symbol)
				if err != nil || len(parsed.Descriptors) == 0 {
					continue // Skip packages or unparseable symbols
				}

				// The last descriptor is the leaf (Method, Type, Term, etc.)
				leaf := parsed.Descriptors[len(parsed.Descriptors)-1]

				// Skip Namespaces (Packages) as definitions - they are architectural noise
				if leaf.Suffix == scip.Descriptor_Namespace {
					continue
				}

				name := leaf.Name
				kind := "Function"

				// Map SCIP Descriptor Suffix to our finite categories
				switch leaf.Suffix {
				case scip.Descriptor_Type:
					kind = "Struct"
				case scip.Descriptor_Term:
					kind = "Variable"
				case scip.Descriptor_Method:
					kind = "Function"
				case scip.Descriptor_Macro:
					kind = "Function"
				case scip.Descriptor_Parameter:
					continue // Skip parameters for top-level graph
				case scip.Descriptor_TypeParameter:
					continue // Skip type parameters
				}

				// Additional refinement for Interfaces (often labeled as Types in SCIP but contain "interface" in symbol)
				if kind == "Struct" && strings.Contains(strings.ToLower(occ.Symbol), "interface") {
					kind = "Interface"
				}

				nodeID := "sym:" + occ.Symbol
				symbolToNodeID[occ.Symbol] = nodeID

				info := symbolToInfo[occ.Symbol]
				docstring := ""
				if info != nil && len(info.Documentation) > 0 {
					docstring = info.Documentation[0]
				}

				// Calculate Lines of Code (LOC) and Ranges
				loc := 0
				startLine := 0
				endLine := 0
				if len(occ.Range) == 3 {
					startLine = int(occ.Range[0] + 1) // [line, startCol, endCol]
					endLine = startLine
					loc = 1
				} else if len(occ.Range) >= 4 {
					startLine = int(occ.Range[0] + 1) // [startLine, startCol, endLine, endCol]
					endLine = int(occ.Range[2] + 1)
					loc = int(occ.Range[2] - occ.Range[0] + 1)
				}

				// Calculate "Hotpath" value (Directive Weights)
				val := 0
				if kind == "Struct" || kind == "Interface" {
					// Count members for struct sizing
					memberCount := 0
					prefix := occ.Symbol
					for sym := range symbolToInfo {
						if strings.HasPrefix(sym, prefix) && sym != occ.Symbol {
							if !strings.Contains(sym[len(prefix):], ".") {
								memberCount++
							}
						}
					}
					// Value = (MemberCount * 5) + (ReferenceCount * 10)
					val = (memberCount * 5) + (refCount[occ.Symbol] * 10)
				} else if kind == "Variable" {
					val = 5 + (refCount[occ.Symbol] * 2)
				} else {
					// Function: Value = (OutboundCalls * 3) + (ReferenceCount * 5)
					val = (outboundCalls[occ.Symbol] * 3) + (refCount[occ.Symbol] * 5)
				}

				// Minimum size floor
				if val < 5 {
					val = 5
				}

				parentID := ""
				if dir != "" && dir != "." {
					parentID = "dir:" + dir
				}

				addNode(GraphNode{
					ID:        nodeID,
					Name:      name,
					Value:     val,
					Category:  catID, // Spatial clustering by folder
					Path:      doc.RelativePath,
					Kind:      kind,
					Docstring: docstring,
					Members:   []Member{},
					Loc:       loc,
					StartLine: startLine,
					EndLine:   endLine,
					Parent:    parentID,
				})
			}
		}
	}

	// 4. Extract members for deep inspection
	for i := range nodes {
		if nodes[i].Kind != "Struct" && nodes[i].Kind != "Interface" {
			continue
		}
		// Nodes[i].ID is "sym:" + symbol
		originalSymbol := nodes[i].ID[4:]

		for sym := range symbolToInfo {
			if strings.HasPrefix(sym, originalSymbol) && sym != originalSymbol {
				parsed, err := scip.ParseSymbol(sym)
				if err != nil || len(parsed.Descriptors) == 0 {
					continue
				}

				// Only direct children (e.g., App#scipFiles. and not App#scipFiles.inner.)
				// This is a bit simplified; in SCIP Go, members are usually descriptors at the end
				// We check if the parent symbol is indeed the parent in descriptors
				isChild := false
				parentParsed, _ := scip.ParseSymbol(originalSymbol)
				if len(parsed.Descriptors) == len(parentParsed.Descriptors)+1 {
					isChild = true
					for j := 0; j < len(parentParsed.Descriptors); j++ {
						if parsed.Descriptors[j].Name != parentParsed.Descriptors[j].Name ||
							parsed.Descriptors[j].Suffix != parentParsed.Descriptors[j].Suffix {
							isChild = false
							break
						}
					}
				}

				if isChild {
					leaf := parsed.Descriptors[len(parsed.Descriptors)-1]
					memberKind := "Field"
					if leaf.Suffix == scip.Descriptor_Method {
						memberKind = "Method"
					}

					nodes[i].Members = append(nodes[i].Members, Member{
						Name:   leaf.Name,
						Kind:   memberKind,
						Symbol: sym,
					})
				}
			}
		}
	}

	// 5. Build Links (Symbol to Symbol)
	for _, doc := range index.Documents {
		var currentScope string
		for _, occ := range doc.Occurrences {
			if strings.Contains(occ.Symbol, "local") {
				continue
			}
			if occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0 {
				if _, exists := symbolToNodeID[occ.Symbol]; exists {
					currentScope = occ.Symbol
				}
				continue
			}

			// Reference
			targetNodeID, exists := symbolToNodeID[occ.Symbol]
			if exists && currentScope != "" {
				sourceNodeID := symbolToNodeID[currentScope]
				if sourceNodeID != "" && sourceNodeID != targetNodeID {
					label := "CALLS"
					if strings.HasSuffix(occ.Symbol, "#") {
						label = "USES"
					} else if strings.Contains(occ.Symbol, "interface") {
						label = "IMPLEMENTS"
					}

					// Get names/paths for Relationship Mode
					sourceName := currentScope
					if parsed, err := scip.ParseSymbol(currentScope); err == nil && len(parsed.Descriptors) > 0 {
						sourceName = parsed.Descriptors[len(parsed.Descriptors)-1].Name
					} else if lastSlash := strings.LastIndex(sourceName, "/"); lastSlash != -1 {
						sourceName = sourceName[lastSlash+1:]
					}

					targetName := occ.Symbol
					if parsed, err := scip.ParseSymbol(occ.Symbol); err == nil && len(parsed.Descriptors) > 0 {
						targetName = parsed.Descriptors[len(parsed.Descriptors)-1].Name
					} else if lastSlash := strings.LastIndex(targetName, "/"); lastSlash != -1 {
						targetName = targetName[lastSlash+1:]
					}
					sourceName = strings.TrimSuffix(strings.TrimSuffix(sourceName, "#"), "().")
					targetName = strings.TrimSuffix(strings.TrimSuffix(targetName, "#"), "().")

					addLink(GraphLink{
						Source: sourceNodeID,
						Target: targetNodeID,
						Label:  fmt.Sprintf("%s -> %s (%s)", sourceName, targetName, label),
					})
				}
			}
		}
	}

	// 6. Final Sweep: Construct Cytoscape Elements
	elements := make([]CyElement, 0)

	// Add Folder Nodes (Compound Nodes)
	seenFolders := make(map[string]bool)
	for dir := range folderToCatID {
		if dir == "" || dir == "." || dir == "root" {
			continue
		}

		// Create hierarchical folder nodes
		parts := strings.Split(dir, string(filepath.Separator))
		currentPath := ""
		for i, part := range parts {
			prevPath := currentPath
			if currentPath == "" {
				currentPath = part
			} else {
				currentPath = currentPath + string(filepath.Separator) + part
			}

			folderID := "dir:" + currentPath
			if !seenFolders[folderID] {
				parentID := ""
				if i > 0 {
					parentID = "dir:" + prevPath
				}

				elements = append(elements, CyElement{
					Group: "nodes",
					Data: GraphNode{
						ID:     folderID,
						Name:   part,
						Kind:   "Folder",
						Parent: parentID,
					},
				})
				seenFolders[folderID] = true
			}
		}
	}

	// Add Nodes (Symbols)
	for _, n := range nodes {
		elements = append(elements, CyElement{
			Group: "nodes",
			Data:  n,
		})
	}

	// Add Edges
	for _, l := range links {
		if seenNodes[l.Source] && seenNodes[l.Target] {
			elements = append(elements, CyElement{
				Group: "edges",
				Data: map[string]interface{}{
					"id":     fmt.Sprintf("e-%s-%s", l.Source, l.Target),
					"source": l.Source,
					"target": l.Target,
					"label":  l.Label,
				},
			})
		}
	}

	return &GraphData{
		Elements:   elements,
		Categories: categories,
		Language:   detectLanguage(index.Documents),
	}, nil
}

// FocusWorkspaceClient attempts to bring the editor window associated with an MCP server to the foreground.
func (a *App) FocusWorkspaceClient(mcpPid int) error {
	if mcpPid <= 0 {
		return fmt.Errorf("invalid PID")
	}

	fmt.Printf("Hub: Debug: Starting window focus for MCP PID %d\n", mcpPid)

	// 1. Collect process ancestry
	ancestry := make([]int, 0)
	current := mcpPid
	visited := make(map[int]bool)
	
	var bestEditorPid int

	for i := 0; i < 20; i++ {
		if current <= 0 || visited[current] {
			break
		}
		visited[current] = true
		ancestry = append(ancestry, current)

		name := strings.ToLower(sysutils.GetProcessName(current))
		fmt.Printf("Hub: Debug: Ancestry Lvl %d: PID %d (%s)\n", i, current, name)

		if bestEditorPid == 0 && sysutils.IsEditor(name) {
			bestEditorPid = current
			fmt.Printf("Hub: Debug: Identified best editor candidate: PID %d (%s)\n", current, name)
		}

		next := sysutils.GetParentPid(current)
		if next == current || next <= 0 {
			break
		}
		current = next
	}

	// 2. Perform platform-specific focus search using sysutils
	err := sysutils.FocusWindow(ancestry, bestEditorPid)
	if err != nil && runtime.GOOS == "linux" {
		wailsRuntime.MessageDialog(a.ctx, wailsRuntime.MessageDialogOptions{
			Type:    wailsRuntime.InfoDialog,
			Title:   "Focus Client",
			Message: sysutils.GetWindowFocusError(),
		})
	}
	return err
}

func (a *App) getProcessName(pid int) string {
	return sysutils.GetProcessName(pid)
}

func (a *App) getParentPid(pid int) int {
	return sysutils.GetParentPid(pid)
}

// remove unused findEditorPid
func (a *App) findEditorPid(startPid int) int {
	return 0
}

// Re-implement focusWindowsInAncestry as a wrapper if needed or remove it.
// Here we remove it since it's now in sysutils.

func detectLanguage(docs []*scip.Document) string {

	exts := make(map[string]int)
	for _, doc := range docs {
		ext := filepath.Ext(doc.RelativePath)
		if ext != "" {
			exts[ext]++
		}
	}

	mainExt := ""
	maxCount := 0
	for ext, count := range exts {
		if count > maxCount {
			maxCount = count
			mainExt = ext
		}
	}

	switch mainExt {
	case ".go":
		return "Go"
	case ".ts", ".tsx":
		return "TypeScript"
	case ".js", ".jsx":
		return "JavaScript"
	case ".py":
		return "Python"
	case ".rs":
		return "Rust"
	case ".cpp", ".hpp", ".cc", ".h":
		return "C++"
	default:
		return "Generic"
	}
}
// SearchCommunityRules searches for rules in the community repository.
func (a *App) SearchCommunityRules(query string, language string, tags string) ([]mcp.CommunityRule, error) {
	index, err := mcp.FetchCommunityRuleIndex()
	if err != nil {
		return nil, err
	}

	language = strings.ToLower(language)
	var tagList []string
	if tags != "" {
		tagList = strings.Split(tags, ",")
		for i, t := range tagList {
			tagList[i] = strings.ToLower(strings.TrimSpace(t))
		}
	}

	var matchingRules []mcp.CommunityRule
	for _, rule := range index.Rules {
		if language != "" && strings.ToLower(rule.Language) != language {
			continue
		}

		if len(tagList) > 0 {
			hasAll := true
			for _, rt := range tagList {
				found := false
				for _, t := range rule.Tags {
					if strings.ToLower(t) == rt {
						found = true
						break
					}
				}
				if !found {
					hasAll = false
					break
				}
			}
			if !hasAll {
				continue
			}
		}
		matchingRules = append(matchingRules, rule)
	}

	if query != "" {
		queryLower := strings.ToLower(query)
		var queryMatches []mcp.CommunityRule
		for _, rule := range matchingRules {
			if strings.Contains(strings.ToLower(rule.ID), queryLower) ||
				strings.Contains(strings.ToLower(rule.Description), queryLower) {
				queryMatches = append(queryMatches, rule)
				continue
			}
			for _, t := range rule.Tags {
				if strings.Contains(strings.ToLower(t), queryLower) {
					queryMatches = append(queryMatches, rule)
					break
				}
			}
		}
		matchingRules = queryMatches
	}

	return matchingRules, nil
}

// GetCommunityRuleDetails returns the rule metadata and YAML content.
func (a *App) GetCommunityRuleDetails(ruleID string) (map[string]interface{}, error) {
	rule, content, err := mcp.FetchCommunityRuleContent(ruleID)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"rule":    rule,
		"content": content,
	}, nil
}

// ImportCommunityRule downloads a community rule to a specific configuration's rule directory.
func (a *App) ImportCommunityRule(configPath string, ruleID string) error {
	rule, content, err := mcp.FetchCommunityRuleContent(ruleID)
	if err != nil {
		return err
	}

	// Validate the rule content
	if err := mcp.ValidateAstGrepRule(content); err != nil {
		return err
	}

	// Resolve rule directory (usually 'rules' relative to config)
	configDir := filepath.Dir(configPath)
	ruleDir := filepath.Join(configDir, "rules") // Default fallback

	data, err := os.ReadFile(configPath)
	if err == nil {
		var config mcp.SgConfig
		if err := yaml.Unmarshal(data, &config); err == nil && len(config.RuleDirs) > 0 {
			ruleDir = filepath.Join(configDir, config.RuleDirs[0])
		}
	}

	if err := os.MkdirAll(ruleDir, 0755); err != nil {
		return fmt.Errorf("failed to create rule directory: %w", err)
	}

	filename := filepath.Base(rule.Path)
	destPath := filepath.Join(ruleDir, filename)

	return os.WriteFile(destPath, []byte(content), 0644)
}

// GetWorkspaceConfigs recursively searches for all sgconfig.yml files in the workspace.
func (a *App) GetWorkspaceConfigs(workspaceRoot string) ([]map[string]interface{}, error) {
	return mcp.GetWorkspaceConfigs(workspaceRoot)
}

// InitializeAstGrepConfig sets up a new ast-grep configuration in the target directory.
func (a *App) InitializeAstGrepConfig(directory string, language string) error {
	return mcp.InitializeAstGrepConfig(directory, language)
}

// RemoveLocalRule removes a specific rule from a configuration's rule directory.
func (a *App) RemoveLocalRule(configPath string, ruleID string) error {
	return mcp.RemoveLocalRule(configPath, ruleID)
}

// GetLocalRulesInDir returns a list of installed rules in a specific directory.
func (a *App) GetLocalRulesInDir(directory string) ([]string, error) {
	files, err := os.ReadDir(directory)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var rules []string
	for _, f := range files {
		if !f.IsDir() && (strings.HasSuffix(f.Name(), ".yml") || strings.HasSuffix(f.Name(), ".yaml")) {
			rules = append(rules, f.Name())
		}
	}
	return rules, nil
}

