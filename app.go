package main

import (
	"archive/tar"
	"archive/zip"
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

	"github.com/hackafterdark/context-sherpa/pkg/inference"
	"github.com/hackafterdark/context-sherpa/pkg/mcp"
	wailsRuntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

// Workspace represents a detected workspace from a Node
type Workspace struct {
	PID      int       `json:"pid"`
	Root     string    `json:"root"`
	Client   string    `json:"client"`
	State    string    `json:"state"`
	LastSeen time.Time `json:"lastSeen"`
}

// UserPreferences represents persistent user settings
type UserPreferences struct {
	Theme string `json:"theme"`
}

// App struct
type App struct {
	ctx        context.Context
	workspaces []Workspace
	isHub      bool
	downloader *inference.Downloader
	inference  *inference.InferenceService
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts.
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	a.workspaces = make([]Workspace, 0)

	// Ensure we have a clean start by checking lock liveness
	if a.tryAcquireHubLock() {
		a.isHub = true
		fmt.Println("Hub: Successfully acquired hub.lock. Starting as Master Hub.")
		
		// Initialize Inference services
		configDir, _ := getSherpaConfigDir()
		modelsDir := filepath.Join(configDir, "models")
		a.downloader = inference.NewDownloader(modelsDir)
		a.inference = inference.NewInferenceService(modelsDir)

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
		cmd := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH")
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
		StartTime: time.Now(),
	}
	data, _ := json.MarshalIndent(lock, "", "  ")

	// Create directory if it doesn't exist (GetHubLockPath does this, but being safe)
	_ = os.MkdirAll(filepath.Dir(lockPath), 0755)

	err := os.WriteFile(lockPath, data, 0644)
	return err == nil
}

// Shutdown is called when the app closes
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

		// Check if we already have this workspace (same root AND same PID)
		found := false
		for i, existing := range a.workspaces {
			if existing.Root == ws.Root && existing.PID == ws.PID {
				ws.LastSeen = time.Now()
				if ws.State == "offline" {
					ws.State = "active"
				}
				a.workspaces[i] = ws
				found = true
				break
			}
		}

		if !found {
			ws.LastSeen = time.Now()
			a.workspaces = append(a.workspaces, ws)
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

		cmd := exec.Command(targetPath, "--version")
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

			cmd := exec.Command(path, "--version")
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

// OpenConfigDir opens the Context-Sherpa root config directory in the OS file explorer
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
		cmd = exec.Command("explorer", path)
	case "darwin":
		cmd = exec.Command("open", path)
	default: // linux
		cmd = exec.Command("xdg-open", path)
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
		cmd = exec.Command("explorer", path)
	case "darwin":
		cmd = exec.Command("open", path)
	default: // linux
		cmd = exec.Command("xdg-open", path)
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

	if runtime.GOOS == "windows" {
		binName += ".exe"
	}

	binDir, err := getSherpaBinDir()
	if err != nil {
		return result
	}

	targetPath := filepath.Join(binDir, binName)
	// For npm-installed tools, they might be in node_modules/.bin
	npmBinPath := filepath.Join(binDir, "node_modules", ".bin", binName)

	if _, err := os.Stat(targetPath); err == nil {
		result["installed"] = true
		result["path"] = targetPath
	} else if _, err := os.Stat(npmBinPath); err == nil {
		result["installed"] = true
		result["path"] = npmBinPath
	}

	if result["installed"].(bool) {
		// Try to get version
		cmd := exec.Command(result["path"].(string), "--version")
		if output, err := cmd.CombinedOutput(); err == nil {
			// Use the first line and trim
			v := strings.Split(strings.TrimSpace(string(output)), "\n")[0]
			result["version"] = v
		}
	} else {
		// Fallback to system PATH
		if path, err := exec.LookPath(binName); err == nil {
			result["installed"] = true
			result["path"] = path
			cmd := exec.Command(path, "--version")
			if output, err := cmd.CombinedOutput(); err == nil {
				v := strings.Split(strings.TrimSpace(string(output)), "\n")[0]
				result["version"] = v
			}
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
		cmd := exec.Command("npm", "install", "--prefix", binDir, npmPkg)
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

	// Save to temp file
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
		cmd = exec.Command("explorer", root)
	case "darwin":
		cmd = exec.Command("open", root)
	default: // linux and others
		cmd = exec.Command("xdg-open", root)
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
		if f.Name == targetFileName {
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
			// Stage 2: Remove entirely after 2 minutes of silence
			if time.Since(ws.LastSeen) > 120*time.Second {
				fmt.Printf("Hub: Sweeper removing dead workspace: %s (PID: %d)\n", ws.Root, ws.PID)
				updated = true
				continue
			}

			// Stage 1: Mark as offline after 60s
			if ws.State != "offline" && time.Since(ws.LastSeen) > 60*time.Second {
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
		StartTime: time.Now(), // Using StartTime as LastHeartbeat for simplicity
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

