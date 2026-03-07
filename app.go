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
	"time"

	"github.com/hackafterdark/context-sherpa/pkg/mcp"
	wailsRuntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

// Workspace represents a detected workspace from a Node
type Workspace struct {
	PID    int    `json:"pid"`
	Root   string `json:"root"`
	Client string `json:"client"`
	State  string `json:"state"`
}

// App struct
type App struct {
	ctx        context.Context
	workspaces []Workspace
	isHub      bool
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts.
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	a.workspaces = make([]Workspace, 0)
	
	// Try to become the Master Hub
	if a.tryAcquireHubLock() {
		a.isHub = true
		fmt.Println("Hub: Successfully acquired hub.lock. Starting as Master Hub.")
		// Start the Hub's registration server on a background goroutine
		go a.startHubServer()
	} else {
		fmt.Println("Hub: Another instance is already Master Hub. Starting as Node viewer.")
	}
}

func (a *App) tryAcquireHubLock() bool {
	lockPath := mcp.GetHubLockPath()
	
	// 1. Try to read existing lock
	if data, err := os.ReadFile(lockPath); err == nil {
		var lock mcp.HubLock
		if err := json.Unmarshal(data, &lock); err == nil {
			// Check if process still exists
			if _, err := os.FindProcess(lock.PID); err == nil {
				// Hub is likely already running
				return false
			}
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
		if r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var ws Workspace
		if err := json.NewDecoder(r.Body).Decode(&ws); err != nil {
			http.Error(w, "Invalid payload", http.StatusBadRequest)
			return
		}

		// Check if we already have this workspace (same root)
		found := false
		for i, existing := range a.workspaces {
			if existing.Root == ws.Root {
				a.workspaces[i] = ws
				found = true
				break
			}
		}

		if !found {
			a.workspaces = append(a.workspaces, ws)
		}

		fmt.Printf("Hub: Workspace registered/updated: %s (PID: %d, Client: %s)\n", ws.Root, ws.PID, ws.Client)
		
		// Emit event to frontend for real-time updates
		wailsRuntime.EventsEmit(a.ctx, "workspace-updated", a.workspaces)

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
		if output, err := cmd.Output(); err == nil {
			result["version"] = strings.TrimSpace(string(output))
		}
	} else {
		// Fallback to checking system PATH
		if path, err := exec.LookPath(binName); err == nil {
			result["installed"] = true
			result["path"] = path

			cmd := exec.Command(path, "--version")
			if output, err := cmd.Output(); err == nil {
				result["version"] = strings.TrimSpace(string(output))
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
		if output, err := cmd.Output(); err == nil {
			result["version"] = strings.TrimSpace(string(output))
		}
	} else {
		// Fallback to system PATH
		if path, err := exec.LookPath(binName); err == nil {
			result["installed"] = true
			result["path"] = path
			cmd := exec.Command(path, "--version")
			if output, err := cmd.Output(); err == nil {
				result["version"] = strings.TrimSpace(string(output))
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

// GetWorkspaces returns the list of registered workspaces
func (a *App) GetWorkspaces() []Workspace {
	return a.workspaces
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
