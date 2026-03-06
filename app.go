package main

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// App struct
type App struct {
	ctx context.Context
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts.
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
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

	// Determine install path based on OS
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
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}

	binDir := filepath.Join(homeDir, "context-sherpa", "bin")
	if osStr != "windows" {
		binDir = filepath.Join(homeDir, ".context-sherpa", "bin")
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
