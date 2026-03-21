package main

import (
	"context"
	"embed"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"runtime/debug"

	"github.com/hackafterdark/context-sherpa/pkg/inference"
	"github.com/hackafterdark/context-sherpa/pkg/mcp"
	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/linux"
	"github.com/wailsapp/wails/v2/pkg/options/mac"
)

//go:embed all:frontend/dist
var assets embed.FS

//go:embed build/appicon.png
var icon []byte

func initDebugLog(path string) *os.File {
	if path == "" {
		path = "production_debug.log"
	}
	logFile, _ := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if logFile != nil {
		log.SetOutput(logFile)
		log.Println("--- NEW LOG SESSION ---")
	}
	return logFile
}

func main() {
	if os.Getenv("APP_BUILD_MODE") == "true" {
		// Just initialize what's needed for bindings and exit
		return
	}
	// 1. Setup panic recovery first
	defer func() {
		if r := recover(); r != nil {
			log.Printf("PANIC RECOVERED: %v\nStack Trace:\n%s", r, string(debug.Stack()))
			fmt.Printf("PANIC RECOVERED: %v\n", r)
		}
	}()

	// 2. Parse flags early to check for debug/logging options
	isMCPMode := false
	mcpFlag := flag.Bool("mcp", false, "Run in headless MCP server mode")
	workspaceRoot := flag.String("workspaceRoot", "", "Workspace root directory (defaults to current working directory)")
	projectRoot := flag.String("projectRoot", "", "Workspace root directory (legacy alias for workspaceRoot)")
	verbose := flag.Bool("verbose", false, "Enable verbose logging for debugging")
	logFile := flag.String("logFile", "", "Path to file where logs will be appended (optional)")
	astGrepPath := flag.String("astGrepPath", "", "Explicit path to ast-grep binary")
	clientName := flag.String("client", "", "Name of the MCP client (optional)")

	// Local SLM CLI Flags
	downloadModel := flag.String("download-model", "", "Download a model from the specified URL")
	listModels := flag.Bool("list-models", false, "List all locally downloaded models")
	modelID := flag.String("model-id", "", "ID for the model being downloaded (default: filename)")

	// Ignore errors from flag parsing as wails dev uses its own flags sometimes
	_ = flag.CommandLine.Parse(os.Args[1:])

	// 3. Optional Early Logging
	if *verbose || *logFile != "" {
		f := initDebugLog(*logFile)
		if f != nil {
			defer f.Close()
		}
	}

	log.Println("Main: Starting Context-Sherpa...")

	// 4. Smart Entry Point: Check for Headless / MCP mode
	if *mcpFlag {
		isMCPMode = true
	} else {
		// Detect pipe/redirect - safely check for stdin presence
		// In Windows GUI mode without a console, Stdin.Stat() can fail
		if fi, err := os.Stdin.Stat(); err == nil && fi != nil {
			if (fi.Mode() & os.ModeCharDevice) == 0 {
				isMCPMode = true
			}
		}
	}

	// 5. Dispatch
	if isMCPMode {
		log.Println("Starting Context-Sherpa in Headless (MCP) Mode...")

		// Use workspaceRoot if provided, otherwise fallback to projectRoot for backward compatibility
		finalWorkspaceRoot := *workspaceRoot
		if finalWorkspaceRoot == "" {
			finalWorkspaceRoot = *projectRoot
		}

		// 2.5 Handle Local SLM CLI commands if in MCP mode (Headless)
		if *listModels || *downloadModel != "" {
			lockPath := mcp.GetHubLockPath() // Using this to find base dir
			// mcp.GetHubLockPath returns the path to hub.lock, we want the directory
			baseDir := filepath.Dir(lockPath)
			modelsDir := filepath.Join(baseDir, "models")

			if *listModels {
				files, _ := os.ReadDir(modelsDir)
				log.Printf("Locally downloaded models in %s:\n", modelsDir)
				for _, f := range files {
					if !f.IsDir() {
						log.Printf("- %s\n", f.Name())
					}
				}
				return
			}

			if *downloadModel != "" {
				id := *modelID
				if id == "" {
					id = filepath.Base(*downloadModel)
				}
				log.Printf("Downloading model %s from %s...\n", id, *downloadModel)
				dl := inference.NewDownloader(modelsDir)
				err := dl.DownloadModel(context.Background(), id, *downloadModel)
				if err != nil {
					log.Fatalf("Download failed: %v\n", err)
				}
				log.Println("Download complete.")
				return
			}
		}

		mcp.Start(finalWorkspaceRoot, *verbose, *logFile, *astGrepPath, *clientName)
		return
	}

	// 3. GUI Mode
	log.Println("Starting Context-Sherpa in GUI (Wails) Mode...")

	// Create an instance of the app structure
	app := NewApp()

	// Load preferences for window state
	prefs := app.GetPreferences()
	if prefs.WindowWidth == 0 {
		prefs.WindowWidth = 1024
	}
	if prefs.WindowHeight == 0 {
		prefs.WindowHeight = 768
	}

	windowStartState := options.Normal
	if prefs.IsMaximized {
		windowStartState = options.Maximised
	}

	// Create application with options
	err := wails.Run(&options.App{
		Title:            "Context Sherpa Hub",
		Width:            prefs.WindowWidth,
		Height:           prefs.WindowHeight,
		WindowStartState: windowStartState,
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		BackgroundColour: &options.RGBA{R: 27, G: 38, B: 54, A: 1},
		OnStartup:        app.startup,
		OnBeforeClose:    app.BeforeClose,
		OnShutdown:       app.Shutdown,
		Bind: []interface{}{
			app,
		},
		Mac: &mac.Options{
			TitleBar: &mac.TitleBar{
				TitlebarAppearsTransparent: true,
				HideTitle:                  true,
				HideTitleBar:               false,
				FullSizeContent:            false,
				UseToolbar:                 false,
				HideToolbarSeparator:       true,
			},
			WebviewIsTransparent: true,
			WindowIsTranslucent:  true,
			About: &mac.AboutInfo{
				Title:   "Context Sherpa Hub",
				Message: "© 2026 Hack After Dark",
				Icon:    icon,
			},
		},
		Linux: &linux.Options{
			Icon:                icon,
			WindowIsTranslucent: false,
			WebviewGpuPolicy:    linux.WebviewGpuPolicyAlways,
			ProgramName:         "context-sherpa-hub",
		},
	})

	if err != nil {
		log.Fatal("Error:", err.Error())
	}
}
