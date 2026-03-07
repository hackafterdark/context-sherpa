package main

import (
	"context"
	"embed"
	"flag"
	"log"
	"os"
	"path/filepath"

	"github.com/hackafterdark/context-sherpa/pkg/inference"
	"github.com/hackafterdark/context-sherpa/pkg/mcp"
	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/mac"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	// 1. Smart Entry Point: Check for Headless / MCP mode
	isMCPMode := false
	mcpFlag := flag.Bool("mcp", false, "Run in headless MCP server mode")
	
	// Also parse legacy flags from cmd/context-sherpa/main.go
	workspaceRoot := flag.String("workspaceRoot", "", "Workspace root directory (defaults to current working directory)")
	projectRoot := flag.String("projectRoot", "", "Workspace root directory (legacy alias for workspaceRoot)")
	verbose := flag.Bool("verbose", false, "Enable verbose logging for debugging")
	logFile := flag.String("logFile", "", "Path to file where logs will be appended (optional)")
	astGrepPath := flag.String("astGrepPath", "", "Explicit path to ast-grep binary")
	clientName := flag.String("client", "", "Name of the MCP client (optional)")
	
	// Little Brain CLI Flags
	downloadModel := flag.String("download-model", "", "Download a model from the specified URL")
	listModels := flag.Bool("list-models", false, "List all locally downloaded models")
	modelID := flag.String("model-id", "", "ID for the model being downloaded (default: filename)")
	
	// Ignore errors from flag parsing as wails dev uses its own flags sometimes
	// We parse os.Args manually or use flag.CommandLine.Parse
	_ = flag.CommandLine.Parse(os.Args[1:])

	if *mcpFlag {
		isMCPMode = true
	} else {
		// Detect pipe/redirect
		fi, _ := os.Stdin.Stat()
		if (fi.Mode() & os.ModeCharDevice) == 0 {
			isMCPMode = true
		}
	}

	// 2. Dispatch
	if isMCPMode {
		log.Println("Starting Context-Sherpa in Headless (MCP) Mode...")
		
		// Use workspaceRoot if provided, otherwise fallback to projectRoot for backward compatibility
		finalWorkspaceRoot := *workspaceRoot
		if finalWorkspaceRoot == "" {
			finalWorkspaceRoot = *projectRoot
		}
		
		// 2.5 Handle Little Brain CLI commands if in MCP mode (Headless)
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

	// Create application with options
	err := wails.Run(&options.App{
		Title:  "Context-Sherpa Hub",
		Width:  1024,
		Height: 768,
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		BackgroundColour: &options.RGBA{R: 27, G: 38, B: 54, A: 1},
		OnStartup:        app.startup,
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
				Title:   "Context-Sherpa",
				Message: "© 2026 Hack After Dark",
			},
		},
	})

	if err != nil {
		log.Fatal("Error:", err.Error())
	}
}
