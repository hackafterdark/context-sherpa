package main

import (
	"embed"
	"flag"
	"log"
	"os"

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
