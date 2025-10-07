package main

import (
	_ "embed"
	"flag"

	"github.com/hackafterdark/context-sherpa/internal/mcp"
)

func main() {
	projectRoot := flag.String("projectRoot", "", "Project root directory (defaults to current working directory)")
	verbose := flag.Bool("verbose", false, "Enable verbose logging for debugging")
	logFile := flag.String("logFile", "", "Path to file where logs will be appended (optional)")
	astGrepPath := flag.String("astGrepPath", "", "Explicit path to ast-grep binary")
	flag.Parse()

	mcp.Start(*projectRoot, *verbose, *logFile, *astGrepPath)
}
