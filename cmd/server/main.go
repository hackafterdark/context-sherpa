package main

import (
	_ "embed"
	"flag"

	"github.com/hackafterdark/context-sherpa/internal/mcp"
)

//go:embed bin/ast-grep
var astGrepBinary []byte

func main() {
	projectRoot := flag.String("projectRoot", "", "Project root directory (defaults to current working directory)")
	verbose := flag.Bool("verbose", false, "Enable verbose logging for debugging")
	logFile := flag.String("logFile", "", "Path to file where logs will be appended (optional)")
	flag.Parse()

	mcp.Start(astGrepBinary, *projectRoot, *verbose, *logFile)
}

// GetAstGrepBinary returns the embedded ast-grep binary for testing
func GetAstGrepBinary() []byte {
	return astGrepBinary
}
