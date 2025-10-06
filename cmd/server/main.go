package main

import (
	"flag"

	"github.com/hackafterdark/context-sherpa/internal/mcp"
)

func main() {
	projectRoot := flag.String("projectRoot", "", "Project root directory (defaults to current working directory)")
	verbose := flag.Bool("verbose", false, "Enable verbose logging for debugging")
	logFile := flag.String("logFile", "", "File path to write verbose logs to (optional, defaults to context-sherpa.log)")
	flag.Parse()

	mcp.Start(*projectRoot, *verbose, *logFile)
}
