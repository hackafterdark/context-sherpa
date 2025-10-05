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
	flag.Parse()

	mcp.Start(astGrepBinary, *projectRoot)
}

// GetAstGrepBinary returns the embedded ast-grep binary for testing
func GetAstGrepBinary() []byte {
	return astGrepBinary
}
