package main

import (
	_ "embed"

	"github.com/hackafterdark/context-sherpa/internal/mcp"
)

//go:embed bin/ast-grep
var astGrepBinary []byte

func main() {
	mcp.Start(astGrepBinary)
}

// GetAstGrepBinary returns the embedded ast-grep binary for testing
func GetAstGrepBinary() []byte {
	return astGrepBinary
}
