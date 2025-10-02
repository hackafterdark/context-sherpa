package main

import (
	_ "embed"

	"ast-grep-linter-mcp-server/internal/mcp"
)

//go:embed bin/sg
var sgBinary []byte

func main() {
	mcp.Start(sgBinary)
}
