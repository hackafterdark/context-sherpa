package main

import (
	_ "embed"

	"github.com/hackafterdark/context-sherpa/internal/mcp"
)

//go:embed bin/sg
var sgBinary []byte

func main() {
	mcp.Start(sgBinary)
}
