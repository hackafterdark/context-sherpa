package main

import (
	_ "embed"

	"github.com/hackafterdark/context-sherpa/internal/mcp"
)

func main() {
	mcp.Start(astGrepBinary)
}
