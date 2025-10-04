//go:build windows

package main

import _ "embed"

//go:embed bin/ast-grep.exe
var astGrepBinary []byte
