//go:build !windows

package sysutils

import (
	"os/exec"
)

// applySilentAttributes is a no-op on non-Windows platforms.
func applySilentAttributes(cmd *exec.Cmd) {
	// Unix-like systems run subprocesses silently by default.
}
