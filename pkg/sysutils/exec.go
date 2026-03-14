package sysutils

import (
	"os/exec"
)

// SilentCommand creates an exec.Cmd configured to run without showing windows on any OS.
// This is primarily a fix for Windows "console flashes," as Unix-like systems run subprocesses silently by default.
func SilentCommand(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	applySilentAttributes(cmd)
	return cmd
}
