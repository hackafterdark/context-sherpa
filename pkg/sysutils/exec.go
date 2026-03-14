package sysutils

import (
	"os/exec"
	"runtime"
	"syscall"
)

// SilentCommand creates an exec.Cmd configured to run without showing windows on any OS.
// This is primarily a fix for Windows "console flashes," as Unix-like systems run subprocesses silently by default.
func SilentCommand(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)

	if runtime.GOOS == "windows" {
		if cmd.SysProcAttr == nil {
			cmd.SysProcAttr = &syscall.SysProcAttr{}
		}
		// HideWindow hides the window of the created process
		cmd.SysProcAttr.HideWindow = true
		// CREATE_NO_WINDOW prevents the terminal flash on Windows
		// creationFlags: 0x08000000
		cmd.SysProcAttr.CreationFlags = 0x08000000
	}

	return cmd
}
