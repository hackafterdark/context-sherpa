//go:build windows

package sysutils

import (
	"os/exec"
	"syscall"
)

// applySilentAttributes configures the command to run without a window on Windows.
func applySilentAttributes(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	// HideWindow hides the window of the created process
	cmd.SysProcAttr.HideWindow = true
	// CREATE_NO_WINDOW prevents the terminal flash on Windows
	// creationFlags: 0x08000000
	cmd.SysProcAttr.CreationFlags = 0x08000000
}
