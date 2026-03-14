package sysutils

import (
	"runtime"
	"strings"
)

// List of recognized editor process names (substring match, case-insensitive)
var Editors = []string{"code", "cursor", "vscodium", "windsurf", "zed", "sublime", "notepad"}

// IsEditor checks if a process name matches a known editor.
func IsEditor(name string) bool {
	nameLower := strings.ToLower(name)
	for _, ed := range Editors {
		if strings.Contains(nameLower, ed) {
			return true
		}
	}
	return false
}

// GetProcessName returns the name of the process with the given PID.
func GetProcessName(pid int) string {
	return getProcessName(pid)
}

// GetParentPid returns the PPID of the given PID.
func GetParentPid(pid int) int {
	return getParentPid(pid)
}

// FocusWindow attempts to bring the window(s) associated with the given PIDs to the foreground.
func FocusWindow(ancestry []int, bestPid int) error {
	return focusWindow(ancestry, bestPid)
}

// GetWindowFocusError returns a platform-specific error message for missing tools.
func GetWindowFocusError() string {
	switch runtime.GOOS {
	case "linux":
		return "Please install 'wmctrl' or 'xdotool' to enable automatic window focusing on Linux."
	default:
		return ""
	}
}
