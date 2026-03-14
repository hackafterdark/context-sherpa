package sysutils

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"unsafe"
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
	if runtime.GOOS == "windows" {
		cmd := SilentCommand("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH", "/FO", "CSV")
		out, err := cmd.Output()
		if err == nil {
			parts := strings.Split(string(out), ",")
			if len(parts) > 0 {
				return strings.Trim(parts[0], "\"")
			}
		}
	} else {
		out, err := exec.Command("ps", "-p", fmt.Sprintf("%d", pid), "-o", "comm=").Output()
		if err == nil {
			return strings.TrimSpace(string(out))
		}
	}
	return ""
}

// GetParentPid returns the PPID of the given PID.
func GetParentPid(pid int) int {
	if runtime.GOOS == "windows" {
		cmd := SilentCommand("powershell", "-NoProfile", "-Command", fmt.Sprintf("(Get-CimInstance Win32_Process -Filter \"ProcessId = %d\").ParentProcessId", pid))
		out, err := cmd.Output()
		if err == nil {
			var ppid int
			fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &ppid)
			return ppid
		}
	} else {
		out, err := exec.Command("ps", "-p", fmt.Sprintf("%d", pid), "-o", "ppid=").Output()
		if err == nil {
			var ppid int
			fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &ppid)
			return ppid
		}
	}
	return 0
}

// FocusWindow attempts to bring the window(s) associated with the given PIDs to the foreground.
func FocusWindow(ancestry []int, bestPid int) error {
	switch runtime.GOOS {
	case "windows":
		return focusWindowsInAncestryWindows(ancestry, bestPid)
	case "darwin":
		targetPid := bestPid
		if targetPid == 0 && len(ancestry) > 0 {
			targetPid = ancestry[len(ancestry)-1]
		}
		if targetPid == 0 {
			return fmt.Errorf("no target process identified for focus")
		}
		cmd := fmt.Sprintf("tell application \"System Events\" to set frontmost of every process whose unix id is %d to true", targetPid)
		return exec.Command("osascript", "-e", cmd).Run()
	case "linux":
		targetPid := bestPid
		if targetPid == 0 && len(ancestry) > 0 {
			targetPid = ancestry[len(ancestry)-1]
		}
		if targetPid == 0 {
			return fmt.Errorf("no target process identified for focus")
		}
		focusCmd := fmt.Sprintf("wmctrl -ia $(wmctrl -lp | awk -v pid=%d '$3==pid {print $1}')", targetPid)
		err := exec.Command("sh", "-c", focusCmd).Run()
		if err != nil {
			focusCmd = fmt.Sprintf("xdotool windowactivate $(xdotool search --pid %d | tail -1)", targetPid)
			err = exec.Command("sh", "-c", focusCmd).Run()
		}
		return err
	default:
		return fmt.Errorf("platform %s not supported for window focusing", runtime.GOOS)
	}
}

func focusWindowsInAncestryWindows(ancestry []int, bestPid int) error {
	user32 := syscall.NewLazyDLL("user32.dll")
	setForegroundWindow := user32.NewProc("SetForegroundWindow")
	enumWindows := user32.NewProc("EnumWindows")
	getWindowThreadProcessId := user32.NewProc("GetWindowThreadProcessId")
	isWindowVisible := user32.NewProc("IsWindowVisible")
	showWindow := user32.NewProc("ShowWindow")

	var bestHwnd uintptr
	var fallbackHwnd uintptr

	ancestryMap := make(map[int]bool)
	for _, pid := range ancestry {
		ancestryMap[pid] = true
	}

	cb := syscall.NewCallback(func(hwnd uintptr, lparam uintptr) uintptr {
		visible, _, _ := isWindowVisible.Call(hwnd)
		if visible == 0 {
			return 1
		}

		var lpdwProcessId uint32
		getWindowThreadProcessId.Call(hwnd, uintptr(unsafe.Pointer(&lpdwProcessId)))
		pid := int(lpdwProcessId)

		if pid == bestPid {
			bestHwnd = hwnd
			return 0 // found the best one
		}

		if ancestryMap[pid] && fallbackHwnd == 0 {
			fallbackHwnd = hwnd
		}

		return 1
	})

	enumWindows.Call(cb, 0)

	targetHwnd := bestHwnd
	if targetHwnd == 0 {
		targetHwnd = fallbackHwnd
	}

	if targetHwnd != 0 {
		// Restore if minimized (SW_RESTORE = 9)
		showWindow.Call(targetHwnd, 9)
		// Set to foreground
		setForegroundWindow.Call(targetHwnd)
		return nil
	}

	return fmt.Errorf("could not find window for target process")
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
