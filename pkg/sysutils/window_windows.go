//go:build windows

package sysutils

import (
	"fmt"
	"strings"
	"syscall"
	"unsafe"
)

func getProcessName(pid int) string {
	cmd := SilentCommand("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH", "/FO", "CSV")
	out, err := cmd.Output()
	if err == nil {
		parts := strings.Split(string(out), ",")
		if len(parts) > 0 {
			return strings.Trim(parts[0], "\"")
		}
	}
	return ""
}

func getParentPid(pid int) int {
	cmd := SilentCommand("powershell", "-NoProfile", "-Command", fmt.Sprintf("(Get-CimInstance Win32_Process -Filter \"ProcessId = %d\").ParentProcessId", pid))
	out, err := cmd.Output()
	if err == nil {
		var ppid int
		fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &ppid)
		return ppid
	}
	return 0
}

func focusWindow(ancestry []int, bestPid int) error {
	return focusWindowsInAncestryWindows(ancestry, bestPid)
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
