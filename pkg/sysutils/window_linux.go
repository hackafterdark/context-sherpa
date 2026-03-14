//go:build linux

package sysutils

import (
	"fmt"
	"os/exec"
	"strings"
)

func getProcessName(pid int) string {
	out, err := exec.Command("ps", "-p", fmt.Sprintf("%d", pid), "-o", "comm=").Output()
	if err == nil {
		return strings.TrimSpace(string(out))
	}
	return ""
}

func getParentPid(pid int) int {
	out, err := exec.Command("ps", "-p", fmt.Sprintf("%d", pid), "-o", "ppid=").Output()
	if err == nil {
		var ppid int
		fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &ppid)
		return ppid
	}
	return 0
}

func focusWindow(ancestry []int, bestPid int) error {
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
}
