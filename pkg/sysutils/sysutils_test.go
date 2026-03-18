package sysutils

import (
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsEditor(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{"VSCode", "code", true},
		{"Cursor", "cursor", true},
		{"VSCodium", "vscodium", true},
		{"Sublime", "sublime", true},
		{"Notepad", "notepad", true},
		{"Chrome", "chrome", false},
		{"Slack", "slack", false},
		{"Uppercase Code", "CODE", true},
		{"Editor in path", "/usr/bin/code", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, IsEditor(tt.input))
		})
	}
}

func TestGetWindowFocusError(t *testing.T) {
	err := GetWindowFocusError()
	if runtime.GOOS == "linux" {
		assert.Contains(t, err, "Please install 'wmctrl' or 'xdotool'")
	} else {
		assert.Empty(t, err)
	}
}

func TestSilentCommand(t *testing.T) {
	cmd := SilentCommand("ls", "-la")
	assert.NotNil(t, cmd)
	// exec.Command resolves "ls" to an absolute path (e.g., /usr/bin/ls)
	assert.True(t, strings.HasSuffix(cmd.Path, "ls"))
	assert.Equal(t, []string{"ls", "-la"}, cmd.Args)
}

func TestProcessInfo(t *testing.T) {
	pid := os.Getpid()
	
	name := GetProcessName(pid)
	assert.NotEmpty(t, name, "Process name should not be empty for current PID")
	
	ppid := GetParentPid(pid)
	assert.NotZero(t, ppid, "Parent PID should not be zero for current PID")
	
	// Parent of current process should also have a name
	parentName := GetProcessName(ppid)
	assert.NotEmpty(t, parentName, "Parent process name should not be empty")
}

func TestFocusWindow(t *testing.T) {
	// Testing valid focus is hard without GUI, but we can test invalid cases
	err := FocusWindow([]int{}, 0)
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "no target process identified") || strings.Contains(err.Error(), "could not find window"), "Error message should mention missing target or window")
	
	// Test with a PID that likely doesn't have a window (current PID in CI/CLI)
	// This might fail if wmctrl/xdotool isn't installed, but FocusWindow should return that error
	err = FocusWindow([]int{os.Getpid()}, 0)
	if err != nil {
		// If it errors, it should be because tools are missing or no window found
		t.Logf("FocusWindow returned expected error or no-op: %v", err)
	}
}
