package mcp

import (
	"testing"
)

func TestIsSymbolMatch(t *testing.T) {
	tests := []struct {
		scipSymbol string
		symbolName string
		want       bool
	}{
		// Go style
		{"scip-go gomod github.com/user/project `App`#", "App", true},
		{"scip-go gomod github.com/user/project `NewApp`().", "NewApp", true},
		{"scip-go gomod github.com/user/project `App`#workspaces.", "App", true},
		{"scip-go gomod github.com/user/project `GetWorkspaces`().", "GetWorkspaces", true},

		// Suffixes
		{"github.com/user/project/App#", "App", true},
		{"github.com/user/project/NewApp().", "NewApp", true},

		// Separators
		{"github.com/user/project/App", "App", true},
		{"github.com/user/project.App", "App", true},
		{"github.com/user/project App", "App", true},

		// Negative cases
		{"github.com/user/project/MyApp", "App", false},
		{"github.com/user/project/Apple", "App", false},
		{"App", "App", true},
	}

	for _, tt := range tests {
		t.Run(tt.scipSymbol, func(t *testing.T) {
			if got := isSymbolMatch(tt.scipSymbol, tt.symbolName); got != tt.want {
				t.Errorf("isSymbolMatch() = %v, want %v", got, tt.want)
			}
		})
	}
}
