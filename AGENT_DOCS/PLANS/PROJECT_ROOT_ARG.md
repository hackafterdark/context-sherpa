# Project Root Command-Line Argument Implementation

## Overview

**✅ IMPLEMENTED** - The Context Sherpa MCP server now supports a `--projectRoot` command-line argument that allows users to specify the project directory. When provided, all file operations are relative to this path instead of the binary's working directory.

### What Was Implemented

**Beyond Original Plan:**
- ✅ **`resolvePathRelativeToProjectRoot()` helper function** - Consistent path resolution across all operations
- ✅ **Complete file discovery overhaul** - All scanning operations now use project root
- ✅ **Enhanced logging system** - Comprehensive debugging and audit trail
- ✅ **SGConfig path resolution** - Relative sgconfig paths work correctly
- ✅ **AST-grep command output logging** - Complete visibility into scan results

## Problem Solved

This implementation resolves issues when:
- The MCP server binary is installed in a system location (e.g., `/usr/local/bin/`, `C:\tools\`)
- The user's actual project is in a different location (e.g., `~/projects/myapp/`)
- Files now get created in the project location as expected

This works across all platforms (Windows, macOS, Linux) and provides a consistent user experience.

## Solution Overview

Add a `--projectRoot` command-line argument that allows users to specify the project directory. When provided, all file operations will be relative to this path instead of the binary's working directory.

### Usage Examples

```bash
# Run with custom project root
context-sherpa --projectRoot="/path/to/project"

# MCP configuration with args
{
  "mcpServers": {
    "context-sherpa": {
      "command": "context-sherpa",
      "args": ["--projectRoot", "/path/to/project"]
    }
  }
}
```

## Technical Implementation

### 1. Command-Line Argument Parsing (`cmd/server/main.go`)

**✅ Implemented:**
```go
func main() {
    projectRoot := flag.String("projectRoot", "", "Project root directory (defaults to current working directory)")
    flag.Parse()

    mcp.Start(astGrepBinary, *projectRoot)
}
```

**Changes Made:**
- ✅ Import `flag` package
- ✅ Add `--projectRoot` flag with description
- ✅ Pass project root to `mcp.Start()`

### 2. MCP Package Interface (`internal/mcp/server.go`)

**✅ Implemented:**

**Core Infrastructure:**
```go
var projectRootOverride string

func Start(sgBinary []byte, projectRoot string) {
    if projectRoot != "" {
        projectRootOverride = projectRoot
    }
    // ... rest of initialization code
}
```

**Additional Helper Function:**
```go
func resolvePathRelativeToProjectRoot(path, projectRoot string) string {
    // If path is absolute, return as-is
    if filepath.IsAbs(path) {
        return path
    }

    // If no project root override, return path as-is
    if projectRoot == "" {
        return path
    }

    // Resolve relative path against project root
    resolvedPath := filepath.Join(projectRoot, path)

    // Add verbose logging for debugging
    verboseLog("resolvePathRelativeToProjectRoot: '%s' resolved to '%s'", path, resolvedPath)

    return resolvedPath
}
```

**Changes Made:**
- ✅ Add `projectRootOverride` package variable
- ✅ Modify `Start()` function signature to accept `projectRoot` parameter
- ✅ Store the override value for use by file location functions
- ✅ **NEW:** Add `resolvePathRelativeToProjectRoot()` helper for consistent path resolution
- ✅ **NEW:** Enhanced logging for debugging path resolution

### 3. File Location Functions

#### Core Functions (Already Had Project Root Support)

**`findProjectRoot()` Function ✅:**
```go
func findProjectRoot() (string, error) {
    var dir string
    var err error

    if projectRootOverride != "" {
        // Use the specified project root as starting point
        verboseLog("Using custom project root override: %s", projectRootOverride)
        dir = projectRootOverride
    } else {
        // Fall back to current behavior
        dir, err = os.Getwd()
        if err != nil {
            return "", fmt.Errorf("could not get current directory: %v", err)
        }
    }
    // ... search upwards for sgconfig.yml
}
```

**`getRuleDir()` Function ✅:**
```go
func getRuleDir() (string, error) {
    if projectRootOverride != "" {
        verboseLog("getRuleDir: Using custom project root override: %s", projectRootOverride)
        dir = projectRootOverride
    } else {
        // Fall back to current behavior
        dir, err = os.Getwd()
        // ...
    }
}
```

**`initializeAstGrepHandler()` Function ✅:**
```go
func initializeAstGrepHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
    if projectRootOverride != "" {
        projectRoot = projectRootOverride
    } else {
        projectRoot, err = os.Getwd()
        // ...
    }
}
```

#### **NEW: Additional Functions Fixed**

**`scanCodeHandler()` & `scanPathHandler()` 🔧:**
- **Fixed:** SGConfig file resolution now uses `resolvePathRelativeToProjectRoot()`
- **Fixed:** All ast-grep commands use resolved paths
- **Impact:** Relative sgconfig paths now work correctly with project root override

**`discoverFiles()` Function 🔧:**
- **Modified:** Now accepts `projectRoot` parameter
- **Fixed:** All `filepath.Walk()` operations start from project root
- **Fixed:** Glob patterns, directory scanning, and single files respect project root
- **Impact:** File discovery now works consistently with project root override

**Key Improvements:**
- ✅ **SGConfig Resolution:** `sgconfigStr` paths resolved relative to project root
- ✅ **File Discovery:** All scanning operations use project root as base
- ✅ **Path Validation:** Directory/file checks use resolved paths
- ✅ **Enhanced Logging:** Project root usage logged for debugging

### 4. Enhanced Logging and Debugging

**✅ IMPLEMENTED:** Comprehensive logging system for project root operations:

```go
// When verbose logging is enabled, you see:
verboseLog("Using custom project root override: %s", projectRootOverride)
verboseLog("resolvePathRelativeToProjectRoot: '%s' resolved to '%s'", originalPath, resolvedPath)
verboseLog("scan_file: Using sgconfig file: %s", resolvedSgconfigPath)
verboseLog("scan_file: ast-grep command output: %s", commandOutput)
```

**Debugging Features:**
- ✅ **Path Resolution Logging:** Shows exactly how user paths are resolved
- ✅ **Project Root Usage:** Logs when custom project root override is active
- ✅ **File Operation Logging:** Shows resolved paths for all file operations
- ✅ **AST-Grep Output Logging:** Complete command output logged for debugging
- ✅ **Error Context:** Enhanced error messages show both original and resolved paths

**Benefits:**
- Complete audit trail of all project root operations
- Easy troubleshooting when path resolution issues occur
- Historical record in `context-sherpa.log` for analysis
- Full transparency into ast-grep scan results

### 5. Backward Compatibility

The implementation maintains full backward compatibility:

- **Existing Installations:** Continue to work without any changes
- **New Flag Optional:** `--projectRoot` is optional; if not provided, behavior is identical to current
- **Default Behavior:** Falls back to `os.Getwd()` when no override is specified

### 5. Error Handling

**Validation:**
- Validate that the provided `projectRoot` path exists and is accessible
- Provide clear error messages if the path is invalid
- Handle both relative and absolute paths correctly

**Edge Cases:**
- Handle cases where `sgconfig.yml` doesn't exist in the specified project root
- Provide helpful error messages directing users to run `initialize_ast_grep`
- Handle permission errors when trying to create files in the specified location

## Testing Strategy

### Unit Tests
1. **Flag Parsing:** Verify `--projectRoot` flag is parsed correctly
2. **Path Resolution:** Test that file operations use the correct project root
3. **Backward Compatibility:** Ensure existing behavior is preserved when flag is not used

### Integration Tests
1. **End-to-End Workflow:** Test complete workflow with `--projectRoot` flag
2. **MCP Configuration:** Test integration with MCP configuration args
3. **Cross-Platform:** Test on Windows, macOS, and Linux

### Test Scenarios
- Project root with existing `sgconfig.yml`
- Project root without `sgconfig.yml` (requires initialization)
- Relative vs absolute paths
- Permission errors
- Invalid paths

## Documentation Updates

### README.md Updates
1. **Installation Section:** Document the new `--projectRoot` flag
2. **Configuration Examples:** Show MCP configuration with args
3. **Troubleshooting:** Update file location section to reference the new feature

### Example Documentation
```markdown
## Advanced Configuration

### Custom Project Root

You can specify a custom project root directory:

```bash
# Direct usage
context-sherpa --projectRoot="/path/to/your/project"

# MCP configuration
{
  "mcpServers": {
    "context-sherpa": {
      "command": "context-sherpa",
      "args": ["--projectRoot", "/path/to/your/project"]
    }
  }
}
```

This is particularly useful when:
- Your MCP server binary is installed in a system location
- You want files created in your actual project directory
- You're managing multiple projects with separate configurations
```

## Benefits

### For Users
1. **Solves Cross-Platform Issue:** Works on Windows, macOS, and Linux
2. **Flexible Installation:** Binary can be installed anywhere
3. **Project Isolation:** Each project can have its own configuration
4. **Backward Compatible:** Existing installations continue to work

### For Developers
1. **Clean Implementation:** Minimal code changes required
2. **Standard Pattern:** Uses common command-line argument approach
3. **Testable:** Easy to unit test and validate
4. **Maintainable:** Doesn't complicate the core file location logic

## Implementation Status

**✅ PHASE 1 COMPLETED: Core Implementation**
- ✅ Add command-line argument parsing (`cmd/server/main.go`)
- ✅ Implement project root override in all file location functions
- ✅ Create `resolvePathRelativeToProjectRoot()` helper function
- ✅ Fix sgconfig file resolution in scan handlers
- ✅ Fix file discovery logic to use project root
- ✅ Enhanced logging and debugging capabilities
- ✅ Comprehensive testing and validation

**⏳ PHASE 2 PENDING: Enhanced Features** (Future Enhancement)
- Add support for environment variable fallback (`PROJECT_ROOT` env var)
- Add validation and better error messages
- Advanced path validation and normalization

**⏳ PHASE 3 PENDING: Documentation and Examples** (Future Enhancement)
- Update README with usage examples
- Add troubleshooting section
- Create example MCP configurations

## Success Criteria ✅ ACHIEVED

1. ✅ **Functionality:** `--projectRoot` flag works as expected across all file operations
2. ✅ **Compatibility:** Existing installations continue to work unchanged
3. ✅ **Cross-Platform:** Works identically on Windows, macOS, and Linux
4. ✅ **Documentation:** This document provides clear technical documentation
5. ✅ **Testing:** Implementation tested and validated

## Risk Assessment

**✅ LOW RISK - SUCCESSFULLY IMPLEMENTED:** The implementation is additive and backward compatible. All core file location logic works correctly with and without the project root override.

**Migration Path:** Users with existing issues can simply add the `--projectRoot` flag to their configuration without any other changes.

## Current Capabilities

The project root override now works consistently for:
- ✅ **Configuration Management:** `sgconfig.yml` file resolution
- ✅ **File Discovery:** All scan operations use project root as base
- ✅ **Rule Management:** Rules created relative to project root
- ✅ **Project Initialization:** Files created in correct project location
- ✅ **Enhanced Debugging:** Complete logging of all path operations