# Scan Path Tool Implementation Plan

## Overview

This document outlines the implementation plan for a new `scan_path` MCP tool that will complement the existing `scan_code` tool by accepting file paths instead of code content. This improvement addresses token efficiency, context window management, and overall system performance.

## Problem Statement

The current `scan_code` tool requires passing entire file contents as arguments, leading to:
- High token consumption for large files
- Cluttered context windows with verbose tool calls
- Performance overhead in data serialization/deserialization
- Scalability limitations for multi-file operations

## Solution Design

### Tool Name
`scan_path` - A file path-based scanning tool that reads files from disk

### Design Philosophy
- **Efficiency First**: Minimize data transfer between agent and server
- **Flexibility**: Support files, directories, and glob patterns in a single interface
- **Consistency**: Follow existing patterns from `scan_code` tool
- **Safety**: Validate paths and handle errors gracefully

## Tool Specification

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | File path, directory path, or glob pattern to scan |
| `sgconfig` | string | No | Path to specific sgconfig.yml file (defaults to "sgconfig.yml") |
| `language` | string | No | Filter by programming language when scanning directories |

### Behavior

1. **Path Resolution**:
   - Single file: `internal/mcp/server.go`
   - Directory: `internal/mcp/` (recursive scan)
   - Glob pattern: `internal/**/*.go` (pattern matching)

2. **Language Filtering**:
   - When `language` is specified, filter files by extension
   - Supported: `go`, `python`, `javascript`, `typescript`, `rust`, `java`, `cpp`, `c`

3. **Error Handling**:
   - Invalid paths return descriptive error messages
   - Missing sgconfig files provide helpful initialization guidance
   - File permission errors are clearly communicated

### Examples

```bash
# Single file
scan_path(path="internal/mcp/server.go")

# Directory scan
scan_path(path="internal/mcp/")

# Language-filtered directory scan
scan_path(path="internal/", language="go")

# Glob pattern
scan_path(path="**/*.go")

# Custom config
scan_path(path="internal/mcp/", sgconfig="custom-sgconfig.yml")
```

## Implementation Details

**FIRST ITERATION INCLUDES: 1MB file size limit for all scanned files**

### File Structure
- **Location**: `internal/mcp/server.go`
- **Function**: `scanPathHandler(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error)`

### Dependencies
- Uses existing `extractSgBinary()`, `findProjectRoot()` functions
- Leverages `filepath.Walk()` for directory traversal
- Integrates with existing ast-grep binary execution

### Algorithm

**IMPORTANT: The 1MB file size limit will be implemented in the first iteration for safety and stability.**

1. **Parameter Extraction**:
   - Extract `path` (required)
   - Extract `sgconfig` (optional, default: "sgconfig.yml")
   - Extract `language` (optional)

2. **Path Validation**:
   - Check if sgconfig file exists
   - Validate provided path exists and is accessible

3. **File Discovery**:
   - If path is a file: scan directly
   - If path is a directory: walk recursively
   - If path contains glob patterns: expand and match
   - Apply language filtering if specified

4. **File Size Validation** (FIRST ITERATION FEATURE):
   - **Check each file against 1MB size limit**
   - **Skip files exceeding limit with optional warning**
   - **Continue scanning remaining valid files**

5. **Batch Scanning**:
   - Group valid files into reasonable batches
   - Execute `sg scan` commands efficiently
   - Aggregate results

6. **Result Formatting**:
   - Return JSON output matching `scan_code` format
   - Include helpful error messages for failures

### Edge Cases

- **Empty directories**: Return success with no matches
- **Binary files**: Skip or handle gracefully
- **Symlinks**: Follow or detect based on configuration
- **Permission denied**: Clear error messages
- **Very large directories**: Reasonable batching to avoid command length limits
- **Files exceeding 1MB**: Skip gracefully, continue with remaining files

## Integration Strategy

### Server Registration
Add tool definition and handler registration in the `Start()` function alongside existing tools.

### Backward Compatibility
- Existing `scan_code` tool remains unchanged
- New `scan_path` tool is additive
- No breaking changes to current API

### Migration Path
- Both tools coexist during transition period
- Documentation updated to recommend `scan_path` for new use cases
- Consider deprecation timeline for `scan_code` in future versions

## Testing Strategy

### Unit Tests

#### Test Case 1: Single File Scanning
- **Input**: Valid Go file path
- **Expected**: Successful scan with results matching `scan_code` behavior
- **Verification**: Compare output with equivalent `scan_code` call

#### Test Case 2: Directory Scanning
- **Input**: Directory path with multiple files
- **Expected**: All files in directory scanned recursively
- **Verification**: Results include matches from all subdirectory files

#### Test Case 3: Language Filtering
- **Input**: Directory path with mixed file types, language="go"
- **Expected**: Only Go files scanned, other files ignored
- **Verification**: Results contain only Go file matches

#### Test Case 4: Glob Pattern Support
- **Input**: Glob pattern like `**/*.go`
- **Expected**: Files matching pattern scanned across entire project
- **Verification**: Results include matches from all matching files

#### Test Case 5: Error Handling
- **Input**: Non-existent file path
- **Expected**: Clear error message indicating file not found
- **Verification**: Error message provides helpful context

#### Test Case 6: Missing Configuration
- **Input**: Valid path but missing sgconfig.yml
- **Expected**: Error message suggesting `initialize_ast_grep` tool
- **Verification**: Error message matches existing `scan_code` behavior

#### Test Case 7: File Size Limit (FIRST ITERATION FEATURE)
- **Input**: Directory containing mix of files under/over 1MB
- **Expected**: Files over 1MB are skipped, files under 1MB are scanned
- **Verification**: Results only include violations from files under 1MB limit

### Integration Tests

#### Test Case 1: End-to-End Workflow
- Initialize project with `initialize_ast_grep`
- Add test rules with `add_or_update_rule`
- Scan files and verify violations detected
- Clean up rules with `remove_rule`

#### Test Case 2: Performance Comparison
- Compare execution time and token usage between `scan_code` and `scan_path`
- Test with various file sizes (small, medium, large)
- Verify `scan_path` shows measurable improvements

### Test Infrastructure

- **Test File Structure**: Create test files in `internal/mcp/` for unit tests
- **Mock Filesystem**: Use `fstest.MapFS` for reliable, isolated testing
- **Benchmark Tests**: Include performance benchmarks for comparison

## Success Metrics

### Performance Improvements
- **Token Reduction**: Target 80-90% reduction in token usage for typical files
- **Context Window**: Measurable reduction in tool call verbosity
- **Execution Speed**: Comparable or improved execution time

### Reliability
- **Test Coverage**: >90% code coverage for new functionality
- **Error Rate**: Zero unexpected failures in normal usage scenarios
- **Edge Case Handling**: Graceful handling of all identified edge cases

## Rollout Plan

### Phase 1: Development & Testing
- Implement core functionality
- Comprehensive unit and integration testing
- Performance benchmarking and optimization

### Phase 2: Documentation & Examples
- Update tool documentation
- Add usage examples to `examples/` directory
- Create migration guide for existing workflows

### Phase 3: Deployment
- Deploy new server binary with both tools
- Update client integrations to use `scan_path` where beneficial
- Monitor performance metrics and user feedback

### Phase 4: Optimization (Future)
- Gather real-world usage data
- Identify additional optimization opportunities
- Consider `scan_code` deprecation timeline

## Risk Assessment

### Technical Risks
- **Path Resolution Complexity**: Medium - Requires careful handling of different path types
- **Performance Regression**: Low - Design explicitly optimizes for efficiency
- **Integration Issues**: Low - Additive change with clear boundaries

### Operational Risks
- **Testing Coverage**: Medium - Requires comprehensive testing across platforms
- **User Adoption**: Low - Provides clear benefits with minimal disruption

## Future Enhancements

- **Batch Processing**: Advanced batching strategies for very large codebases
- **Incremental Scanning**: Skip previously scanned, unchanged files
- **Parallel Processing**: Concurrent scanning for improved performance
- **Caching**: Cache scan results for frequently accessed files
- **Remote Files**: Support for scanning files from remote sources (HTTP, Git, etc.)

## Additional Considerations

### Performance Optimizations

#### Scanning Strategies
- **Incremental Scanning**: Add file modification time checking to skip unchanged files
- **Parallel Processing**: Process multiple files concurrently when scanning directories
- **Batch Processing**: Group files into optimal batch sizes to minimize command overhead
- **Memory Management**: Stream large files instead of loading entirely into memory

#### Caching Opportunities
- **Result Caching**: Cache scan results for frequently accessed files (with TTL)
- **File Metadata Caching**: Cache file language detection and basic metadata
- **Rule Caching**: Cache parsed rule configurations to avoid repeated YAML parsing

### Security Considerations

#### Path Traversal Protection
- Validate all input paths to prevent directory traversal attacks
- Implement path normalization and sanitization
- Restrict scanning to project directory and subdirectories

#### Permission Handling
- Graceful handling of permission-denied errors
- Clear error messages that don't expose sensitive path information
- Optional verbose mode for debugging access issues

### Monitoring & Observability

#### Logging Strategy
- Structured logging for scan operations with timing information
- Error tracking with context about failed files/paths
- Performance metrics (files scanned, time taken, violations found)

#### Metrics Collection
- Track tool usage patterns (single files vs directories vs globs)
- Monitor performance characteristics across different file sizes
- Error rate tracking and common failure modes

### Advanced Configuration Options

#### Optional Parameters (Future)
- `maxFileSize`: Skip files larger than specified size (default: 1MB)
- `excludePatterns`: Glob patterns for files to exclude
- `includePatterns`: Glob patterns for files to include (when scanning directories)
- `followSymlinks`: Whether to follow symbolic links
- `maxDepth`: Maximum directory depth for recursive scanning

#### Configuration File Support
- Support for `.ast-grep-scan.json` configuration file
- Project-specific scanning preferences
- Team-wide scanning standards

### Documentation Strategy

#### User Documentation
- Tool reference with parameter descriptions and examples
- Migration guide from `scan_code` to `scan_path`
- Best practices for different use cases (single file, directory, CI/CD)

#### Developer Documentation
- API documentation for tool integration
- Performance tuning guide
- Troubleshooting common issues

### Troubleshooting Guide

#### Common Issues
1. **"Path not found" errors**
   - Check file/directory exists and is accessible
   - Verify working directory context

2. **Permission denied errors**
   - Check file permissions
   - Verify user has read access to target paths

3. **No results returned**
   - Verify rules are properly configured
   - Check if language filtering is too restrictive
   - Ensure sgconfig.yml exists and is valid

4. **Performance issues**
   - Consider language filtering for large directories
   - Use specific paths instead of broad globs when possible
   - Files larger than 1MB are automatically skipped for safety

### Dependencies & Prerequisites

#### System Requirements
- Read access to target files and directories
- Valid `sgconfig.yml` configuration file
- ast-grep binary available in system PATH or embedded

#### Project Setup
- ast-grep project must be initialized (`initialize_ast_grep` tool)
- Rules must be configured in the specified rule directories
- Project root must be detectable (presence of `sgconfig.yml`)

### API Evolution Strategy

#### Version Compatibility
- Maintain backward compatibility with existing `scan_code` format
- Follow semantic versioning for any breaking changes
- Provide deprecation warnings for outdated patterns

#### Future Enhancements
- **Streaming Results**: Support for processing large result sets incrementally
- **Remote Scanning**: Ability to scan files from remote sources (HTTP, Git, etc.)
- **Custom Output Formats**: Support for formats other than JSON (SARIF, JUnit XML, etc.)
- **Rule Introspection**: Provide metadata about which rules were applied

### Real-World Usage Scenarios

#### Development Workflow
```bash
# Quick single file check during development
scan_path(path="internal/mcp/server.go")

# Pre-commit hook: scan modified files only
scan_path(path="internal/mcp/")

# Code review: scan entire feature branch
scan_path(path="internal/**/*.go", language="go")
```

#### CI/CD Integration
```bash
# Scan entire codebase for security issues
scan_path(path="./", excludePatterns="vendor/**,node_modules/**")

# Quality gate: fail build if errors found
scan_path(path="./src/") | jq '.[] | select(.severity=="error")'
```

#### Code Quality Analysis
```bash
# Multi-language project scan
scan_path(path="./", language="go")
scan_path(path="./", language="python")
scan_path(path="./", language="javascript")
```

This plan provides a comprehensive roadmap for implementing the `scan_path` tool while ensuring quality, performance, and maintainability.