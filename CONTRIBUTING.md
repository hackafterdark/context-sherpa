# Contributing to Context Sherpa

Welcome! We're excited that you're interested in contributing to Context Sherpa. This document provides comprehensive guidelines for contributing to the project, whether through code changes, documentation improvements, or community rule contributions.

## Table of Contents

- [Project Overview](#project-overview)
- [Development Environment Setup](#development-environment-setup)
- [Building and Testing](#building-and-testing)
- [Code Standards and Conventions](#code-standards-and-conventions)
- [Contribution Workflow](#contribution-workflow)
- [Contributing Community Rules](#contributing-community-rules)
- [Reporting Issues](#reporting-issues)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Communication](#communication)
- [Acknowledgments](#acknowledgments)

## Project Overview

Context Sherpa is an AI-powered code analysis MCP server that enables AI coding agents to:
- Lint and validate code using ast-grep rules
- Dynamically create, update, and remove linting rules based on natural language feedback
- Access a community repository of pre-built rules
- Provide a self-contained, cross-platform binary with no external dependencies

The project consists of:
- **MCP Server**: Go-based server providing tools for AI agents
- **Build System**: GitHub Actions workflow for cross-platform binaries
- **Community Rules**: Public repository of shareable ast-grep rules

## Development Environment Setup

### Prerequisites

- **Go 1.21+**: [Install Go](https://golang.org/doc/install)
- **Git**: For version control
- **ast-grep binary**: For local development and testing

### 1. Clone the Repository

```bash
git clone https://github.com/hackafterdark/context-sherpa.git
cd context-sherpa
```

### 2. Set Up the ast-grep Binary

Download the appropriate ast-grep binary for your platform:

1. Visit [ast-grep releases](https://github.com/ast-grep/ast-grep/releases/latest)
2. Download the binary for your OS/architecture
3. Place it in `cmd/server/bin/` with the name `ast-grep`

**Platform-specific instructions:**
- **Linux/macOS**: `ast-grep` (no extension)
- **Windows**: Download `ast-grep.exe`, then rename to `ast-grep` (remove .exe)

### 3. Install Development Dependencies

```bash
# Install Go dependencies
go mod download

# Install development tools (optional but recommended)
go install github.com/cosmtrek/air@latest  # For hot reloading during development
```

### 4. Verify Setup

```bash
# Test that everything works
go test ./...

# Build the project
go build -o context-sherpa ./cmd/server

# Verify the binary
./context-sherpa --help
```

## Building and Testing

### Build Commands

```bash
# Build for current platform
go build -o context-sherpa ./cmd/server

# Build with specific flags
go build -ldflags "-X main.version=$(git describe --tags)" -o context-sherpa ./cmd/server

# Build for multiple platforms (using the project's build system)
go run ./scripts/build/build.go  # If available
```

### Testing

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests for specific package
go test ./internal/mcp/...

# Run tests with verbose output
go test -v ./...

# Run benchmarks
go test -bench=. ./...
```

### Development Workflow

For active development, we recommend using a hot-reload tool:

```bash
# Using air (if installed)
air

# Or manually rebuild on changes
find . -name "*.go" | entr -r go build -o context-sherpa ./cmd/server
```

## Code Standards and Conventions

This project follows the Go community's best practices and conventions.

### General Guidelines

- **Write idiomatic Go**: Follow standard conventions for naming, structure, and style
- **Prefer copying over abstraction**: Small amounts of utility code are better copied than abstracted into shared packages
- **Variable naming**: Use short names for limited scope, longer names for broader scope
- **Code formatting**: All code must be formatted with `go fmt`
- **Documentation**: All exported types, functions, and constants must have godoc-compliant comments

### Logging

- Use `github.com/charmbracelet/log` for all application logging
- Avoid standard `log` or `fmt` packages for application logging
- Log levels should be appropriate for the context

### Testing

- **Unit tests required**: All new functions must have corresponding unit tests
- **Test file naming**: `*_test.go` files alongside the code they test
- **Test function naming**: `Test<FunctionName>` for unit tests
- **Table-driven tests**: Use descriptive names for sub-tests
- **Assertions**: Use `stretchr/testify/assert` or `stretchr/testify/require`

### Comments and Documentation

```go
// MyFunction performs an important operation on the provided data.
// It returns an error if the operation fails due to invalid input.
//
// Parameters:
//   - ctx: The request context for cancellation and timeouts
//   - data: The input data to process
//
// Returns:
//   - result: The processed result
//   - err: Any error that occurred during processing
func MyFunction(ctx context.Context, data InputData) (OutputData, error) {
    // Implementation
}
```

## Contribution Workflow

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally
3. Add the upstream remote:

```bash
git remote add upstream https://github.com/hackafterdark/context-sherpa.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/amazing-feature
# or
git checkout -b fix/bug-description
```

### 3. Make Your Changes

- Follow the code standards outlined above
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- **Plan your work** (especially when using AI): Use the context from the `AGENT_DOCS/` directory and this `CONTRIBUTING.md` file to create or update a plan markdown file documenting your thought process, goals, and implementation approach. This provides valuable historical context for future contributors and helps maintain consistency across the project.
- **Update agent rules if needed**: Consider updating `AGENTS.md` if your changes affect how AI agents should interact with the codebase, such as new patterns to follow, updated conventions, or modified workflows that agents need to be aware of.

### 4. Test Your Changes

```bash
# Run the full test suite
go test ./...

# Check for any linting issues
go vet ./...

# Format your code
go fmt ./...
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "Add: comprehensive feature description

- What the change does
- Why it's needed
- Any breaking changes
- Testing performed"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/amazing-feature
```

Then visit your fork on GitHub and create a pull request against the main branch.

## Contributing Community Rules

The [Context Sherpa Community Rules](https://github.com/hackafterdark/context-sherpa-community-rules) repository is where users can share ast-grep rules for common patterns and best practices.

### Rule Contribution Process

1. **Create Your Rule**: Write an ast-grep rule in YAML format
2. **Add Tests**: Include valid and invalid test cases
3. **Submit PR**: Follow the community repository's contribution guidelines
4. **Automated Validation**: CI will validate your rule against test cases

### Rule Format

```yaml
id: your-rule-id
language: go
author: YourGitHubUsername
message: "Clear description of what this rule catches"
severity: error
metadata:
  tags: ["security", "performance", "style"]
  description: "Detailed explanation of the rule's purpose and examples"
rule:
  pattern: # Your ast-grep pattern here
```

## Reporting Issues

We use GitHub Issues for bug reports and feature requests.

### Bug Reports

When reporting a bug, please include:

- **Clear title**: Summarize the issue
- **Description**: Detailed description of the problem
- **Steps to reproduce**: Exact steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Go version, Context Sherpa version
- **Logs**: Any relevant error messages or logs

### Feature Requests

For feature requests, please include:

- **Use case**: Why this feature would be useful
- **Description**: Detailed description of the proposed feature
- **Examples**: Code examples or mockups if applicable
- **Alternatives**: Alternative solutions you've considered

## Submitting Pull Requests

### PR Requirements

- **Tests**: All new code must include appropriate tests
- **Documentation**: Update documentation for new features
- **Linting**: Code must pass `go vet` and be formatted with `go fmt`
- **Single responsibility**: Each PR should address one issue or feature
- **Descriptive title**: Clear, concise title describing the change

### PR Process

1. **Create PR**: Use GitHub's PR interface
2. **Fill template**: Complete all sections of the PR template
3. **CI checks**: Ensure all CI checks pass
4. **Review**: Address any reviewer feedback
5. **Merge**: Maintainers will merge approved PRs

## Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community discussion
- **Email**: For security issues or private matters

## Acknowledgments

We'd like to thank:

- **ast-grep team**: For creating an amazing tool that powers Context Sherpa's pattern matching capabilities
- **MCP-go contributors**: For the excellent Go implementation of the Model Context Protocol
- **Our community**: For contributing rules, reporting issues, and helping improve the project

---

Thank you for contributing to Context Sherpa! Your efforts help make AI-assisted development safer and more reliable for everyone.