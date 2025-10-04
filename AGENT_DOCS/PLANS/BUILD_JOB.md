# Multi-Architecture Build Plan for context-sherpa

**Version: 1.0**
**Date: 2025-10-04**
**Author: Roo (AI Architect)**

## Overview

This document outlines a comprehensive build strategy for creating portable, cross-platform binaries of the context-sherpa MCP server. The build process will support multiple architectures while maintaining the project's core requirement of a single, self-contained executable with no external dependencies.

## Architecture Support Matrix

The build system will target the following architectures:

| Platform | Architecture | Go Target | ast-grep Release Asset |
|----------|--------------|-----------|------------------------|
| Linux | x86_64 | `linux/amd64` | `app-x86_64-unknown-linux-gnu.zip` |
| Linux | ARM64 | `linux/arm64` | `app-aarch64-unknown-linux-gnu.zip` |
| macOS | Intel | `darwin/amd64` | `app-x86_64-apple-darwin.zip` |
| macOS | Apple Silicon | `darwin/arm64` | `app-aarch64-apple-darwin.zip` |
| Windows | x86_64 | `windows/amd64` | `app-x86_64-pc-windows-msvc.zip` |
| Windows | ARM64 | `windows/arm64` | `app-aarch64-pc-windows-msvc.zip` |

## Build Architecture Design

### Core Components

1. **Binary Download Module**: Automated downloading and verification of ast-grep binaries from GitHub releases
2. **Cross-Platform Build System**: Makefile-based build orchestration with architecture-specific logic
3. **CI/CD Pipeline**: GitHub Actions workflow for automated builds and releases
4. **Artifact Management**: Automated attachment of build artifacts to GitHub releases

### Key Design Principles

- **Single Source of Truth**: All build configuration centralized in Makefile
- **Reproducible Builds**: Consistent build environment and dependency management
- **Fail-Fast Testing**: Run tests early in the build pipeline
- **Immutable Artifacts**: Each build produces verifiable, immutable binaries
- **Semantic Versioning**: Automated versioning based on Git tags

## Implementation Plan

### Phase 1: Local Build Infrastructure

#### 1.1 Makefile Creation

Create a comprehensive `Makefile` with the following targets:

```makefile
# Core build targets
.PHONY: all build clean test release local-release

# Variables
BINARY_NAME := context-sherpa
VERSION ?= $(shell git describe --tags --always --dirty)
BUILD_DIR := build
AST_GREP_VERSION := latest
GO_FLAGS := -ldflags "-X main.version=$(VERSION) -s -w"

# Platform-specific configurations
PLATFORMS := \
    linux-amd64 \
    linux-arm64 \
    darwin-amd64 \
    darwin-arm64 \
    windows-amd64 \
    windows-arm64

AST_GREP_URLS := \
    linux-amd64=https://github.com/ast-grep/ast-grep/releases/download/$(AST_GREP_VERSION)/app-x86_64-unknown-linux-gnu.zip \
    linux-arm64=https://github.com/ast-grep/ast-grep/releases/download/$(AST_GREP_VERSION)/app-aarch64-unknown-linux-gnu.zip \
    darwin-amd64=https://github.com/ast-grep/ast-grep/releases/download/$(AST_GREP_VERSION)/app-x86_64-apple-darwin.zip \
    darwin-arm64=https://github.com/ast-grep/ast-grep/releases/download/$(AST_GREP_VERSION)/app-aarch64-apple-darwin.zip \
    windows-amd64=https://github.com/ast-grep/ast-grep/releases/download/$(AST_GREP_VERSION)/app-x86_64-pc-windows-msvc.zip \
    windows-arm64=https://github.com/ast-grep/ast-grep/releases/download/$(AST_GREP_VERSION)/app-aarch64-pc-windows-msvc.zip

# Main targets
all: build

build: $(PLATFORMS)

$(PLATFORMS): %-amd64 %-arm64
    @echo "Building for $*"

test:
    go test ./...

clean:
    rm -rf $(BUILD_DIR)
    rm -f $(BINARY_NAME)*

# Release targets
release: test clean
    @echo "Creating release builds..."
    $(MAKE) build
    $(MAKE) package-release

local-release: test clean
    $(MAKE) build-current-arch
    $(MAKE) package-local

# Utility targets
build-current-arch:
    $(eval ARCH := $(shell go env GOARCH))
    $(eval OS := $(shell go env GOOS))
    $(MAKE) $(OS)-$(ARCH)

package-release:
    @echo "Packaging release artifacts..."
    # Create checksums, archives, etc.

package-local:
    @echo "Packaging for local use..."
```

#### 1.2 Build Script Enhancements

**Directory Structure:**
```
context-sherpa/
├── Makefile                    # Main build orchestration
├── scripts/
│   ├── build/
│   │   ├── download_ast_grep.sh    # Binary download utility
│   │   ├── verify_integrity.sh     # Checksum verification
│   │   └── embed_binary.go         # Go binary embedding tool
│   └── release/
│       ├── create_checksums.sh     # SHA256 generation
│       └── package_artifacts.sh    # Archive creation
└── cmd/server/bin/             # ast-grep binaries (populated during build)
```

**Key Scripts:**

1. **Binary Download (`scripts/build/download_ast_grep.sh`)**:
   ```bash
   #!/bin/bash
   set -euo pipefail

   PLATFORM="$1"
   AST_GREP_VERSION="${2:-latest}"
   BIN_DIR="cmd/server/bin"

   # Download and extract ast-grep binary for platform
   # Verify checksums
   # Place in correct directory for embedding
   ```

2. **Build Orchestration**:
   - Automated dependency verification
   - Parallel build execution where possible
   - Clear error reporting and rollback on failure

#### 1.3 Local Development Workflow

```bash
# Quick local build
make build-current-arch

# Full test suite
make test

# Release build (all platforms)
make release

# Development cycle
make clean
go mod tidy
make test
make build-current-arch
```

### Phase 2: CI/CD Pipeline

#### 2.1 GitHub Actions Workflow

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v4
        with:
          go-version-file: go.mod

      - name: Run tests
        run: go test ./...

      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results
          path: test-results/

  build:
    name: Build for ${{ matrix.os }}-${{ matrix.arch }}
    needs: test
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            goos: linux
            goarch: amd64
            ast_grep_asset: app-x86_64-unknown-linux-gnu.zip
          - os: ubuntu-latest
            goos: linux
            goarch: arm64
            ast_grep_asset: app-aarch64-unknown-linux-gnu.zip
          - os: macos-latest
            goos: darwin
            goarch: amd64
            ast_grep_asset: app-x86_64-apple-darwin.zip
          - os: macos-latest
            goos: darwin
            goarch: arm64
            ast_grep_asset: app-aarch64-apple-darwin.zip
          - os: windows-latest
            goos: windows
            goarch: amd64
            ast_grep_asset: app-x86_64-pc-windows-msvc.zip
          - os: windows-latest
            goos: windows
            goarch: arm64
            ast_grep_asset: app-aarch64-pc-windows-msvc.zip
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Go
        uses: actions/setup-go@v4
        with:
          go-version-file: go.mod

      - name: Download ast-grep binary
        run: |
          curl -L -o ast-grep.zip https://github.com/ast-grep/ast-grep/releases/latest/download/${{ matrix.ast_grep_asset }}
          # Extract and place in cmd/server/bin/

      - name: Build binary
        run: |
          go build -o context-sherpa-${{ matrix.goos }}-${{ matrix.goarch }} ./cmd/server

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: binaries
          path: context-sherpa-*

  create-release:
    name: Create Release
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download all artifacts
        uses: actions/download-artifact@v3
        with:
          name: binaries
          path: binaries/

      - name: Generate checksums
        run: |
          cd binaries
          sha256sum context-sherpa-* > checksums.txt

      - name: Create release archive
        run: |
          tar -czf context-sherpa-${{ github.ref_name }}.tar.gz binaries/

      - name: Create GitHub release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            binaries/context-sherpa-*
            binaries/checksums.txt
            context-sherpa-${{ github.ref_name }}.tar.gz
          generate_release_notes: true
```

#### 2.2 Pipeline Features

1. **Automated Testing**: Runs full test suite before any builds
2. **Parallel Builds**: All platforms build simultaneously
3. **Artifact Collection**: Centralized artifact management
4. **Checksum Generation**: SHA256 verification for all binaries
5. **Release Notes**: Automated generation from commit history
6. **Tag-Based Triggering**: Only runs on version tags (v*.*.*)

### Phase 3: Artifact Management

#### 3.1 Release Assets

Each release will include:

```
context-sherpa-v1.0.0/
├── binaries/
│   ├── context-sherpa-linux-amd64
│   ├── context-sherpa-linux-arm64
│   ├── context-sherpa-darwin-amd64
│   ├── context-sherpa-darwin-arm64
│   ├── context-sherpa-windows-amd64.exe
│   └── context-sherpa-windows-arm64.exe
├── checksums.txt
└── context-sherpa-v1.0.0.tar.gz
```

#### 3.2 Naming Convention

- **Binary Names**: `context-sherpa-{os}-{arch}`
- **Windows Extension**: `.exe` suffix for Windows binaries
- **Version Integration**: Embed version in binary metadata
- **Checksum Files**: `checksums.txt` with SHA256 sums

### Phase 4: Quality Assurance

#### 4.1 Testing Integration

- **Unit Tests**: All packages tested before builds
- **Integration Tests**: End-to-end MCP server validation
- **Platform Tests**: Basic smoke tests for each binary
- **Security Scanning**: Binary vulnerability assessment

#### 4.2 Build Verification

```yaml
# Post-build verification
- name: Verify binary
  run: |
    ./context-sherpa-${{ matrix.goos }}-${{ matrix.goarch }} --version
    file context-sherpa-${{ matrix.goos }}-${{ matrix.goarch }}
```

## Usage Instructions

### For Developers

```bash
# Set up development environment
make setup

# Run tests
make test

# Build for current architecture
make build-current-arch

# Build for all platforms
make release

# Clean build artifacts
make clean
```

### For Release Managers

1. **Create Release**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Monitor Pipeline**: GitHub Actions will automatically:
   - Run tests
   - Build all platforms
   - Create GitHub release with assets

3. **Verify Release**: Download and test binaries from GitHub release

### For End Users

```bash
# Download appropriate binary for your platform
# Linux x86_64
wget https://github.com/hackafterdark/context-sherpa/releases/download/v1.0.0/context-sherpa-linux-amd64

# macOS Apple Silicon
wget https://github.com/hackafterdark/context-sherpa/releases/download/v1.0.0/context-sherpa-darwin-arm64

# Windows x86_64
wget https://github.com/hackafterdark/context-sherpa/releases/download/v1.0.0/context-sherpa-windows-amd64.exe

# Windows ARM64
wget https://github.com/hackafterdark/context-sherpa/releases/download/v1.0.0/context-sherpa-windows-arm64.exe

# Make executable and run (Linux/macOS)
chmod +x context-sherpa-*
./context-sherpa-*

# Or run directly (Windows)
context-sherpa-windows-amd64.exe
```

## Maintenance and Evolution

### Version Management

- **Semantic Versioning**: `MAJOR.MINOR.PATCH` format
- **Pre-release Tags**: `v1.0.0-alpha.1`, `v1.0.0-beta.1`
- **Build Metadata**: Embedded in binaries for debugging

### Dependency Updates

- **ast-grep**: Monitor for new releases and update URLs
- **Go Version**: Keep aligned with latest stable
- **GitHub Actions**: Update to latest versions regularly

### Platform Support

- **Addition**: New platforms require updating matrix and URLs
- **Deprecation**: Maintain backward compatibility notices
- **Testing**: Add platform-specific tests as needed

## Success Metrics

- **Build Success Rate**: >95% across all platforms
- **Release Automation**: Zero manual intervention required
- **Download Verification**: All binaries checksum-verified
- **User Experience**: Single-command setup for all platforms

## Risk Mitigation

- **Build Failures**: Comprehensive error handling and rollback
- **Network Issues**: Retry logic for ast-grep downloads
- **Platform Incompatibility**: Clear error messages and fallbacks
- **Security**: Checksum verification for all external binaries

This plan provides a robust, scalable foundation for multi-architecture builds while maintaining the project's core value of simplicity and portability.