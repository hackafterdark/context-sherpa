# Multi-Architecture Build System for context-sherpa
# Version: 2.0
#
# Note: This Makefile no longer embeds ast-grep binary.
# For development, use 'make download-ast-grep' to get ast-grep binary for testing.
# For production use, install ast-grep via package manager (recommended).

# Variables
BINARY_NAME := context-sherpa
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_DIR := build
AST_GREP_VERSION := 0.39.5
GO_FLAGS := -ldflags "-X main.version=$(VERSION) -s -w"

# Default target
.PHONY: all
all: build-current-arch

# Setup development environment
.PHONY: setup
setup:
	@echo "Setting up development environment..."
	go mod tidy
	go mod download

# Run tests
.PHONY: test
test: setup
	@echo "Running tests..."
	go test ./...

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(BINARY_NAME)*

# Download ast-grep binary for development/testing (linux/amd64)
.PHONY: download-ast-grep
download-ast-grep:
	@echo "Downloading ast-grep binary for development/testing..."
	@curl -L -o ast-grep.zip https://github.com/ast-grep/ast-grep/releases/download/$(AST_GREP_VERSION)/app-x86_64-unknown-linux-gnu.zip
	@unzip -o ast-grep.zip
	# Verify ast-grep binary works
	@if [ -f "ast-grep" ]; then \
		./ast-grep --version >/dev/null && echo "✅ ast-grep binary downloaded and verified" || (echo "❌ ast-grep binary not working"; exit 1); \
	fi
	@rm ast-grep.zip
	@echo "ast-grep binary downloaded to project root for development use"

# Build for current architecture (linux/amd64)
.PHONY: build-current-arch
build-current-arch: test
	@echo "Building for current architecture (linux/amd64)..."
	@mkdir -p $(BUILD_DIR)
	go build $(GO_FLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 ./cmd/context-sherpa
	@echo "Binary built: $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64"

# Create release package
.PHONY: release
release: test clean build-current-arch
	@echo "Creating release package..."
	@mkdir -p $(BUILD_DIR)/release
	@cd $(BUILD_DIR) && cp $(BINARY_NAME)-linux-amd64 $(BINARY_NAME) && tar -czf release/$(BINARY_NAME)-$(VERSION)-linux-amd64.tar.gz $(BINARY_NAME)*
	@cd $(BUILD_DIR) && sha256sum $(BINARY_NAME)* > release/checksums.txt
	@echo "Release package created: $(BUILD_DIR)/release/$(BINARY_NAME)-$(VERSION)-linux-amd64.tar.gz"
	@echo "Checksums: $(BUILD_DIR)/release/checksums.txt"

# Development workflow (setup + test + build)
.PHONY: dev
dev: setup test build-current-arch

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup              - Set up development environment"
	@echo "  test               - Run tests"
	@echo "  build-current-arch - Build for current platform (linux/amd64)"
	@echo "  release            - Create release package"
	@echo "  clean              - Clean build artifacts"
	@echo "  download-ast-grep  - Download ast-grep binary for development/testing"
	@echo "  dev                - Development workflow (setup + test + build)"
	@echo "  help               - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make dev                    # Full development cycle"
	@echo "  make build-current-arch     # Quick local build"
	@echo "  make download-ast-grep      # Get ast-grep for development"
	@echo "  make release                # Create release package"
	@echo ""
	@echo "Note: ast-grep binary is no longer embedded in the context-sherpa binary."
	@echo "For production use, install ast-grep via package manager (recommended)."