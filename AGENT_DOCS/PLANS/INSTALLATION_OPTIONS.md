# Installation Strategy & Options

## Overview

This document outlines the multi-tiered installation strategy for Context Sherpa, designed to provide the best possible developer experience across different platforms while respecting OS security models and user preferences.

## 🎯 **Core Philosophy**

**"Meet users where they are, with the least friction possible."**

- **Primary Goal:** Zero-configuration installation for most users
- **Secondary Goal:** Clear, actionable guidance for edge cases
- **Constraint:** Respect OS security models (no bypasses, only proper workarounds)
- **Target:** Technical audience who values transparency and control

## 📋 **Installation Tiers**

### **🥇 Tier 1: Go Install (Recommended)**

**Best For:** Developers with Go toolchain installed

**Method:**
```bash
# Install Context Sherpa
go install github.com/hackafterdark/context-sherpa/cmd/context-sherpa@latest

# Install ast-grep (choose one method)
# Option 1: Homebrew (macOS/Linux)
brew install ast-grep

# Option 2: Cargo (Rust)
cargo install ast-grep --locked

# Option 3: NPM
npm i @ast-grep/cli -g

# Option 4: Other methods (see https://ast-grep.github.io/guide/quick-start.html)
# pip install ast-grep-cli
# sudo port install ast-grep (MacPorts)

# Use with project root
context-sherpa --projectRoot="/path/to/your/project"
```

**Why This Is Best:**
- ✅ **Zero Security Issues** - Context Sherpa built locally, automatically trusted
- ✅ **Zero Configuration** - Both tools install to PATH, work immediately
- ✅ **Standard Practice** - Idiomatic for developers
- ✅ **Version Management** - Easy updates with package managers
- ✅ **Dependency Clarity** - ast-grep treated as explicit peer dependency
- ✅ **Cross-Platform** - Works identically on Linux, macOS, Windows

**README Documentation:**
```markdown
## Installation

### Recommended: Go Install
If you have the Go toolchain installed, this is the simplest and most secure method:

```bash
# Install Context Sherpa
go install github.com/hackafterdark/context-sherpa/cmd/context-sherpa@latest

# Install ast-grep (choose one method)
# Option 1: Homebrew (macOS/Linux)
brew install ast-grep

# Option 2: Cargo (Rust)
cargo install ast-grep --locked

# Option 3: NPM
npm i @ast-grep/cli -g

# Option 4: Other methods (see https://ast-grep.github.io/guide/quick-start.html)
# pip install ast-grep-cli
# sudo port install ast-grep (MacPorts)

# Verify installation
context-sherpa --version
ast-grep --version  # or 'sg --version' on Linux
```

**For Windows Users:** Install ast-grep using one of the methods above, or download from [GitHub releases](https://github.com/ast-grep/ast-grep/releases/latest).
```

---

### **🥈 Tier 2: Multi-Ecosystem Package Managers**

**Best For:** Users who prefer their platform's native package managers

#### **Option A: Homebrew (macOS/Linux)**
```bash
# Future implementation
brew install hackafterdark/tap/context-sherpa
```

**Implementation:**
- **Repository:** `github.com/hackafterdark/homebrew-tap`
- **Formula:** Specify `ast-grep` as dependency
- **Automation:** GitHub Actions for updates

#### **Option B: MacPorts (macOS)**
```bash
# Future implementation
sudo port install context-sherpa
```

**Implementation:**
- **Port File:** Create `devel/context-sherpa/Portfile`
- **Dependencies:** Specify `ast-grep` requirement
- **Updates:** Automated via GitHub releases

#### **Option C: NPM (Cross-Platform)**
```bash
# Future implementation
npm install -g @hackafterdark/context-sherpa
```

**Implementation:**
- **Package:** `@hackafterdark/context-sherpa`
- **Distribution:** Pre-built binaries for each platform
- **Dependencies:** Include `ast-grep` binaries in package

#### **Option D: Pip (Cross-Platform)**
```bash
# Future implementation
pip install context-sherpa
```

**Implementation:**
- **Package:** `context-sherpa` on PyPI
- **Distribution:** Platform-specific wheels with native binaries
- **Dependencies:** Bundle `ast-grep` for each platform

**Why These Are Strategic:**
- ✅ **Ecosystem Adoption** - Meet users in their preferred tooling
- ✅ **Automatic Dependencies** - Handle ast-grep installation per ecosystem
- ✅ **Professional Distribution** - Standard software distribution methods
- ✅ **Cross-Platform Reach** - Cover all major platforms and ecosystems

**README Documentation:**
```markdown
### For macOS & Linux: Homebrew
```bash
# Install with all dependencies
brew install hackafterdark/tap/context-sherpa
```

**Note:** This requires creating and maintaining a Homebrew tap. See implementation details below.
```

---

### **🥉 Tier 3: Manual Installation (Flexible)**

**Best For:** Users who prefer direct control or have specific requirements

**Method:**
```bash
# Download Context Sherpa
curl -L -o context-sherpa https://github.com/hackafterdark/context-sherpa/releases/latest/download/context-sherpa-darwin-amd64

# Download ast-grep
curl -L -o ast-grep https://github.com/ast-grep/ast-grep/releases/latest/download/ast-grep-x86_64-apple-darwin.tar.gz

# Install (choose any directory)
mkdir -p ~/bin
mv context-sherpa ~/bin/
tar -xzf ast-grep-x86_64-apple-darwin.tar.gz -C ~/bin/
export PATH=$HOME/bin:$PATH

# Or place in project directory
mv context-sherpa /path/to/your/project/
mv ast-grep /path/to/your/project/
```

**Why This Works:**
- ✅ **Complete Transparency** - Users see exactly what they're getting
- ✅ **Smart Detection** - Works regardless of where you place the binaries
- ✅ **No Security Issues** - Standard external tool usage
- ✅ **Flexible Placement** - Same directory, project directory, or system PATH
- ✅ **Technical Audience Friendly** - Appeals to users who want control

**README Documentation:**
```markdown
### Manual Installation
Download pre-compiled binaries from [GitHub Releases](https://github.com/hackafterdark/context-sherpa/releases/latest).

1. **Download Context Sherpa** for your platform
2. **Download ast-grep** from [ast-grep releases](https://github.com/ast-grep/ast-grep/releases/latest)
3. **Place both in same directory** (preferably in your PATH)

**macOS Security:** Remove quarantine attribute:
```bash
xattr -d com.apple.quarantine ./context-sherpa-darwin-amd64
```

**Windows Security:** Right-click files → Properties → Security → Unblock
```

## 🔧 **Technical Architecture**

### **Binary Strategy: External Dependency with Smart Detection**

**Decision:** ✅ **EXTERNAL DEPENDENCY MODEL** - ast-grep as explicit peer dependency

**Architecture:**
- **Simple Detection:** Two-location fallback strategy
- **MCP-Aware:** Respects stdio communication limitations
- **User Control:** Optional `--astGrepPath` flag for explicit specification

**Current Code Analysis:**
- `cmd/context-sherpa/main.go` (formerly `cmd/server/main.go`)
- `Start()` function receives `sgBinary []byte` parameter (line 79)
- `extractSgBinary()` handles embedded binary extraction (lines 640-676)
- **Windows:** Uses system `ast-grep.exe` instead of embedded (lines 642-658)
- **Non-Windows:** Extracts embedded binary to temp file (lines 661-676)

**Required Changes:**
1. **Remove embedded binary** (`cmd/context-sherpa/main.go` lines 10-11, 23-25)
2. **Add `--astGrepPath` flag** (`cmd/context-sherpa/main.go` lines 14-19)
3. **Update `Start()` signature** (`internal/mcp/server.go` line 79)
4. **Replace `extractSgBinary()`** with `findAstGrepBinary()` (lines 640-676)
5. **Remove `sgBinaryData` variable** (line 22)
6. **Update all callers** to use new function signatures

**Implementation:**
```go
func findAstGrepBinary(astGrepPath string) (string, error) {
    // 1. User explicitly specified path (highest priority)
    if astGrepPath != "" {
        if _, err := os.Stat(astGrepPath); err == nil {
            verboseLog("Using user-specified ast-grep path: %s", astGrepPath)
            return astGrepPath, nil
        }
        return "", fmt.Errorf("ast-grep not found at specified path: %s", astGrepPath)
    }

    // 2. System PATH (standard location)
    if path, err := exec.LookPath("ast-grep"); err == nil {
        verboseLog("Found ast-grep in PATH: %s", path)
        return path, nil
    }

    if runtime.GOOS == "windows" {
        if path, err := exec.LookPath("ast-grep.exe"); err == nil {
            verboseLog("Found ast-grep.exe in PATH: %s", path)
            return path, nil
        }
    }

    // 3. Clear error - explain MCP server limitation
    return "", fmt.Errorf(`ast-grep not found in PATH.

As an MCP server communicating via stdio, I cannot:
- Detect where your editor/IDE is running from
- Access your current working directory
- Find binaries in project-specific locations

Please ensure ast-grep is available in one of these ways:

1. Install in system PATH (see https://ast-grep.github.io/guide/quick-start.html):
   # Choose one of these installation methods:
   brew install ast-grep                    # macOS/Linux
   cargo install ast-grep --locked         # Rust
   npm i @ast-grep/cli -g                  # Node.js
   pip install ast-grep-cli                # Python
   sudo port install ast-grep              # MacPorts

2. Specify explicit path:
   context-sherpa --astGrepPath="/path/to/ast-grep"`)
}
```

**Search Strategy:**
1. **Explicit Path** (`--astGrepPath` flag) - User-specified location
2. **System PATH** - Standard installation location
3. **Clear Error** - Explains MCP limitations and provides solutions

**Benefits:**
- ✅ **Maximum Simplicity** - Only two possible locations to check
- ✅ **MCP-Compatible** - Works within stdio communication constraints
- ✅ **User Control** - Explicit path specification when needed
- ✅ **Clear Limitations** - Honest about MCP server constraints
- ✅ **Enhanced Logging** - Shows which detection method succeeded

### **Implementation Steps**

**Phase 1: Core Infrastructure Changes**
1. **Remove Binary Embedding** (`cmd/context-sherpa/main.go`)
   - Remove `//go:embed bin/ast-grep` directive (line 10)
   - Remove `astGrepBinary []byte` variable (line 11)
   - Remove `GetAstGrepBinary()` test function (lines 23-25)

2. **Add Command-Line Flag** (`cmd/context-sherpa/main.go`)
   - Add `astGrepPath := flag.String("astGrepPath", "", "Explicit path to ast-grep binary")`
   - Pass `*astGrepPath` to `mcp.Start()`

3. **Update Start Function** (`internal/mcp/server.go`)
   - Change signature: `Start(sgBinary []byte, projectRoot string, verbose bool, logFilePath string, astGrepPath string)`
   - Remove `sgBinaryData` global variable (line 22)
   - Replace `extractSgBinary()` calls with `findAstGrepBinary(astGrepPath)`

**Phase 2: Binary Detection Implementation**
1. **Replace `extractSgBinary()` Function** (lines 640-676)
   ```go
   func findAstGrepBinary(astGrepPath string) (string, error) {
       // 1. Explicit user path
       if astGrepPath != "" {
           if _, err := os.Stat(astGrepPath); err == nil {
               return astGrepPath, nil
           }
           return "", fmt.Errorf("ast-grep not found at: %s", astGrepPath)
       }

       // 2. System PATH
       if path, err := exec.LookPath("ast-grep"); err == nil {
           return path, nil
       }

       // 3. Clear error with installation guidance
       return "", fmt.Errorf(`ast-grep not found in PATH.
       Install using: https://ast-grep.github.io/guide/quick-start.html`)
   }
   ```

**Phase 3: Update All Callers**
1. **Update `scanCodeHandler`** (line 290) - Remove `extractSgBinary()` call
2. **Update `scanPathHandler`** (line 353) - Remove `extractSgBinary()` call
3. **Update tool registration** - Pass `astGrepPath` to handlers
4. **Update test files** - Remove embedded binary test dependencies

**Phase 4: Enhanced Error Messages**
1. **Replace generic errors** with specific installation guidance
2. **Add platform-specific instructions** for Windows users
3. **Include official documentation links** to ast-grep quickstart

### **Homebrew Tap Setup**

**Repository Structure:**
```
hackafterdark/homebrew-tap/
├── Formula/
│   └── context-sherpa.rb
└── README.md
```

**Formula Content (`context-sherpa.rb`):**
```ruby
class ContextSherpa < Formula
  desc "AI-powered code analysis server for linting and dynamic rule management"
  homepage "https://github.com/hackafterdark/context-sherpa"
  url "https://github.com/hackafterdark/context-sherpa/releases/latest/download/context-sherpa-darwin-amd64.tar.gz"
  sha256 "DOWNLOAD_CHECKSUM"
  license "MIT"

  depends_on "ast-grep"

  def install
    bin.install "context-sherpa-darwin-amd64" => "context-sherpa"
  end

  test do
    assert_match "Context Sherpa", shell_output("#{bin}/context-sherpa --version")
  end
end
```

### **GitHub Actions Automation**

**Release Workflow:**
```yaml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Create Release
        uses: actions/create-release@v1
        # ... release creation logic

  homebrew-tap-update:
    runs-on: ubuntu-latest
    steps:
      - name: Update Homebrew Formula
        uses: some-action/to-update-homebrew-tap
        # ... update formula with new version/checksum
```

## 📊 **User Journey Mapping**

### **Developer with Go Installed:**
```
User sees README → "go install" is first option →
Runs: go install github.com/hackafterdark/context-sherpa/cmd/context-sherpa@latest →
Binary installed to GOPATH/bin, automatically trusted →
Ready to use ✅
```

### **macOS User without Go:**
```
User sees README → Homebrew option (when available) →
Runs: brew install hackafterdark/tap/context-sherpa →
Dependencies installed automatically →
Ready to use ✅
```

### **User Needing Manual Control:**
```
User sees README → Manual installation section →
Downloads specific versions →
Follows platform-specific instructions →
Ready to use ✅
```

## 🎯 **Success Metrics**

### **Adoption Goals:**
- **70% of users** use `go install` method
- **20% of users** use Homebrew (once available)
- **10% of users** use manual installation

### **Experience Goals:**
- **Zero users** encounter security warnings (except manual installations)
- **< 5 minutes** average installation time
- **> 95% success rate** for first-time installations

## 🚀 **Development Roadmap**

### **Phase 1: Current State (✅ Completed)**
- ✅ `go install` method documented with correct paths
- ✅ Manual installation with `xattr` command documented
- ✅ Project root override implemented and tested
- ✅ External dependency model with smart detection implemented

### **Phase 2: Multi-Ecosystem Distribution (⏳ Pending)**
- ⏳ **Homebrew Tap** - macOS/Linux package manager
- ⏳ **MacPorts Port** - Alternative macOS package manager
- ⏳ **NPM Package** - `@hackafterdark/context-sherpa` with native binaries
- ⏳ **Pip Package** - `context-sherpa` with platform-specific wheels
- ⏳ Implement automated updates for all package managers

### **Phase 3: Windows Ecosystem (⏳ Future)**
- ⏳ **Scoop/Chocolatey** packages for Windows
- ⏳ **Windows Store** packaging for broader reach
- ⏳ Winget integration for modern Windows

### **Phase 4: Enhanced Experience (⏳ Future)**
- ⏳ One-click installers for GUI users
- ⏳ Docker container for isolated environments
- ⏳ IDE plugins for direct integration

## 🧪 **Implementation Testing Strategy**

### **Pre-Implementation Validation**
1. **Binary Size Impact:** Measure reduction after removing embedded binary (~50-80% expected)
2. **Build Time Improvement:** Verify faster compilation without embedding step
3. **Memory Usage:** Confirm reduced memory footprint during execution

### **Post-Implementation Testing**
1. **Core Functionality Tests:**
   ```bash
   # Test 1: Standard Go install
   go install github.com/hackafterdark/context-sherpa/cmd/context-sherpa@latest
   go install github.com/ast-grep/ast-grep/cmd/sg@latest
   context-sherpa --projectRoot="/test/project"

   # Test 2: Explicit path specification
   ./context-sherpa --astGrepPath="/custom/path/ast-grep" --projectRoot="/test/project"

   # Test 3: Missing dependency scenario
   # Should provide clear error with installation instructions
   ```

2. **Error Handling Tests:**
   - **PATH Detection:** Verify correct binary found in standard locations
   - **Explicit Path:** Test user-specified path resolution
   - **Missing Binary:** Validate helpful error messages with installation links
   - **Platform Differences:** Test Windows `.exe` handling vs Unix binary names

3. **Cross-Platform Tests:**
   - **Linux:** Standard PATH installation and manual placement
   - **macOS:** Homebrew installation and quarantine handling
   - **Windows:** Same directory placement and explicit path usage

### **Expected Outcomes**
- ✅ **50-80% binary size reduction** (removing embedded ast-grep)
- ✅ **Faster builds** (no binary embedding step)
- ✅ **Cleaner error messages** with actionable guidance
- ✅ **Zero security warnings** across all platforms
- ✅ **Simplified CI/CD** pipeline (no binary management)

## 💡 **Key Insights**

1. **External Dependency Model is Superior** - Ditching `go embed` eliminates all security issues and complexity
2. **Two-Location Strategy is Optimal** - `--astGrepPath` flag + system PATH covers all realistic use cases
3. **MCP Server Constraints are Real** - Must work within stdio communication limitations
4. **Cmd/ Structure is Correct** - `cmd/context-sherpa` follows Go standards and supports ecosystem growth
5. **Multi-Ecosystem Distribution is Strategic** - NPM, Pip, MacPorts extend reach beyond Go community
6. **Go Install is Perfect** - For technical audience, this is the cleanest possible experience
7. **Package Managers Build Trust** - Users prefer familiar installation methods
8. **Transparency Builds Confidence** - Clear requirements and honest limitations build user trust

This strategy provides **maximum flexibility** while ensuring **minimum friction** for the majority of users.