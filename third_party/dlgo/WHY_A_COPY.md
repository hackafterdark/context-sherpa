# Why a Local Copy of `dlgo`?

The `dlgo` inference engine (found in `third_party/dlgo`) is a local fork of `github.com/computerex/dlgo`. 

We have chosen to vendor and modify it locally rather than importing it as a standard Go module for several critical reasons:

### 1. Compilation Compatibility
The upstream repository contains several benchmark and example files located in its root and subdirectories that use `package main`. Go's module system does not allow importing a package that contains multiple `main` files in this manner, as it leads to compilation errors when used as a library dependency. We have stripped these files out of our local copy.

### 2. MCP Protocol Integrity (Stdout Pollution)
`context-sherpa` operates as a **Model Context Protocol (MCP)** server. MCP relies on a strict JSON-RPC communication stream over `os.Stdout`. 

Upstream `dlgo` performs various `fmt.Printf` and `fmt.Print` calls for progress bars and status updates directly to `stdout`. Any data written to `stdout` that is not valid JSON-RPC will corrupt the communication pipe and causes the connection to **Antigravity** (and other MCP clients) to crash. 

By vendoring the code, we have the ability to:
- Surgically silence internal logging.
- Ensure that the inference process remains "silent" and protocol-safe.

### 3. Stability and "Zero-CGO" Guarantee
This project aims for a zero-dependency, zero-CGO setup for local LLM inference. Having the engine code local ensures that we can maintain this guarantee without risk of upstream updates introducing CGO-based optimizations (like GPU wrappers) that would break our target cross-platform stability.

### 4. Dependency Footprint
We have removed unused GPU-specific benchmark code and unnecessary dependencies (like Vulkan wrappers) that were present in the full repository, keeping our final executable size optimized.

---
*Note: If the upstream repository eventually supports a "library mode" with configurable logging and a clean package structure, we may consider moving back to a standard Go module import.*
