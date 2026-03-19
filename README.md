# ![Context Sherpa](./docs/appicon.png) Context Sherpa

**Context Sherpa** is a specialized platform for **Context Engineering**. It bridges the gap between developers and AI coding agents by providing LLMs with the precise, high-fidelity signals they need to operate with expert-level accuracy while dramatically reducing token consumption.

![Code View](./docs/code-atlas-2.png)

## Why Context Engineering?

Agentic coding tools often struggle with either "hallucinating" codebase relationships or overwhelming their context windows with irrelevant files. Context Sherpa solves this through two primary goals:

1.  **Increase Accuracy**: Provide AI agents with strong, symbolic signals (definitions, references, and impact analysis) to ensure code changes are correct and idiomatic.
2.  **Optimize Context**: Using SCIP-based indexing and structural analysis, Context Sherpa allows agents to pinpoint exactly what they need, often reducing the tokens required for a task by up to 90% compared to raw file-searching.

### An Optimized Alternative

Traditional tools like **grep** are fast but lack symbol awareness, while **semantic search** often returns noisy, over-processed results. Context Sherpa's tool suite is designed as a high-fidelity alternative:
- **vs. Grep**: SCIP indexing provides precise symbol-resolution (definitions vs. references), eliminating the guesswork of text-based search.
- **vs. Semantic Search**: Where vectors often fail to distinguish between similar-looking code, our **Structural Analysis** (ast-grep) uses the code's abstract syntax tree for exact pattern matching—**no vector databases or text embeddings required**.
- **vs. Indexing**: Unlike massive, centralized indexing services, Context Sherpa is local-first, lightweight, and requires zero cloud configuration or complex RAG infrastructure.

Of course the benefit of text embeddings in a vector database and other traditional methods is that they are more of a one size fits all solution. They can be used for many programming languages and even non-code files. Remember that a different SCIP tool must be used for each programming language. Lucky for you, you can use both. Context Sherpa will add SCIP tools for more languages in the future. 

![Context Sherpa Scan Results](./docs/example-scan-output.png)

---

## Core Capabilities

- **Code Atlas Explorer**: A premium GUI to visualize and inspect your codebase relationships.
- **Agent Rule Management**: Dynamically manage project-specific standards using `ast-grep` and natural language feedback.
- **Integrated Local Reasoning**: Connect to local providers (Ollama, LM Studio) to enable **Tiered Inference**. This architecture utilizes local models for high-frequency context distillation and semantic triage, drastically reducing the token noise and costs associated with sending raw source code to frontier models.
- **Universal MCP Server**: A high-performance "headless" mode that integrates directly with tools like Cursor, Cline, and Roo Code.

---

## 🎮 GUI vs. 🖥️ Headless Mode

Context Sherpa is a single executable that adapts to your needs:

| Feature | GUI Mode | Headless (MCP) Mode |
| :--- | :--- | :--- |
| **Code Atlas Explorer** | ✅ Yes | ❌ No |
| **Rule Visualizer/Editor** | ✅ Yes | ❌ API only |
| **Dependency Manager** | ✅ Yes | ❌ Manual |
| **MCP Tool Access** | ✅ Directly via Agent | ✅ Directly via Agent |
| **Resource Usage** | Standard App | Ultra Lightweight |

---

## ⚡ Quick Start

### Installation

Download the latest version for your platform from [GitHub Releases](https://github.com/hackafterdark/context-sherpa/releases).

- **Windows**: Download the `.exe` and run it.
- **macOS**: Download the app, move to `/Applications`, and clear quarantine if necessary:
  ```bash
  xattr -d com.apple.quarantine /Applications/Context-Sherpa.app
  ```
- **Linux**: Download the binary and provide execution permissions (`chmod +x`).

> [!IMPORTANT]
> **Security Notice**: Pre-built binaries are currently **not signed**. When running the application for the first time, you may encounter OS security warnings (e.g., Windows SmartScreen or macOS Gatekeeper). You will need to manually allow the application to run.
> 
> This is a known limitation of the current release process. Future pre-built binaries will undergo formal code signing for convenience. Building from source (instructions below) avoids these security warnings.

### Building from Source

Context Sherpa is built with **Go** and **Wails**. For detailed building instructions, please see [BUILDING.md](docs/BUILDING.md).

1.  **Build the Frontend**:
    ```bash
    cd frontend && npm install && npm run build && cd ..
    ```
2.  **Build the GUI**:
    ```bash
    # Install Wails CLI if not already present
    go install github.com/wailsapp/wails/v2/cmd/wails@latest

    # Build the binary
    wails build
    ```

---

## 🛠️ MCP Toolchain

AI agents connected to Context Sherpa gain access to a powerful set of tools for structural, symbolic, and semantic analysis.

For a full list of available tools and their parameters, see [TOOLS.md](docs/TOOLS.md).

### Configuration Example
Add Context Sherpa to your AI agent's `mcp_settings.json`:

```json
{
  "mcpServers": {
    "context-sherpa": {
      "command": "context-sherpa",
      "args": ["--projectRoot", "/path/to/your/project"]
    }
  }
}
```

---

## 🏔️ The Tiered Inference Framework

Context Sherpa is built on a three-layer "Tiered Inference" strategy that balances speed, cost, and intelligence:

- **Tier 1: Deterministic Logic (SCIP / ast-grep)**: Instant, 0-token cost structural mapping. Precise symbolic analysis that provides the ground truth for your codebase.
- **Tier 2: Integrated Local Reasoning (Ollama / LM Studio)**: local distillation and semantic triage on your own hardware. This layer offsets costs and drastically reduces the token burden on Tier 3 frontier models.
- **Tier 3: Strategic Frontier Intelligence (Claude / Gemini)**: High-level architectural planning and complex refactoring directed by the "Big Brain" (your primary agent).

By offloading Tier 1 and Tier 2 tasks to Context Sherpa, you save up to 90% in token costs while significantly increasing the accuracy of Tier 3 strategic work.

---

## 📦 Setting Up Dependencies

Context Sherpa uses 3rd party tools to provide high-fidelity intelligence. These can be managed directly in the **GUI Settings** area:

1.  **ast-grep**: The core structural analysis engine.
2.  **SCIP Indexers**: Language-specific indexers for Go, TypeScript, and Python.
3.  **Inference Engines**: Connect to **Ollama** or **LM Studio** to enable semantic reasoning tools.

---

## 📚 Documentation

- [GUI Guide](docs/GUI_GUIDE.md) - A visual tour of Context Sherpa's premium interface.
- [MCP Tools Reference](docs/TOOLS.md) - Detailed documentation for all 20+ MCP tools.
- [Building Context Sherpa](docs/BUILDING.md) - How to compile the GUI and Headless versions.
- [Contributing Guide](CONTRIBUTING.md) - How to help improve Context Sherpa.

---

## Acknowledgments

Context Sherpa leverages several incredible open-source projects:

- **[ast-grep](https://ast-grep.github.io/)**: The core engine for high-performance structural code analysis.
- **[Wails](https://wails.io/)**: The framework powering our cross-platform desktop experience.
- **[Sourcegraph SCIP](https://github.com/sourcegraph/scip)**: Providing the foundation for precise symbolic code intelligence via `scip-go`, `scip-typescript`, and `scip-python`.
- **Local Inference**: We are grateful to the communities behind **Ollama** and **LM Studio** for providing the infrastructure that powers our tiered semantic inference.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.