# Plan for Semantic Analysis Integration in MCP Server

## 1. Overview

This document outlines a phased approach to integrate semantic analysis into the `ast-grep-linter-mcp-server`. The goal is to evolve beyond purely syntactic pattern matching (`ast-grep`) and enable rules that can leverage type information and symbol resolution. This will dramatically increase the accuracy of our linting, reduce false positives, and allow for a much more powerful class of rules.

We will begin with Go, using the official Go language server (`gopls`), and then establish a framework for incorporating other languages over time.

---

## 2. Core Concept: The Hybrid Analysis Model

The server will operate in a hybrid mode, combining the speed of `ast-grep` for syntactic analysis with the precision of a language server for semantic validation.

The workflow for a semantic rule will be:
1.  **Syntactic Pre-filtering (`ast-grep`):** A broad `ast-grep` rule finds all potential candidates for a violation. For example, it finds all standalone function calls.
2.  **Semantic Verification (Language Server):** For each candidate found by `ast-grep`, the MCP server queries a language server (`gopls`) to get semantic information (e.g., "What are the return types of this function?").
3.  **Final Decision:** The server combines the syntactic and semantic information to make a final, accurate decision on whether to report a violation.

---

## 3. Phase 1: Go Language Support via `gopls`

### 3.1. Architecture

The MCP server will manage a long-running `gopls` process. It will communicate with `gopls` using the Language Server Protocol (LSP) over stdio.

### 3.2. New MCP Tools for Semantic Queries

We will introduce new tools to the MCP server that abstract the LSP communication.

#### Tool 1: `get_definition`
-   **Purpose:** Finds the definition of a symbol at a given position.
-   **Arguments:**
    -   `file_path`: `string`
    -   `line`: `int`
    -   `column`: `int`
-   **Process:**
    1.  Receives the request.
    2.  Formats and sends a `textDocument/definition` request to the `gopls` process.
    3.  Parses the `gopls` response, which contains the location (file and position) of the definition.
    4.  Reads the source code at the definition's location to extract the full function signature or type definition.
-   **Returns:** A JSON object with the definition's `file_path`, `start_line`, `end_line`, and the full `signature` text.

#### Tool 2: `get_type_info`
-   **Purpose:** Gets type information for the symbol at a given position.
-   **Arguments:**
    -   `file_path`: `string`
    -   `line`: `int`
    -   `column`: `int`
-   **Process:**
    1.  Receives the request.
    2.  Formats and sends a `textDocument/hover` request to `gopls`.
    3.  Parses the `gopls` response to extract the type information string.
-   **Returns:** A JSON object containing the `type` as a string.

### 3.3. Enhanced `scan_code` Tool

The `scan_code` tool will be upgraded to support a new type of rule.

**New Rule Property: `semantic_check`**

Rules in `sgconfig.yml` can have an optional `semantic_check` property.

```yaml
id: go-unchecked-error-semantic
language: go
rule:
  # 1. Syntactic pre-filter: find all standalone function calls
  pattern: $A($$$)
  inside:
    kind: expression_statement
message: "The error returned by '$A' is not checked."
severity: "error"
# 2. Semantic verification step
semantic_check:
  # The MCP server will execute this check
  - type: 'function_returns_error'
    # It will pass the matched node's info to the check
    input: '$A'
```

**New `scan_code` Workflow:**
1.  Run `ast-grep` as usual.
2.  For each finding from a rule that has a `semantic_check`:
    a. Extract the AST node specified by `input` (e.g., `$A`).
    b. Get its position (file, line, column).
    c. Call the appropriate new MCP tool (`get_definition` or `get_type_info`).
    d. Analyze the returned signature/type to see if it matches the check (e.g., does it include `error` as a return type?).
    e. If the semantic check passes, the finding is confirmed and reported. If it fails, the finding is discarded as a false positive.

---

## 4. Phase 2: Framework for Multi-Language Support

### 4.1. Language Server Management

The MCP server will be updated to manage a pool of language server processes.
-   A configuration file (`language_servers.json`) will map language IDs (e.g., "go", "typescript", "python") to the command needed to start their respective language servers (e.g., `gopls`, `tsserver`, `pylance`).
-   The server will start and manage these processes on demand.

### 4.2. Generic Semantic Checks

The `semantic_check` types will be kept as generic as possible to be reusable across languages.
-   `function_returns_type`: Checks if a function returns a specific type name.
-   `variable_is_type`: Checks if a variable is of a certain type.
-   `is_deprecated`: Checks for deprecation annotations.

This creates a powerful, extensible system where adding support for a new language primarily involves adding its language server to the configuration and ensuring its output can be parsed.

---

## 5. Implementation Steps

1.  **[Go]** Implement the `gopls` process management within the MCP server.
2.  **[Go]** Implement the `get_definition` tool by creating an LSP client that can send `textDocument/definition` requests.
3.  **[Go]** Update the `scan_code` handler to perform the hybrid analysis workflow described in section 3.3.
4.  **[Go]** Test the new `go-unchecked-error-semantic` rule.
5.  **[Framework]** Refactor the language server management to support multiple languages via a configuration file.
6.  **[Framework]** Generalize the semantic check logic.
7.  **[TypeScript]** Add `tsserver` to the configuration and implement a semantic rule for TypeScript as a proof of concept.