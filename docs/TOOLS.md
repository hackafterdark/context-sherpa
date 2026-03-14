# Context Sherpa MCP Tools

Context Sherpa exposes a suite of tools through the Model Context Protocol (MCP). These tools allow AI agents to interact with your codebase, perform analysis, and manage coding rules.

## Tool Categories

- [Semantic Reasoning](#semantic-reasoning)
- [Structural Analysis (ast-grep)](#structural-analysis-ast-grep)
- [Symbolic Code Intelligence (SCIP)](#symbolic-code-intelligence-scip)

---

## Semantic Reasoning

These tools leverage **Integrated Local Reasoning** via your preferred model provider (**Ollama** or **LM Studio**) to enable Tier 2 of the [Tiered Inference Framework](../README.md#tiered-inference-framework).

### `classify_repo_intent`
Determines which Sherpa tool (Symbolic, Structural, or Semantic) is best suited for a user's high-level query.
- **Parameters**:
  - `query` (string, required): The user's natural language query.

### `summarize_code_intent`
Distills raw code into a concise, 3-sentence functional summary covering Inputs, Outputs, and Side-effects.
- **Parameters**:
  - `code` (string, required): The source code to summarize.

### `query_local_reasoning`
A catch-all tool for asking open-ended semantic questions about code that don't fit other tools.
- **Parameters**:
  - `prompt` (string, required): The question for the local SLM.
  - `modelId` (string, optional): The specific model to use.
  - `max_tokens` (number, optional): Max tokens to generate.

### `list_local_models`
Lists all available models from the configured local inference engine (Ollama or LM Studio).

### `switch_local_model`
Changes the active default model in the Hub settings.
- **Parameters**:
  - `modelId` (string, required): The ID of the model to set as default.

### `pull_inference_model`
Requests the local inference engine to download a new model.
- **Parameters**:
  - `modelId` (string, required): The ID of the model to pull (e.g. `qwen2.5:0.5b`).


---

## Structural Analysis (ast-grep)

These tools use [ast-grep](https://ast-grep.github.io/) to perform precise, pattern-based scanning and rule management.

### `scan_code`
Scans a code snippet against the project's configured `ast-grep` rules.
- **Parameters**:
  - `code` (string, required): Source code to scan.
  - `language` (string, required): Programming language (e.g., `go`, `python`, `typescript`).
  - `sgconfig` (string, optional): Path to a specific `sgconfig.yml`.

### `scan_path`
Scans a file, directory, or glob pattern for rule violations.
- **Parameters**:
  - `path` (string, required): Path or glob (e.g., `src/**/*.go`).
  - `language` (string, optional): Language filter.

### `add_or_update_rule`
Adds a new rule or updates an existing one in the project.
- **Parameters**:
  - `rule_id` (string, required): Unique ID for the rule.
  - `rule_yaml` (string, required): The full `ast-grep` YAML definition.

### `remove_rule`
Deletes a rule from the project.
- **Parameters**:
  - `rule_id` (string, required): The ID of the rule to remove.

### `initialize_ast_grep`
Sets up the workspace for `ast-grep` by creating `sgconfig.yml` and a `rules/` directory.

### `search_community_rules`
Searches the [Context Sherpa Community Rules](https://github.com/hackafterdark/context-sherpa-community-rules) for pre-built rules.

### `import_community_rule`
Downloads and imports a community rule into your local project.
- **Parameters**:
  - `rule_id` (string, required): ID of the community rule to import.

---

## Symbolic Code Intelligence (SCIP)

These tools provide deep knowledge of symbol relationships (definitions and references) using SCIP.

### `initialize_scip`
Indexes the workspace for the specified language.
- **Parameters**:
  - `language` (string, optional): `go`, `typescript`, or `python`. Auto-detects if omitted.

### `get_symbol_map`
Returns a map of all definitions and references for a given symbol.
- **Parameters**:
  - `symbolName` (string, required): Name of the symbol (e.g., `AuthService`).

### `list_symbols_in_file`
Lists all classes, functions, and variables defined in a specific file.
- **Parameters**:
  - `file_path` (string, required): Path to the file.

### `search_definitions`
Searches for symbol definitions project-wide matching a query.
- **Parameters**:
  - `query` (string, required): The symbol name or pattern.

### `analyze_impact_triage`
Identifies which call sites are most likely to be affected by a change based on SCIP reference data.
- **Parameters**:
  - `references` (string, required): List of SCIP references to triage.
