# Plan: Community-Driven ast-grep Rule Marketplace

This document outlines the plan for creating a community-driven, shareable repository of ast-grep rules, integrated with the `context-sherpa` MCP server.

## 1. Core Principles

- **Transparency**: The entire rule set will be hosted in a public GitHub repository, allowing anyone to browse, inspect, and verify the rules.
- **Credential-Free Access**: The system will **not** require any GitHub API keys, personal access tokens, or user authentication. Access will be anonymous and public.
- **Community-Driven**: Contributions will be managed through standard GitHub pull requests, making it easy for anyone to submit new rules or improve existing ones.
- **Interactive & Discoverable**: The MCP server will provide an interactive experience for users to find, understand, and import rules into their local projects.

## 2. Architecture: GitHub Raw Content

To avoid API authentication and rate-limiting, the MCP server will interact with the rule repository by fetching raw file content directly via HTTP requests.

- **Rule Source**: A dedicated public GitHub repository named `github.com/hackafterdark/context-sherpa-community-rules`.
- **Access Method**: Use `raw.githubusercontent.com` URLs to fetch an index file and individual rule YAML files. This is a simple, fast, and anonymous way to read public repository content.

### Repository Structure

To be future-proof and support multiple linter tools, the repository will be organized by tool.

```
/context-sherpa-community-rules/
├── ast-grep/
│   ├── rules/
│   │   └── go/
│   │       └── security/
│   │           └── go-sql-injection.yml
│   └── tests/
│       └── go/
│           └── security/
│               └── go-sql-injection/
│                   ├── valid.go
│                   └── invalid.go
├── semgrep/
│   └── ... (future tool)
├── .github/
│   └── workflows/
│       ├── ast-grep-ci.yml    // Action to validate ast-grep PRs
│       └── update-index.yml   // Action to update the single index
├── index.json
└── README.md
```

- **`index.json`**: A **single, root-level** manifest file that acts as the central catalog for all rules, across all tools and languages. This allows the MCP server to fetch the entire rule set with a single, efficient HTTP request.

### Automated Indexing via GitHub Actions

To ensure a frictionless contribution process, the `index.json` file will be **100% managed by automation**.

### Automated CI and Validation

To guarantee the quality and reliability of all community rules, a robust CI process will be enforced using GitHub Actions on every pull request.

-   **Contributor Workflow**: A contributor must submit their rule (`.yml`) along with corresponding test cases (`valid` and `invalid` files).
-   **PR Validation (`ci.yml`)**:
    1.  **Test Existence**: The CI will check that for every new or modified rule, a corresponding test directory with `valid` and `invalid` files exists.
    2.  **Test Execution**: The CI will run `ast-grep` against the test files:
        -   It **must** find one or more violations in the `invalid` file.
        -   It **must** find zero violations in the `valid` file.
    3.  **Block Merge**: Pull requests cannot be merged until these CI checks pass.

### Automated Indexing (`update-index.yml`)

-   **Trigger**: This action runs only after a PR is successfully merged into the `main` branch.
-   **Process**: It scans the `rules/` directory, regenerates the `index.json` file from the metadata in all the rule YAML files, and commits the updated index.

### Rule Metadata & Index Schema

To support rich search and give credit to contributors, each rule's YAML file will contain metadata that the GitHub Action will aggregate into the `index.json`.

**Example Rule (`go-sql-injection.yml`):**
```yaml
id: go-sql-injection
language: go
author: YourGitHubUsername
message: "..."
severity: error
metadata:
  tags: security, sql, database
  description: "Detects the use of fmt.Sprintf in database calls, which is a common SQL injection vulnerability."
rule:
  ...
```

**Generated `index.json` Entry:**
```json
{
  "id": "ast-grep-go-sql-injection",
  "tool": "ast-grep",
  "path": "ast-grep/rules/go/security/go-sql-injection.yml",
  "language": "go",
  "author": "YourGitHubUsername",
  "tags": ["security", "sql", "database"],
  "description": "Detects the use of fmt.Sprintf in database calls, which is a common SQL injection vulnerability."
}
```

## 3. MCP Server Tools & Workflow

A suite of new tools will be added to the `context-sherpa` server to create an interactive workflow.

### Proposed Tools:

1.  **`search_community_rules`**
    -   **Description**: "Search the community rule repository for ast-grep rules."
    -   **Arguments**:
        -   `query` (string, required): A natural language query (e.g., "sql injection", "check for todos").
        -   `language` (string, optional): Filter by language (e.g., "go", "python").
        -   `tags` (string, optional): Comma-separated list of tags to filter by (e.g., "security,database").
    -   **Logic**: Fetches the single, root `index.json`. It then performs a fast, in-memory search based on the user's query, language, and tag filters.

2.  **`get_community_rule_details`**
    -   **Description**: "Get the full YAML content and explanation for a community rule."
    -   **Arguments**:
        -   `rule_id` (string, required): The unique ID of the rule (e.g., "go-sql-injection").
    -   **Logic**: Finds the rule's path in `index.json` and fetches the raw YAML file from the repository. It can also provide a summary from the index.

3.  **`import_community_rule`**
    -   **Description**: "Download a community rule and add it to the local project."
    -   **Arguments**:
        -   `rule_id` (string, required): The ID of the rule to import.
    -   **Logic**: Fetches the rule's YAML content and uses the existing `add_or_update_rule` logic to save it to the user's local `rules` directory.

## 4. Server-Side Implementation & Caching

To ensure the system is performant and does not overload the language model's context window, the MCP server will handle all data processing.

-   **Context Window Protection**: The entire `index.json` is **never** sent to the agent. The server processes the large index file and returns only a small, relevant list of search results. This keeps the agent's context clean and focused.
-   **In-Memory Caching**: The MCP server is a persistent process. It will implement an in-memory cache for the `index.json` file with a Time-To-Live (TTL) of 5-10 minutes. This means the index will only be downloaded from GitHub periodically, making subsequent searches instantaneous. No files will be written to the user's disk.
-   **Efficient Go Search**: The search logic will be implemented idiomatically in Go:
    1.  The cached JSON will be unmarshalled into a slice of structs (`[]Rule`).
    2.  Standard Go functions will be used to filter this slice by `language` and `tags`.
    3.  A fuzzy-search algorithm will be applied to the `query` parameter against the `id`, `description`, and `tags` fields to provide flexible, natural-language matching.

### 5. User Interaction Flow:

1.  **User**: "Find a rule to prevent SQL injection in Go."
2.  **Agent**: Calls `search_community_rules(query="SQL injection", language="go")`.
3.  **MCP Server**: Fetches `index.json`, finds the `go-sql-injection` rule, and returns its info.
4.  **Agent**: "I found a rule: `go-sql-injection`. It detects `fmt.Sprintf` in database calls. Would you like to see the details or import it?"
5.  **User**: "Import it."
6.  **Agent**: Calls `import_community_rule(rule_id="go-sql-injection")`.
7.  **MCP Server**: Fetches the rule's YAML from `raw.githubusercontent.com` and saves it to the user's local project.
8.  **Agent**: "The rule has been added to your project. You can now use it in scans."

This plan creates a powerful, accessible, and community-focused ecosystem for sharing code analysis rules without adding complexity for the end-user.
