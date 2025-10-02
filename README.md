# Context Sherpa

Context Sherpa is an AI-powered code analysis server that helps developers guide AI coding agents. It provides tools for linting, validating, and dynamically managing code rules based on natural language feedback. The server is designed to be a portable, cross-platform binary with no external runtime dependencies, making it easy to set up and use.

## What is Context Sherpa?

As a developer using an AI coding agent, you want to:

- Have your agent automatically validate the code it generates against your project's specific coding patterns.
- Be able to provide natural language feedback (e.g., "From now on, all async functions must have a try/catch block") to your agent.
- Have the agent intelligently convert your feedback into a permanent, machine-readable linting rule using `ast-grep`.
- Be able to easily remove rules that are no longer needed.
- Ensure this system is self-contained in a single executable that you can easily run without managing servers, dependencies, or security risks.

## Installation

1.  Navigate to the [releases page](https://github.com/hackafterdark/context-sherpa/releases/latest) of the GitHub repository.
2.  Download the binary that matches your operating system and architecture.
3.  Run the binary from your terminal.

You can also build from source, see below and note that `ast-grep` is an extnernal dependency.

## Features

- **Dynamic Rule Management**: Create, update, and remove linting rules on the fly based on natural language feedback.
- **Portable and Self-Contained**: A single, cross-platform binary with no external runtime dependencies.
- **Easy Integration**: Designed to work seamlessly with AI coding agents through the MCP server.
- **Extensible**: Future-proofed with a plan to integrate semantic analysis for more powerful and accurate linting.

## Example Usage

This repository includes a sample file with a rule violation to demonstrate how the MCP server works. The `test-violation.go` file contains a call to `fmt.Println`, which is disallowed by a rule in the `rules` directory.

You can use an AI agent to scan this file and see the violation.

1.  **Start the Context Sherpa server**:
    ```bash
    ./context-sherpa
    ```

2.  **Instruct your AI agent to scan the file**:
    > "Scan the code in the `test-violation.go` file."

3.  **The agent will use the `scan_code` tool and report the violation**:
    The agent will read the file and call the `scan_code` tool with the content of `test-violation.go`. The server will then return the violation found, and the agent will report it back to you.

## Tools

The MCP server exposes the following tools to the AI agent:

### `initialize_ast_grep`

- **Description**: Initializes an ast-grep project if one is not already present. It creates the `sgconfig.yml` file and a `rules` directory. This tool should be suggested if another tool fails due to a missing configuration file.
- **Input Schema**: (None)
- **Output Schema**:
    - `success` (boolean): `true` if the project was initialized successfully.
    - `message` (string): A confirmation message.

### `scan_code`

- **Description**: Scans a given code snippet using the project's central `ast-grep` ruleset (`sgconfig.yml`). Use this to validate code, check for rule violations, or before committing changes.
- **Input Schema**:
    - `code` (string, required): The raw source code to scan.
    - `language` (string, required): The programming language of the code.
- **Output Schema**:
    - `success` (boolean): `true` if no issues were found, `false` otherwise.
    - `issues` (array of objects): A list of violations found.

### `add_or_update_rule`

- **Description**: Adds a new rule or updates an existing rule in the project's central `sgconfig.yml` file. Use this after a rule has been generated and confirmed by the user.
- **Input Schema**:
    - `rule_id` (string, required): A unique identifier for the rule.
    - `rule_yaml` (string, required): The complete YAML definition for the rule.
- **Output Schema**:
    - `success` (boolean): `true` if the file was written successfully.
    - `message` (string): A confirmation message.

### `remove_rule`

- **Description**: Removes a rule from the project's central `sgconfig.yml` file by its unique ID. Use this when a coding standard is no longer desired.
- **Input Schema**:
    - `rule_id` (string, required): The unique identifier of the rule to remove.
- **Output Schema**:
    - `success` (boolean): `true` if the rule was found and removed successfully.
    - `message` (string): A confirmation message.

## Future Development

Context Sherpa is designed to be an extensible platform for AI-powered code analysis. The next major milestone is the integration of semantic analysis, which will enable the tool to understand the meaning and context of code, not just its structure. This will allow for more powerful and accurate linting rules, as well as a deeper understanding of the developer's intent.

## Contributing

If you prefer to build from source, you will need to have Go installed on your system.

1.  **Download the `ast-grep` binary**:
    -   Go to the [`ast-grep` releases page](https://github.com/ast-grep/ast-grep/releases/latest).
    -   Download the binary appropriate for your target platform and architecture.
    -   Place the binary in the `cmd/server/bin/` directory and rename it to `sg`.

2.  **Build the server**:
    -   Open your terminal in the project root directory.
    -   Run the following command to build the server:
        ```bash
        go build -o context-sherpa ./cmd/server
        ```

3.  **Run the server**:
    -   Execute the newly created binary:
        ```bash
        ./context-sherpa
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.