# PRD & Technical Plan: context-sherpa - AI-Powered Code Analysis Server

**Version: 1.3**
**Date: 2025-10-01**
Author: Gemini

## 1. Overview

This document outlines the requirements for an MCP (Model-as-a-Tool Protocol) server, written in Go, that provides an AI coding agent with tools to interact with `ast-grep`. The primary objective is to create a system where an AI agent can not only lint and validate code using a predefined set of rules but also dynamically create, update, and remove those rules based on natural language feedback from a developer.

The final product will be a single, portable, cross-platform binary with no external runtime dependencies, making setup trivial for the end-user.

## 2. Core Objective & User Story

As a developer using an AI coding agent, I want to:

-   Have my agent automatically validate the code it generates against my project's specific coding patterns.
-   Be able to provide natural language feedback (e.g., "From now on, all async functions must have a try/catch block") to my agent.
-   Have the agent intelligently convert my feedback into a permanent, machine-readable linting rule using `ast-grep`.
-   Be able to easily remove rules that are no longer needed.
-   Ensure this system is self-contained in a single executable that I can easily run without managing servers, dependencies, or security risks.

## 3. Key Features & Tool Definitions for the AI Agent

The MCP server will expose the following **seven** tools to the AI agent. The agent will use the tool descriptions to decide which tool to call based on the user's request.

### Tool 1: `initialize_ast_grep`

-   **Description**: "Initializes an ast-grep project if one is not already present. It creates the `sgconfig.yml` file and a `rules` directory. This tool should be suggested if another tool fails due to a missing configuration file."
-   **Input Schema**: (None)
-   **Output Schema**:
    -   `success` (boolean): `true` if the project was initialized successfully.
    -   `message` (string): A confirmation message (e.g., "ast-grep project initialized successfully. Created sgconfig.yml and rules/ directory.").

### Tool 2: `scan_code`

-   **Description**: "Scans a given code snippet using the project's central `ast-grep` ruleset (`sgconfig.yml`). Use this to validate code, check for rule violations, or before committing changes."
-   **Input Schema**:
    -   `code` (string, required): The raw source code to scan.
    -   `language` (string, required): The programming language of the code (e.g., `javascript`, `python`, `go`).
-   **Output Schema**:
    -   `success` (boolean): `true` if no issues were found, `false` otherwise.
    -   `issues` (array of objects): A list of violations found. Each object contains:
        -   `ruleId` (string): The ID of the rule that was violated.
        -   `message` (string): The error message for the violation.
        -   `line` (integer): The line number where the violation occurred.

### Tool 3: `add_or_update_rule`

-   **Description**: "Adds a new rule or updates an existing rule in the project's central `sgconfig.yml` file. Use this after a rule has been generated and confirmed by the user."
-   **Input Schema**:
    -   `rule_id` (string, required): A unique identifier for the rule (e.g., `no-console-log`).
    -   `rule_yaml` (string, required): The complete YAML definition for the rule.
-   **Output Schema**:
    -   `success` (boolean): `true` if the file was written successfully.
    -   `message` (string): A confirmation message (e.g., "Rule 'no-console-log' was added successfully.").

### Tool 4: `remove_rule`

-   **Description**: "Removes a rule from the project's central `sgconfig.yml` file by its unique ID. Use this when a coding standard is no longer desired."
-   **Input Schema**:
    -   `rule_id` (string, required): The unique identifier of the rule to remove.
-   **Output Schema**:
    -   `success` (boolean): `true` if the rule was found and removed successfully.
    -   `message` (string): A confirmation message (e.g., "Rule 'no-console-log' was removed successfully.").

### Tool 5: `search_community_rules`

-   **Description**: "Search the community rule repository for ast-grep rules. Use this to discover new rules for common problems like security vulnerabilities or style issues."
-   **Input Schema**:
    -   `query` (string, required): A natural language query (e.g., "sql injection", "check for todos").
    -   `language` (string, optional): Filter by language (e.g., "go", "python").
    -   `tags` (string, optional): Comma-separated list of tags to filter by (e.g., "security,database").
-   **Output Schema**:
    -   `success` (boolean): `true` if the search was successful.
    -   `results` (string): A formatted string listing the matching rules, including their ID, description, and tags.

### Tool 6: `get_community_rule_details`

-   **Description**: "Get the full YAML content and explanation for a specific community rule. Use this to let the user inspect a rule before importing it."
-   **Input Schema**:
    -   `rule_id` (string, required): The unique ID of the rule from the search results.
-   **Output Schema**:
    -   `success` (boolean): `true` if the rule was found.
    -   `details` (string): A formatted string containing the rule's full metadata and YAML content.

### Tool 7: `import_community_rule`

-   **Description**: "Download a community rule and add it to the local project's rule directory. Use this after the user has confirmed they want to add a specific rule."
-   **Input Schema**:
    -   `rule_id` (string, required): The unique ID of the rule to import.
-   **Output Schema**:
    -   `success` (boolean): `true` if the rule was imported successfully.
    -   `message` (string): A confirmation message (e.g., "Rule 'ast-grep-go-sql-injection' was imported successfully.").

## 4. Technical Implementation Plan

This plan details the steps to build the server using Go, embedding the `ast-grep` binary directly.

### Step A: Project Setup
(No changes)

### Step B: Bundle the ast-grep Binary
(No changes)

### Step C: Implement the MCP Server

-   In `main()`, create a new MCP server instance.
-   Register a handler for each of the three tools defined above.
    -   `server.RegisterTool("scan_code", scanCodeHandler)`
    -   `server.RegisterTool("add_or_update_rule", addOrUpdateRuleHandler)`
    -   `server.RegisterTool("remove_rule", removeRuleHandler)`
-   Start the server to listen for requests from the AI agent.

### Step D: Implement Tool Handler Functions

-   `scanCodeHandler(req mcp.Request) mcp.Response`: (No changes)
-   `addOrUpdateRuleHandler(req mcp.Request) mcp.Response`: (No changes)
-   `removeRuleHandler(req mcp.Request) mcp.Response`: (No changes)

## 5. Non-Functional Requirements
(No changes)

## 6. Testing Strategy & Usage Example

### Testing

-   **Unit Tests**: Each handler function (`scanCodeHandler`, `addOrUpdateRuleHandler`, etc.) will have corresponding unit tests. External dependencies will be mocked.
-   **Integration Tests**: An end-to-end test script will be created to compile the binary, start it, and simulate an MCP client making a sequence of calls to add, scan, and remove rules, verifying the `sgconfig.yml` content at each stage.

### Example Usage Workflow

This scenario illustrates the updated interaction between a developer, the AI agent, and the MCP server.

1.  **Agent handles a missing configuration file:**
    -   **Developer**: "Hey agent, please add a rule to disallow `fmt.Println` in our Go code."
    -   **AI Agent**: Calls the `add_or_update_rule` tool.
    -   **MCP Server**: Fails because `sgconfig.yml` does not exist and returns an error: "Error: sgconfig.yml not found. Please run the 'initialize_ast_grep' tool first to set up the project."
    -   **AI Agent**: "It looks like this project hasn't been set up for ast-grep yet. Would you like me to initialize it for you?"
    -   **Developer**: "Yes, please."
    -   **AI Agent**: Calls the `initialize_ast_grep` tool.
    -   **MCP Server**: Creates `sgconfig.yml` and the `rules/` directory.
    -   **AI Agent**: "The project has been initialized. I will now add the rule."
    -   **AI Agent**: Calls `add_or_update_rule` again, which now succeeds.

2.  **Developer sets a new rule:**
    -   **Developer**: "Hey agent, we should never use the `eval()` function in our JavaScript. It's a security risk. Can you create a rule to block it?"
    -   **AI Agent**: Understands the intent. It uses its internal knowledge to generate a valid `ast-grep` YAML rule.
    -   **AI Agent**: "I've drafted this rule: [shows YAML to user]. Does this look correct?"
    -   **Developer**: "Yes, that's perfect."
    -   **AI Agent**: Calls the `add_or_update_rule` tool with the YAML to save it to `sgconfig.yml`.

2.  **Agent self-corrects its own code:**
    -   **Developer**: "Now, please refactor this old utility file to be more modern."
    -   **AI Agent**: Generates a refactored version of the code, mistakenly using `eval()`.
    -   **Agent (Internal Step)**: Before showing the code, it calls the `scan_code` tool on its own output.
    -   **MCP Server**: Runs `ast-grep scan`, finds the `eval()` call, and returns a violation.
    -   **AI Agent**: Sees the violation, corrects its code to use a safer alternative, and re-scans to confirm compliance.
    -   **AI Agent**: "Here is the refactored code. I've also updated a part that was using `eval()` to be more secure."

3.  **Developer removes a rule:**
    -   **Developer**: "You know what, the `no-console-log` rule is too annoying for our debug builds. Please remove it."
    -   **AI Agent**: Calls the `remove_rule` tool with `rule_id: "no-console-log"`.
    -   **MCP Server**: Removes the rule from `sgconfig.yml`.
    -   **AI Agent**: "Done. The `no-console-log` rule has been removed."

4.  **Developer discovers and imports a community rule:**
    -   **Developer**: "Are there any good community rules for finding unchecked errors in Go?"
    -   **AI Agent**: Calls `search_community_rules(query="unchecked error", language="go")`.
    -   **MCP Server**: Fetches the community `index.json`, finds a matching rule, and returns its details.
    -   **AI Agent**: "I found a rule: `ast-grep-go-unchecked-error`. It finds function calls that return an error that is not assigned or checked. Would you like to see the details?"
    -   **Developer**: "Yes, show me."
    -   **AI Agent**: Calls `get_community_rule_details(rule_id="ast-grep-go-unchecked-error")`.
    -   **MCP Server**: Fetches the rule's YAML and returns it.
    -   **AI Agent**: Displays the full rule details and YAML to the developer.
    -   **Developer**: "Looks great, let's add it to our project."
    -   **AI Agent**: Calls `import_community_rule(rule_id="ast-grep-go-unchecked-error")`.
    -   **MCP Server**: Downloads the rule YAML and saves it to the local `rules/` directory.
    -   **AI Agent**: "The rule has been imported and is now active."