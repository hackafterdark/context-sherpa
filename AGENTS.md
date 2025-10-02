# AGENTS.md - Project-Wide Conventions

This document provides essential context and rules for AI agents working on this Go API project. It covers the Go backend, the development workflow, and project-wide standards.

## 1. Development Workflow

The development process is managed by standard Go tooling.

-   **Primary Command**: To run the application in development mode, execute the following command from the **project root directory**:
    ```bash
    go run ./...
    ```
-   **Building for Production**: To create a production-ready executable, run the following command from the **project root directory**:
    ```bash
    go build -o myapp
    ```
-   **Running Tests**: To run all tests, execute the following command from the **project root directory**:
    ```bash
    go test ./...
    ```

## 2. Go Backend Conventions

Adherence to these conventions is crucial for a clean, maintainable, and idiomatic Go codebase.

-   **Idiomatic Go**: Write clear, simple, and idiomatic Go. Follow standard conventions for naming variables, packages, and interfaces. Avoid creating unnecessary abstractions or wrapper functions.
-   **Prefer Copying over Abstraction**: When a small amount of utility code is needed in multiple places, prefer copying the code over creating a new shared package. A little duplication is often better than introducing a premature or incorrect abstraction that creates an unnecessary dependency. This follows the Go proverb: "A little copying is better than a little dependency."
-   **Variable Naming**: Follow Go's convention for variable naming. Use short, concise variable names (e.g., `i`, `buf`, `req`) for variables with a limited scope and lifespan. As a variable's scope and distance from its declaration grows, use a longer, more descriptive name (e.g., `userCache`, `incomingRequest`). This improves readability by signaling the variable's importance and scope.
-   **Code Formatting**: All Go code must be formatted with `go fmt`. It is strongly recommended to configure your editor to run `go fmt` on save.
-   **Logging**: Use the `github.com/charmbracelet/log` package for all logging. Do not use the standard `log` or `fmt` packages for application logging.
-   **Testing**: Write unit tests for all new functions, especially for business logic and utility packages. Place test files alongside the code they are testing (e.g., `app_test.go`).
-   **GoDoc Comments**: All exported (public) types, functions, constants, and variables **must** have `godoc`-compliant comments. The comment block must begin with the name of the element it is describing, followed by a period. This is the standard format recognized by the `go doc` tool and is essential for generating clean, readable documentation. Do not include extraneous words like "godoc" in the first line of the comment.


## 3. Directory Structure

This project follows the principles of the [Standard Go Project Layout](https://github.com/golang-standards/project-layout).

-   **/cmd**: Main application entry points. Each subdirectory is a separate executable.
    -   **Application Structure**: Within a specific application's directory (e.g., `/cmd/my-app`), **Do not** create an extra `/internal` directory inside an application's subdirectory (e.g., `/cmd/my-app/internal`). The `cmd` directory structure already prevents cross-application imports, making this redundant.
-   **/internal**: Use this directory at the **root level** for code that is shared between multiple applications within this project but should not be exported for use by other projects.
-   **/pkg**: Public library code that is safe for external use. This is where reusable packages should be placed.
-   **/configs**: Configuration file templates or default configurations.
-   **/scripts**: Scripts to perform various build, install, analysis, or other operations.


## 4. Dependency Management (Go Modules)

All Go dependencies must be managed using Go Modules. The `go.mod` and `go.sum` files are the source of truth for the project's dependency graph.

-   **Adding a Dependency**: To add a new dependency, do not manually edit the `go.mod` file. Instead, run the following command in the project root:
    ```bash
    go get github.com/some/package
    ```
    This command will find the latest version of the package, add it to `go.mod`, and update `go.sum` with its checksum.

-   **Tidying Dependencies**: After adding or removing imports in your code, always run `go mod tidy`. This command is essential for maintaining a clean `go.mod` file. It will:
    -   Remove any dependencies that are no longer used in the code.
    -   Add any dependencies that are imported in the code but are missing from `go.mod`.

-   **Updating Dependencies**: To update a specific dependency to its latest version, use `go get`:
    ```bash
    go get -u github.com/some/package
    ```
    To update all dependencies, use `go get -u ./...`.

-   **Security**: Never manually edit the `go.sum` file. It is managed entirely by the `go` toolchain to ensure the integrity and security of your dependencies.


## 5. Testing Philosophy and Guidelines

A comprehensive test suite is critical for ensuring code quality, preventing regressions, and enabling confident refactoring.

-   **Unit Testing**:
    -   **Scope**: Unit tests **must** be written for all new or modified functions and methods, focusing on a single unit of logic in isolation. All external dependencies must be mocked.

-   **General Guidelines**:
    -   **Naming**: Test functions must be named `Test<FunctionName>`. For table-driven tests, use descriptive names for each sub-test case.
    -   **Assertions**: Use the `stretchr/testify/assert` or `stretchr/testify/require` packages for clear and concise test assertions.


## 6. Human-Agent Collaboration

To ensure a smooth and transparent development process, agents must adhere to the following conventions when collaborating with human developers.

-   **Planning and Confirmation**: For any non-trivial task, the agent **must** first outline its plan of action and wait for user approval before making changes. This ensures alignment and prevents unnecessary work.
-   **Clarity over Brevity**: The agent **must** be explicit in its explanations. When proposing a change, it should explain *what* the change is and *why* it's being made.
-   **Comment with Intent**: Go beyond explaining *what* the code does. Add comments that explain *why* a particular implementation was chosen, especially for non-obvious logic or complex decisions. This provides crucial context for human developers.
-   **Standardized `TODO` Comments**: When work is incomplete or requires human intervention, use a standardized `TODO` comment. This creates a clear and trackable action item.
    -   **Format**: `// TODO (agent): [A clear, concise description of what is needed]`
    -   **Example**: `// TODO (agent): Refactor this to use the new UserService once it's available.`
-   **Clarity in Commits**: All git commits must have clear and descriptive messages. The message should summarize the change and its purpose, providing context for code reviewers.
-   **Handling Ambiguity**: If a task description is ambiguous or underspecified, do not make significant assumptions.
    -   **Priority**: Ask for clarification from a human developer.
    -   **Minor Assumptions**: If a minor, non-critical assumption is necessary to proceed, it must be clearly documented in a code comment alongside the implementation.

-   **Testing Mandate**: All generated or modified code, regardless of scope, **must** be accompanied by corresponding test cases. This is not limited to new functions; any change to existing logic requires either new tests or updates to existing tests to validate the change.

-   **Continuous Verification**: After every code modification—whether adding a feature, fixing a bug, or refactoring—the agent **must** run the full test suite using `go test ./...`. This ensures a tight feedback loop and prevents the introduction of regressions. The agent must not consider a task complete until all tests pass.

-   **Confirm Package Substitutions**: If a user explicitly requests the use of a specific package, library, or dependency, and the agent cannot find or install it using standard tools (`go get`, `context7`, etc.), the agent **must not** unilaterally substitute it with an alternative. Instead, it **must** inform the user of the issue and ask for explicit permission to proceed with a substitute, providing the reason for the proposed change.
    -   **Example Scenario**:
        -   **User Request**: "Use the `go-fuzz` library for testing."
        -   **Agent Action (Incorrect)**: Agent fails to find `go-fuzz`, assumes it's a typo, and uses the standard `testing` package instead without asking.
        -   **Agent Action (Correct)**: "I was unable to find the `go-fuzz` library using my tools. Would you like me to proceed by using the standard `testing` package instead, or can you provide a specific version or import path for `go-fuzz`?"


## 7. MCP Tool Usage Conventions

To effectively manage the `ast-grep` linting rules, the agent must follow a specific workflow. This ensures that changes are intentional, validated, and approved by the user.

-   **Self-Correction Loop**: Before presenting any generated or modified code to the user, the agent **must** first validate its own output by calling the `scan_code` tool. If any violations are found, the agent must attempt to fix them and re-scan until the code is compliant. This prevents the agent from suggesting code that violates established project patterns.

-   **Rule Management Workflow**: When a user asks to create a new linting rule, the agent must:
    1.  **Generate**: Use its own intelligence to generate the complete `ast-grep` rule in YAML format based on the user's request.
    2.  **Confirm**: Present the generated YAML to the user for confirmation. The agent **must not** proceed without explicit user approval.
    3.  **Add**: Once the user approves, call the `add_or_update_rule` tool with the confirmed YAML to save it.

-   **Project Initialization**: If any tool fails due to a missing `sgconfig.yml` file, the agent should:
    1.  **Detect**: Recognize the error message indicating a missing configuration.
    2.  **Suggest**: Inform the user that the project has not been initialized for `ast-grep`.
    3.  **Initialize**: Ask for permission to run the `initialize_ast_grep` tool to create the necessary configuration files.

-   **Idempotent Rule IDs**: When calling `add_or_update_rule`, the `rule_id` should be a descriptive, kebab-case string derived from the rule's purpose (e.g., `disallow-eval-function`, `require-try-catch-async`). This ensures that rule IDs are predictable and avoids accidental duplication.

-   **User Confirmation for Removal**: Before calling the `remove_rule` tool, the agent **must** confirm with the user which specific `rule_id` they wish to remove. The agent should not infer or guess the rule ID.
