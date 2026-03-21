---
id: symbolic-index-sync
name: Symbolic Index Synchronization
description: Maintains accuracy of the SCIP symbolic map after file changes, refactors, or when entering a new workspace.
tools_used: [initialize_scip, search_definitions, ls, dir]
trigger_conditions: ["Ghost Symbol", error, post-multi-file refactor, missing .context-sherpa folder]
---

# Skill: Symbolic Index Synchronization

**Goal:** Ensure the AI's symbolic map matches the current state of the disk, especially after creating new files or large refactors.

## Detection Logic (When to Sync)

1. **The "Ghost Symbol" Error:** If the agent just wrote a new struct/function but `search_definitions` returns 0 results for it.
2. **Post-Refactor:** After the agent completes a multi-file edit (e.g., renaming a package).
3. **New Workspace:** Upon first entering a workspace that lacks a `.context-sherpa/` directory.

## Execution Pattern

1. **Verification:** Call `ls` or `dir` on the project root to confirm the new files exist on disk.
2. **Trigger:** Call the `initialize_scip` tool. If the language is not automatically detected, specify it based on the file extensions found.
3. **Verification:** Re-run the original `search_definitions` or `get_symbol_map` call to confirm the new logic is now "visible" to the symbolic engine.

## Token Efficiency Hint

Do not re-index for every single line of code added. Only trigger a sync when you need to perform relational analysis (e.g., "Who calls this new method?") that simple `grep` or the editor's local buffer cannot handle accurately.