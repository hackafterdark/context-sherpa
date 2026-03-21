---
id: architectural-grounding
name: Architectural Grounding (Large File Discovery)
description: Safely explores and categorizes large entry-point files (>300 lines) without consuming massive context tokens.
tools_used: [list_symbols_in_file, summarize_code_intent, search_definitions]
trigger_conditions: [Assigned to a task in a file > 300 lines, exploring "App" or "Server" structs]
---

# Skill: Architectural Grounding

**Goal:** Understand the responsibility of a complex file without reading the raw source.

## Execution Pattern

1. **Metadata Triage:** If a file is known to be large (e.g., `app.go`, `main.go`, `server.go`), call `list_symbols_in_file` with `distill: true` immediately.
2. **Component Isolation:** Identify the 2-3 specific methods or structs in the distilled summary that are relevant to the query.
3. **Symbolic Zoom:** Call `search_definitions` or `get_symbol_map` on those specific components to find their definitions.
4. **Surgical Read:** ONLY call `read_file` or `view_file` for the specific line ranges identified in step 3. NEVER read more than 100 lines at once.

## Success Metric
The agent should be able to explain the "Flow" of a feature (like Database Init) while having read fewer than 15% of the total lines in the file.