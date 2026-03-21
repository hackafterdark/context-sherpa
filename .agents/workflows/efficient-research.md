---
description: Efficient Codebase Exploration Workflow
---

# Workflow: Efficient Codebase Exploration

Use this workflow when tasked with exploring a new area of the codebase, understanding a large file, or tracing a bug through multiple layers.

## Steps

1. **Global Search (Tier 1)**
   - Use `search_definitions` or `get_symbol_map` to find the exact file and line number of the target symbol.
   - **Avoid** broad `grep_search` calls unless you are searching for literal text strings that aren't indexed symbols.

2. **File Distillation (Tier 2)**
   - Once a file is identified, check its size. If it is over 100 lines, **must** call `list_symbols_in_file` with `distill: true`.
   - Study the semantic categories and function signatures provided in the distilled output.

3. **Intent Verification (Tier 3)**
   - For specific functions identified in step 2 that seem relevant, call `summarize_code_intent`.
   - Use the 3-sentence summary (Inputs, Outputs, Side-effects) to confirm if this is the logic you need to read.

4. **Surgical Read (Tier 4)**
   - Use the line numbers from the distillation step to perform a targeted `view_file` call.
   - **Limit** the read to 50-100 lines at a time.
   - **Never** read the entire file if you only need to understand one method.

## Critical Rules
- **Token First**: Every `view_file` call without a preceding `list_symbols_in_file` (for large files) is a token efficiency failure.
- **Precision First**: Prioritize SCIP (symbolic) tools over regex (text) tools.
