# Agent Guidelines for Context Engineering

This guide provides best practices and specialized tools for AI coding agents working on **Context Sherpa**. By following these guidelines, agents can achieve expert-level accuracy while maintaining extreme token efficiency.

## 🏔️ Core Philosophy: Context-First Navigation

Before reading any file, an agent must establish context using symbolic and structural signals. This prevents "hallucinations" and keeps context windows lean.

### 1. Symbolic-First Research
- **Rule**: ALWAYS prefer `search_definitions` or `get_symbol_map` (SCIP) over `grep_search`.
- **Reason**: SCIP understands the logical relationships between files, whereas grep is just text matching.
- **Efficiency**: Symbolic searches pinpoint the exact line, avoiding the need to "hunt" through files.

### 2. High-Density File Distillation
- **Rule**: If a file is >100 lines, NEVER call `view_file` or `read_file` without first calling `list_symbols_in_file` with `distill: true`.
- **Reason**: The "distilled" summary provides a categorized table of contents and a semantic overview.
- **Efficiency**: You can identify the relevant 5% of a file and only read those specific line ranges.

### 3. Structural Analysis (ast-grep)
- **Rule**: Use `ast_grep_scan` for exact code patterns (e.g., "all functions with a database parameter").
- **Reason**: It uses the Abstract Syntax Tree (AST), making it far more accurate than regex for complex code shapes.

---

## 🤖 Specialized Agent Skills

Found in the [`.agents/skills/`](../.agents/skills/) directory, these skills extend an agent's capabilities:

- **[Architectural Grounding](../.agents/skills/architectural-grounding/SKILL.md)**: Safely explores large entry-point files.
- **[Symbolic Index Synchronization](../.agents/skills/symbolic-index-sync/SKILL.md)**: Ensures your "Big Brain" symbolic map is current after refactors.

## ⚡ Optimized Workflows

Found in the [`.agents/workflows/`](../.agents/workflows/) directory:

- **[/efficient-research](../.agents/workflows/efficient-research.md)**: A 4-tier process for navigating unknown codebases with maximum precision and minimum token cost.

---

## 💡 Best Practices for Token Efficiency

1.  **Summarize Before Returning**: Before sending a large block of code (>500 tokens) back to your main logic loop, use `summarize_code_intent`.
2.  **Surgical Reads**: Limit `view_file` to 50-100 lines at a time. If you need more, you probably haven't distilled the file enough.
3.  **Tiered Inference**: Use local models (Tier 2) for "noisy" work like code distillation or semantic triage, reserving Tier 3 (Gemini/Claude) for high-level strategy.
