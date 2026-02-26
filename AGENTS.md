# Agent Instructions — Explorations Repository

This repository contains interactive **Marimo notebooks** exploring data science, machine
learning, and scientific computing topics. All notebooks are written in Python using the
Marimo reactive notebook framework.

## Repository Overview

- All notebooks are standalone `.py` files using the Marimo app format
- Visualizations use **Altair** (primary), Matplotlib, and Plotly
- Dependencies managed with **uv** (`pyproject.toml` + `uv.lock`)
- Documentation summaries go in `copilot reports/`; experimental scripts go in `old scripts/`

## Key Conventions

- **Marimo cells**: Use `@app.cell` decorator; cells are reactive (like spreadsheet formulas)
- **Altair charts**: Use `alt.Chart(df).mark_*().encode(...)` pattern; always call `.interactive()`
- **Type hints**: Required on all function signatures
- **Function length**: Keep under 50 lines; split complex logic into helpers
- **Imports**: Each cell declares only what it needs; return all variables used by other cells

## Skills

Read the following skill files for domain-specific coding patterns before writing any code:

| Skill | File | Topic |
|-------|------|-------|
| Marimo | [`skills/marimo.md`](./skills/marimo.md) | Reactive notebook cell patterns |
| Altair | [`skills/altair.md`](./skills/altair.md) | Grammar-of-graphics visualization |
| Meta | [`skills/meta.md`](./skills/meta.md) | How to create and register new skills |

## File Management

- Use `uv` exclusively for dependency and environment management
- Preserve existing files; refactor in place rather than recreating
- Place utility and experimental scripts in `old scripts/`
- Place summaries and reports in `copilot reports/`

## Testing

There is no automated test suite. Validate notebooks by running:

```bash
marimo run <notebook>.py   # headless execution check
```
