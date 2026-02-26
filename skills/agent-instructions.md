# AI Coding Assistant Instructions — Explorations Repository

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

## 📋 File Management
- **Use `uv` exclusively** for dependency and environment management (not pip, conda, or poetry)
- **Preserve existing files**: Refactor and update files in place; never delete and recreate
- **Organize non-core scripts**: Place utility and experimental scripts in `old scripts/` folder
- **Minimize terminal output**: Avoid `cat` for large files (bloats context); use file reading tools instead

## 💻 Code Quality Standards
- **Type hints required**: Add type annotations to all function signatures for clarity
- **Function scope**: Keep functions under 50 lines—break into smaller, focused functions
- **Debug early**: Include assertions in code cells to catch issues immediately

## 📓 Notebook Development
- **Interactive-first design**: Use marimo cells (not standard Python) for better interactivity
- **Markdown for text**: Display explanations and large text blocks in markdown cells, not print statements
- **Organize output**: Use `mo.vstack()`, `mo.hstack()`, and layout functions for structure

## 📚 Documentation & Organization
- **Centralize documentation**: Place all summaries, explanations, and reports in `copilot reports/` folder
- **Preserve chat context**: Keep responses concise; avoid unnecessary verbosity

## 🛠️ Skills

Read the following skill files for domain-specific coding patterns before writing any code:

| Skill | Source file | Purpose |
|-------|-------------|---------|
| Marimo | [`skills/marimo.md`](skills/marimo.md) | Reactive notebook cell patterns |
| Altair | [`skills/altair.md`](skills/altair.md) | Grammar-of-graphics visualization |
| Meta | [`skills/meta.md`](skills/meta.md) | How to create and register new skills |

## Testing

There is no automated test suite. Validate notebooks by running:

```bash
marimo run <notebook>.py   # headless execution check
```

## Symlinks — Multi-Repo Reuse

To share skills from this repository in another repository without duplicating content:

```bash
# In a sibling repo, symlink to the canonical skill files:
ln -s ../../Explorations/skills/marimo.md  skills/marimo.md
ln -s ../../Explorations/skills/altair.md  skills/altair.md

# Then set up platform wrappers as described in skills/meta.md.
```
