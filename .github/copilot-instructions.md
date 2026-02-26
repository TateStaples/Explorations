# GitHub Copilot Instructions — Explorations Repository

This repository contains interactive **Marimo notebooks** exploring data science, machine learning, and scientific computing topics. All notebooks are written in Python using the Marimo reactive notebook framework.

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

## Skill Files

Domain-specific coding guidance lives in `skills/` as the **single source of truth**.
The files in `.github/instructions/` are symlinks that point there.

| Skill | Source file | Purpose |
|-------|-------------|---------|
| Marimo | [`skills/marimo.md`](../skills/marimo.md) | Marimo notebook patterns |
| Altair | [`skills/altair.md`](../skills/altair.md) | Altair visualization patterns |
| Meta | [`skills/meta.md`](../skills/meta.md) | How to create & register new skills |

## Symlinks — Multi-Repo Reuse

To share skills from this repository in another repository without duplicating content:

```bash
# In a sibling repo, symlink to the canonical skill files:
ln -s ../../Explorations/skills/marimo.md  skills/marimo.md
ln -s ../../Explorations/skills/altair.md  skills/altair.md

# Then set up platform wrappers as described in skills/meta.md.
```

Symlinked files are resolved by VS Code and all major AI tooling. Keep symlink targets
under version control in their source repository so updates propagate automatically.

