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

Domain-specific coding guidance lives in `.github/instructions/`:

| File | Applies to | Purpose |
|------|-----------|---------|
| [`marimo.instructions.md`](./instructions/marimo.instructions.md) | `**/*.py` | Marimo notebook patterns |
| [`altair.instructions.md`](./instructions/altair.instructions.md) | `**/*.py` | Altair visualization patterns |

## Symlinks — Multi-Repo Reuse

To share these skills across multiple repositories without duplicating them, use symlinks:

```bash
# In a sibling repo, symlink to this repo's instructions:
ln -s ../../Explorations/.github/instructions/marimo.instructions.md \
      .github/instructions/marimo.instructions.md

ln -s ../../Explorations/.github/instructions/altair.instructions.md \
      .github/instructions/altair.instructions.md
```

Alternatively, maintain a dedicated `skills/` repository and symlink from there:

```bash
# Central skills repo at ~/skills/
ln -s ~/skills/marimo.instructions.md .github/instructions/marimo.instructions.md
ln -s ~/skills/altair.instructions.md .github/instructions/altair.instructions.md
```

Symlinked files are picked up by VS Code and GitHub Copilot just like regular files. Keep
symlink targets under version control in their source repository so updates propagate
automatically to all consumers.
