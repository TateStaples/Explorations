# AI Coding Assistant Guidelines for Claude

Clear, specific instructions to enhance collaboration and code quality.

## 📋 File Management
- **Use `uv` exclusively** for dependency and environment management (not pip, conda, or poetry)
- **Preserve existing files**: Refactor and update files in place; never delete and recreate
  - Use `replace_string_in_file` or `multi_replace_string_in_file` for edits
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
  - Examples: README.md, completion reports, analysis summaries
- **Preserve chat context**: Keep responses concise; avoid unnecessary verbosity