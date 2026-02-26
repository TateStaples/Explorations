# Meta-Skill: How to Create and Share Skills

This guide explains how to write a new skill file and wire it up to every supported AI
coding platform so it becomes a **single source of truth** that all agents consume
automatically.

---

## What Is a Skill File?

A skill file is a Markdown document that gives an AI coding assistant focused, domain-specific
guidance — coding patterns, idioms, common pitfalls, and copy-paste examples — for a
particular library or workflow.

Skill files live in `skills/` at the root of the repository:

```
skills/
  marimo.md    ← Marimo reactive-notebook patterns
  altair.md    ← Altair visualization patterns
  meta.md      ← This file
```

They contain **only content** — no platform-specific frontmatter. Platform wrappers (symlinks
or thin entry-point files) handle configuration; the skill itself stays portable.

---

## Step 1 — Write the Skill File

Create `skills/<topic>.md`. A good skill file includes:

| Section | What to put there |
|---------|-------------------|
| **One-liner intro** | "Use these patterns whenever …" |
| **Core anatomy** | The fundamental structure or grammar with an annotated example |
| **Quick-reference tables** | Type suffixes, option names, key APIs |
| **Copy-paste patterns** | 3–5 named, realistic code snippets |
| **Pitfalls table** | Two-column ❌ Avoid / ✅ Prefer |

Keep each pattern block under ~30 lines. Prefer real code over prose.

---

## Step 2 — Wire Up Platform Symlinks

After creating `skills/<topic>.md`, register the skill with each platform by creating a
**symlink** from that platform's expected location to the canonical file. Symlinks keep the
content in sync automatically — editing `skills/<topic>.md` updates every platform at once.

### GitHub Copilot

```bash
# From repo root
ln -s ../../skills/<topic>.md .github/instructions/<topic>.instructions.md
```

Copilot reads all files in `.github/instructions/` automatically. Without a `applyTo`
frontmatter the rule applies globally; add a thin wrapper if you need file scoping.

### Cursor

```bash
mkdir -p .cursor/rules
ln -s ../../skills/<topic>.md .cursor/rules/<topic>.mdc
```

Cursor reads `.cursor/rules/*.mdc`. Without a `globs` frontmatter the rule is available
for manual `@`-reference. Add a thin `.mdc` wrapper with `globs` if you want auto-injection.

### Claude

Add a bullet to `CLAUDE.md` under the **Skills** section pointing to `skills/<topic>.md`.
Claude reads `CLAUDE.md` at session start and will follow the reference.

### Gemini CLI / Gemini in IDEs

Add a bullet to `GEMINI.md` under the **Skills** section pointing to `skills/<topic>.md`.

### OpenCode / Codex / generic agents

Add a bullet to `AGENTS.md` pointing to `skills/<topic>.md`.

---

## Step 3 — Update the Entry-Point Files

Open each of the four root entry-point files and add one line to the skills table:

| File | Audience |
|------|----------|
| `CLAUDE.md` | Anthropic Claude (Claude.ai, API, VS Code extension) |
| `GEMINI.md` | Google Gemini CLI and IDE integrations |
| `AGENTS.md` | OpenCode, OpenAI Codex, and any agent reading `AGENTS.md` |
| `.github/copilot-instructions.md` | GitHub Copilot (VS Code, JetBrains, CLI) |

---

## Cross-Repository Reuse via Symlinks

To share skills from this repository in **another** repository without duplicating content:

```bash
# Inside the other repo, run once per skill:
ln -s ../../Explorations/skills/marimo.md  skills/marimo.md
ln -s ../../Explorations/skills/altair.md  skills/altair.md

# Then set up platform wrappers as normal (pointing to skills/*.md in THAT repo).
```

Or maintain a dedicated central `~/skills/` directory and symlink from there:

```bash
ln -s ~/skills/marimo.md  skills/marimo.md
```

---

## Checklist for a New Skill

```
[ ] Create skills/<topic>.md with content (no platform frontmatter)
[ ] ln -s ../../skills/<topic>.md  .github/instructions/<topic>.instructions.md
[ ] ln -s ../../skills/<topic>.md  .cursor/rules/<topic>.mdc
[ ] Add bullet to CLAUDE.md skill table
[ ] Add bullet to GEMINI.md skill table
[ ] Add bullet to AGENTS.md skill table
[ ] Add row to .github/copilot-instructions.md skill table
```
