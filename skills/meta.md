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
  agent-instructions.md  ← Canonical agent entry-point (all platforms symlink here)
  marimo.md              ← Marimo reactive-notebook patterns
  altair.md              ← Altair visualization patterns
  meta.md                ← This file
```

They contain **only content** — no platform-specific frontmatter. Platform wrappers (symlinks)
point to these files; the skills themselves stay portable.

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

## Step 2 — Platform Wiring (already done — no extra work needed)

All platform directories are **directory-level symlinks** pointing to `skills/`:

```
.github/instructions  → ../skills   (GitHub Copilot reads every file here)
.cursor/rules         → ../skills   (Cursor reads every file here)
```

Because the directories themselves are symlinks, **every new `skills/<topic>.md` file is
automatically visible to Copilot and Cursor** with no additional wiring. Just create the
skill file and you're done.

### Claude / Gemini / OpenCode / Copilot (agent entry-points)

All four root agent entry-point files are symlinks to `skills/agent-instructions.md`:

```
CLAUDE.md                       → skills/agent-instructions.md
GEMINI.md                       → skills/agent-instructions.md
AGENTS.md                       → skills/agent-instructions.md
.github/copilot-instructions.md → ../skills/agent-instructions.md
```

To update the skills table seen by every agent, edit **only**
`skills/agent-instructions.md` — all platforms pick up the change automatically.

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
     → Copilot (.github/instructions/) and Cursor (.cursor/rules/) pick it up automatically
     → Add a row to skills/agent-instructions.md skills table for Claude/Gemini/OpenCode
```
