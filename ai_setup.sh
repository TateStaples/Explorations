#!/bin/bash

# ==========================================
# 1. ROOT CONTEXT FILES
# ==========================================

# Create the universal source-of-truth file only if it does not already exist
if [[ -e AGENTS.md ]]; then
	echo "ℹ️ AGENTS.md already exists; leaving it unchanged."
else
	cat << 'EOF' > AGENTS.md
# Project Context
[Insert one-sentence description of the project, framework, and package manager]

# Global Rules
- Enforce strict typing.
- Never commit secrets or modify production database schemas.

# Build & Test
- Build: npm run build
- Test: npm run test
EOF
fi

# Use a native Mac symlink for CLAUDE.md instead of the text import
ln -shf AGENTS.md CLAUDE.md

# ==========================================
# 2. SKILLS DIRECTORY ARCHITECTURE
# ==========================================

# Create the source-of-truth skills directory only if it does not already exist
if [[ -e skills ]]; then
	echo "ℹ️ skills/ already exists; leaving it unchanged."
else
	mkdir -p skills
fi

# Create parent directories for AI platform config folders
mkdir -p .agent .cursor .github .agents
mkdir -p .claude

# Create relative symlinks so each platform points to the shared skills/
ln -shf ../skills .claude/skills
ln -shf ../skills .agent/skills
ln -shf ../skills .cursor/skills
ln -shf ../skills .github/skills
ln -shf ../skills .agents/skills

# ==========================================
# 3. GITIGNORE SETUP
# ==========================================

# Ignore the symlinked directories so only the source of truth is tracked
cat << 'EOF' >> .gitignore

# AI Agent Symlinked Skills (Source of truth is tracked in skills/)
.claude/skills
.agent/skills
.agents/skills
.cursor/skills
.github/skills
EOF

echo "✅ Setup complete. Root files, unified skills directories, and gitignore generated."