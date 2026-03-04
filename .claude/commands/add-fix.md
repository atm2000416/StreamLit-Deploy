---
description: Document and verify a new fix
allowed-tools: Bash, Read, Grep, Glob, Write, Edit
argument-hint: [fix number, e.g. "12"]
---
A new fix has been applied. Complete these steps:

1. Read FIXES.md to get current fix count
2. Ask me to describe the problem and solution (or read from recent conversation)
3. Add Fix #$ARGUMENTS to FIXES.md following the existing format:
   - Problem, Solution, Verify checks, Test query
4. Update the /audit command to include verification for this fix
5. Update CLAUDE.md if the fix involves new architectural concepts
6. Run /audit to confirm all fixes (old + new) pass
7. Run syntax check
8. Suggest a git commit message
CMDEOF
