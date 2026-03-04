---
description: Audit all search pipeline fixes against current code
allowed-tools: Bash, Read, Grep, Glob
---
Read FIXES.md to get the complete list of fixes and their verification criteria.
For EVERY fix listed, run its "Verify" checks against streamlit_app.py.
Also check:
- Alias consistency: every alias appears in all 3 maps
- Syntax: python3 AST parse
- No duplicate definitions of key constants

Report results as:
  [PASS] Fix #N: short name
  [FAIL] Fix #N: short name — what's wrong

End with: "X/Y fixes verified. Z new fixes since last audit."
CMDEOF
