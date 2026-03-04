---
description: Run test scenarios to discover potential new issues
allowed-tools: Bash, Read, Grep, Glob
---
Read FIXES.md to understand all known fixes and their test queries.
Then systematically trace these scenarios through the code to find NEW issues:

**Regression scenarios** (verify existing fixes still work):
Run each "Test query" from FIXES.md through a code trace.

**Edge case scenarios** (discover new issues):
1. "guitar camps in mississauga" — broad-code (music=37) + GTA alias
2. "acting camps for girls age 12 in ottawa" — broad-code (theatre=59) + age + gender
3. "coding camps" → "make it overnight" — MERGE branch + semantic-only activity
4. "hockey camps in toronto" → "actually, ballet" — activity change + FRESH branch
5. "camps near me" — no activity, no location (what happens?)
6. "french immersion camps in quebec" — bilingual edge case
7. "LEGO camps in vancouver" — case sensitivity + semantic-only
8. "swimming camps" → "yes" (when AI didn't suggest anything) — affirmative without suggestion
9. "dance camps in toronto" — broad category name (should NOT trigger refinement)
10. "overnight ballet camps for 8 year old girls in GTA" — every filter at once

For each scenario:
- Trace the exact code path
- Check if the result would be correct
- Flag anything suspicious as a potential Fix #12, #13, etc.

Report: "X scenarios clean, Y potential issues found" with details on each issue.
CMDEOF
