description: Add a new camp activity to all mapping layers

allowed-tools: Bash, Read, Grep, Glob, Write, Edit

argument-hint: [activity name] [SQL code numbers or "semantic-only"]

---

Add the activity "$ARGUMENTS" to the search pipeline. You MUST update ALL of these:

1. If SQL codes provided: add to ACTIVITY_CODES_SQL inside search_camps()

   If "semantic-only": add to SEMANTIC_ONLY_ACTIVITIES at module level

2. Add to _ACTIVITY_ALIASES inside search_camps()

3. Add to _PROCESS_ALIASES inside process_query()

4. Add to _ACTIVITY_SYNONYM_MAP at module level

5. If moving from SEMANTIC_ONLY to SQL codes: REMOVE from SEMANTIC_ONLY_ACTIVITIES

6. Add any common synonyms/typos to all three alias maps

After changes, run: python3 -c "import ast; ast.parse(open('streamlit_app.py').read()); print('OK')"

Then run /audit to verify nothing broke.

