description: Trace how a search query flows through the pipeline

allowed-tools: Read, Grep, Glob

argument-hint: [search query, e.g. "ballet camps in vaughan"]

---

Trace how the query "$ARGUMENTS" would flow through the search pipeline in streamlit_app.py:

1. What would extract_filters return? (activity, province, region, age, style)

2. Is the activity in ACTIVITY_CODES_SQL? What codes? Or is it in SEMANTIC_ONLY_ACTIVITIES?

3. Which search path fires: SQL codes, semantic-only (limit=200), or hybrid?

4. Would HARD GATE trigger? Would BROAD-CODE REFINEMENT apply?

5. What confidence gate applies? (taxonomy 0.80, unknown 0.855, or code-match bypass)

6. If no results: which _build_no_results_response case fires?

7. If results: what score range would you expect? Would _needs_semantic_refinement be True?

Read the actual code to answer — don't guess from memory.

