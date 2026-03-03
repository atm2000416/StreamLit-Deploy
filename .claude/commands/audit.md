description: Run regression audit on all 11 search pipeline fixes

allowed-tools: Bash, Read, Grep, Glob

---

Audit streamlit_app.py to verify all 11 fixes are intact. Check each one and report pass/fail:

1. SEMANTIC_ONLY_ACTIVITIES defined at module level (column 0, not inside a function)

2. _ACTIVITY_SYNONYM_MAP exists and synonym_terms used in _validate_filters

3. TAXONOMY_CONFIDENCE_THRESHOLD = 0.80 with both taxonomy and unknown branches

4. _semantic_only_limit uses 200 for semantic-only activities

5. Umbrella categories: sports [29,54,66...], arts, computers, education, health, adventure [181,24,41,58,265]. Military/ropes/travel NOT in SEMANTIC_ONLY.

6. Location inheritance in 2+ branches with _broad_provinces guard

7. _activity_originally_had_codes flag set and used in _act_is_taxonomy

8. Two-pass merge: _merge_filters with activity=None, limit=200

9. _diagnose_availability function with specialty_label LIKE search, diagnosis passed to 2+ call sites

10. Fuzzy affirmative detection (_affirm_words) + pre-tree SUGGESTION ACCEPTED path

11. BROAD_PARENT_CODES with blend formula 0.7*semantic + 0.3*0.9, ballet/guitar/acting in check list

Also verify: alias consistency (sport→sports, puppy→animals, outdoor adventure→adventure each in 3 maps).

Run syntax check at the end.

