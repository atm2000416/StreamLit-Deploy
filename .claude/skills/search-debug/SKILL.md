---

name: search-debug

description: Debug camp search issues from trace logs. Use when analyzing why a search returned wrong results, no results, or incorrect match percentages.

---

When debugging search issues from trace logs:

1. Identify the activity and check:

   - Is it in ACTIVITY_CODES_SQL? What codes?

   - Is it in SEMANTIC_ONLY_ACTIVITIES?

   - Is it a sub-activity of a BROAD_PARENT_CODE (22=dance, 37=music, 59=theatre)?

2. Check the search path from the trace:

   - Did SQL use specialty codes? (look for "sc.specialty IN (...)")

   - What fallback value? (None, province_only, no_match)

   - Did HARD GATE fire? Did it re-fetch with limit=200?

3. For match percentage issues:

   - If code_match_used=True and NOT broad-code: all camps get 0.9 (expected)

   - If code_match_used=True AND broad-code: should see BROAD-CODE REFINEMENT in logs

   - If semantic-only: check elbow detection scores and LOW_CONFIDENCE gate

4. For "I'm not familiar" errors:

   - Check _act_is_taxonomy: is activity in SEMANTIC_ONLY or does _activity_originally_had_codes=True?

   - Check all three alias maps for consistency

5. For follow-up failures:

   - Check which decision tree branch fired (SUGGESTION, REUSE, FRESH, MERGE, DEFAULT)

   - Check if is_affirmative matched (fuzzy word-level or exact list)

   - Check if _suggested_region was parsed from AI's last message

   - Check if location was inherited or lost

