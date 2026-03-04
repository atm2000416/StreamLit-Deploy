# CampSearch AI Chatbot

## What This Is

Streamlit-based camp search chatbot for camps.ca. Uses Gemini for filter extraction

and blurb generation, Google embedding-001 for semantic scoring, MySQL for camp data.

Single file: streamlit_app.py (~3,200 lines).

## Search Pipeline (read before ANY changes)

The pipeline flows in this order:

1. extract_filters() — Gemini parses user query into {activity, province, region, age, style, gender}

2. search_camps() — SQL query with specialty codes + location cascade (city→region→province→global)

3. HARD GATE — if SQL codes matched but no local results (fallback fired), re-fetch limit=200 for semantic pool

4. BROAD-CODE REFINEMENT — sub-activities (ballet, guitar, acting) under broad parent codes (dance=22, music=37, theatre=59) get semantic scoring WITHIN SQL 
results instead of flat 0.9

5. semantic_score_camps() — embedding-001 cosine similarity (only for non-code-matched or broad-code activities)

6. Elbow detection — cuts results at largest score gap (min 8, max 25 results)

7. Low-confidence gate — unrecognized: 0.855 threshold, taxonomy: 0.80 threshold

8. score_and_rank() — weighted composite: semantic 50%, location 20%, age 15%, style 8%, gender 7%

9. generate_blurbs() — Gemini writes camp descriptions

## Three Mapping Layers (MUST stay in sync)

When adding any new activity, update ALL THREE:

- ACTIVITY_CODES_SQL (inside search_camps, ~line 783) — activity name → list of SQL specialty codes

- SEMANTIC_ONLY_ACTIVITIES (module level, ~line 653) — set of activities WITHOUT SQL codes

- Alias maps (three separate dicts that must all match):

  - _ACTIVITY_ALIASES inside search_camps()

  - _PROCESS_ALIASES inside process_query()

  - _ACTIVITY_SYNONYM_MAP at module level (used by _validate_filters)

If an activity has SQL codes, it must NOT be in SEMANTIC_ONLY_ACTIVITIES.

If an activity has no SQL codes, it MUST be in SEMANTIC_ONLY_ACTIVITIES.

## Decision Tree (process_query)

When processing a follow-up message, the order is:

1. SUGGESTION — AI suggested a new location, user affirmed → apply suggested location + keep last activity

2. REUSE — pure affirmative ("yes", "sure") → copy last_filters exactly

3. FRESH — 2+ new signals, activity changed, gender flip, or no prior context

4. MERGE — additive reply to AI question or refinement phrase

5. DEFAULT — treat as fresh with location inheritance

Affirmative detection is FUZZY — uses word-level matching, not just exact phrase list.

Location inheritance: follow-up queries inherit province/region from previous turn unless they specify their own.

## Key Constants and Thresholds

- LOW_CONFIDENCE_THRESHOLD = 0.855 (for unrecognized activities)

- TAXONOMY_CONFIDENCE_THRESHOLD = 0.80 (for recognized activities not found locally)

- BROAD_PARENT_CODES = {22: dance, 37: music, 59: theatre}

- Broad-code blend formula: 0.7 * semantic_score + 0.3 * 0.9

- Semantic-only pool limit: 200 (vs 20 for SQL-coded activities)

## Verification

After any change, always run:

python3 -c "import ast; ast.parse(open('streamlit_app.py').read()); print('OK')"

To run the app locally:

streamlit run streamlit_app.py

## Database

- MySQL: camp_directory.sessions_clean (session data with specialty codes)

- MySQL: camp_directory.camps_clean (camp profiles with descriptions)

- Specialty codes are integers (e.g., 22=dance, 29=hockey, 37=music)

- GTA aliases: vaughan→york region, mississauga→peel, markham→york region, etc.

## Known Gotchas

- embedding-001 compresses ALL scores into 0.82-0.91 band — fixed thresholds don't work, must use elbow detection

- _activity_originally_had_codes flag preserves recognition status through HARD GATE

- "sure, go ahead" must be caught by fuzzy affirmative detection (not just exact list match)

- The AI's suggested location must be parsed from the assistant's last message using regex

## Deployment
Hosted on Streamlit Community Cloud. To deploy changes:
1. `git add <files>` → `git commit` → `git push`
2. Streamlit Cloud auto-deploys on push to main — no manual step needed.

## Fix Tracking
All fixes are tracked in FIXES.md. Before making any change, read it.
After fixing a new issue, add it as the next numbered fix with the same format.
Always update the /audit command to include the new fix's verification check.
