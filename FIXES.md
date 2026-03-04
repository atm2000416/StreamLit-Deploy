# Search Pipeline Fixes

## Status: 13 verified fixes

### Fix 1: SEMANTIC_ONLY scoping
- **Problem**: SEMANTIC_ONLY_ACTIVITIES defined inside function, not accessible elsewhere
- **Solution**: Moved to module level (column 0)
- **Verify**: Line starts at col 0, single definition, 3+ references
- **Test query**: "basketball camps" (semantic-only activity)

### Fix 2: Synonym map stripping
- **Problem**: _validate_filters stripped normalized activities like 'animals'
- **Solution**: Added _ACTIVITY_SYNONYM_MAP, synonym_terms check
- **Verify**: synonym_terms used in _validate_filters
- **Test query**: "puppy camps in belleville"

### Fix 3: Low-confidence gate
- **Problem**: Low threshold bypassed taxonomy items
- **Solution**: Dual threshold — 0.80 taxonomy, 0.855 unknown
- **Verify**: Both branches present in code
- **Test query**: "xyzzy camps" vs "ballet camps in vaughan"

### Fix 4: SQL pool size
- **Problem**: Limit 20 too small for semantic-only
- **Solution**: 200 for semantic-only activities
- **Verify**: "200 if _activity_for_limit" in code
- **Test query**: "basketball camps in toronto"

### Fix 5: Umbrella categories
- **Problem**: "sports camps" returned nothing — no mapping
- **Solution**: Added sports, arts, computers, education, health, adventure
- **Verify**: All 6 in ACTIVITY_CODES_SQL, military/ropes/travel removed from SEMANTIC_ONLY
- **Test query**: "sports camps in toronto", "adventure camps"

### Fix 6: Location inheritance
- **Problem**: Follow-ups lost province/region
- **Solution**: Inherit from previous turn if new message lacks location
- **Verify**: "inherited location" in 2+ branches, _broad_provinces guard
- **Test query**: "puppy camps in belleville" → "what about basketball?"

### Fix 7: Ballet "not familiar" error
- **Problem**: Ballet has code 22, but after HARD GATE fallback, system said "not familiar with ballet"
- **Solution**: _activity_originally_had_codes flag preserves recognition
- **Verify**: Flag set and used in _act_is_taxonomy
- **Test query**: "ballet camps in vaughan"

### Fix 8: Two-pass merge
- **Problem**: HARD GATE re-fetch had no semantic scoring
- **Solution**: Re-fetch with activity=None, limit=200, then semantic score the pool
- **Verify**: _merge_filters, _merge_camps, activity=None re-fetch
- **Test query**: "ballet camps in vaughan" (code match but no local hit)

### Fix 9: Diagnostic follow-ups
- **Problem**: Generic "try a different province" when no results
- **Solution**: _diagnose_availability queries DB for where activity exists
- **Verify**: Function exists, diagnosis passed to 2+ call sites
- **Test query**: "ballet camps in vaughan" → should suggest Toronto

### Fix 10: Affirmative + location suggestion
- **Problem**: "sure, go ahead" to AI's suggestion re-ran same failing search
- **Solution**: Fuzzy affirmative detection + pre-tree SUGGESTION ACCEPTED path
- **Verify**: _affirm_words set, _suggestion_applied guard before decision tree
- **Test query**: "ballet in vaughan" → "sure, go ahead" (should search Toronto)

### Fix 11: Broad-code semantic refinement
- **Problem**: Ballet maps to code 22 (dance), hip-hop camps got 90% match
- **Solution**: BROAD_PARENT_CODES triggers semantic scoring within SQL results
- **Verify**: blend formula 0.7*semantic + 0.3*0.9, ballet/guitar/acting in check
- **Test query**: "ballet camps in toronto" (ballet should rank above hip-hop)

### Fix 12: Paddle-sport broad-code refinement
- **Problem**: "Sea kayaking" resolved to code 41 (all paddle sports), canoe tripping camps dominated results with flat 0.9 score
- **Solution**: Added 41 to BROAD_PARENT_CODES; added kayaking/sea kayaking/rowing/etc. to _ACTIVITY_CODES_CHECK; added 'sea kayaking' alias to all 3 maps
- **Verify**: 41 in BROAD_PARENT_CODES, 'kayaking'/'sea kayaking' in _ACTIVITY_CODES_CHECK, 'sea kayaking' in all 3 alias maps
- **Test query**: "sea kayaking camps" (sea kayaking camps should rank above canoe tripping)

---

### Fix 13: BROAD-CODE uses original user text for semantic scoring
- **Problem**: Gemini strips qualifiers (e.g. "sea kayaking" → "kayaking"), so BROAD-CODE REFINEMENT scored against the wrong term — canoe tripping ranked above sea kayaking
- **Solution**: Pass `user_text` instead of `activity_query` to `semantic_score_camps` in BROAD-CODE REFINEMENT
- **Verify**: `_broad_semantic_q = user_text if user_text else activity_query` present before semantic_score_camps call in BROAD-CODE block
- **Test query**: "sea kayaking camps for kids" (sea kayaking camps should rank above canoe tripping)

## Known Issues (not yet fixed)
<!-- Add new issues here as they're discovered -->
