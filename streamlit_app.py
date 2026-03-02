"""
Camp Discovery Chatbot - Production Version
Business Logic: Client-Only Member Camps with Verified URLs
Platform: Streamlit Cloud
Databases: Aiven MySQL (campdb, camp_directory, common_update)
Vector DB: Pinecone
AI: Google Gemini 2.5 Flash Lite
"""

import streamlit as st
import os
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus
from functools import lru_cache
import time

# ═════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════
@st.cache_resource
def get_config():
    """Load configuration from Streamlit secrets or environment variables"""
    try:
        return {
            "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"],
            "PINECONE_API_KEY": st.secrets["PINECONE_API_KEY"],
            "DB_HOST": st.secrets["DB_HOST"],
            "DB_PORT": st.secrets.get("DB_PORT", "10536"),
            "DB_USER": st.secrets["DB_USER"],
            "DB_PASS": st.secrets["DB_PASS"],
            "DB_CAMPDB": st.secrets.get("DB_CAMPDB", "campdb"),
            "DB_CAMP_DIR": st.secrets.get("DB_CAMP_DIRECTORY", "camp_directory"),
            "DB_COMMON": st.secrets.get("DB_COMMON_UPDATE", "common_update"),
            "INDEX_NAME": st.secrets.get("INDEX_NAME", "searching-doolie"),
            "INDEX_HOST": st.secrets["INDEX_HOST"],
            "NAMESPACE": st.secrets.get("NAMESPACE", "default"),
        }
    except Exception:
        return {k: os.getenv(k, "") for k in [
            "GEMINI_API_KEY", "PINECONE_API_KEY", "DB_HOST", "DB_PORT", 
            "DB_USER", "DB_PASS", "DB_CAMPDB", "DB_CAMP_DIR", "DB_COMMON",
            "INDEX_NAME", "INDEX_HOST", "NAMESPACE"
        ]}

def get_db_uri(config, db_name):
    """Generate MySQL connection URI"""
    return (
        f"mysql+pymysql://{config['DB_USER']}:{quote_plus(config['DB_PASS'])}"
        f"@{config['DB_HOST']}:{config['DB_PORT']}/{db_name}"
    )

# ═════════════════════════════════════════════
# CLIENT DATABASE CACHE
# Single source of truth - only paying members
# ═════════════════════════════════════════════
@st.cache_resource
def load_client_camps(config):
    """
    Load ALL paying client camps from database
    Generates verified URLs to camps.ca or ourkids.net
    Cached on app startup - runs once
    """
    from sqlalchemy import create_engine, text
    
    def generate_verified_url(camp_name, camp_id=None, source_site="camps.ca"):
        """Generate verified URL for camps.ca or ourkids.net"""
        slug = camp_name.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        slug = slug.strip('-')
        
        if source_site == "ourkids":
            full_url = f"https://www.ourkids.net/camp/{slug}"
            friendly = f"ourkids.net/camp/{slug}"
        else:
            full_url = f"https://www.camps.ca/camp/{slug}"
            friendly = f"camps.ca/camp/{slug}"
        
        return full_url, friendly
    
    client_camps = {}
    
    for db_name in [config["DB_CAMPDB"], config["DB_CAMP_DIR"], config["DB_COMMON"]]:
        try:
            engine = create_engine(get_db_uri(config, db_name), pool_pre_ping=True)
            with engine.connect() as conn:
                tables_result = conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in tables_result.fetchall()]
                
                for table in tables:
                    if 'camp' not in table.lower():
                        continue
                    
                    try:
                        cols_result = conn.execute(text(f"SHOW COLUMNS FROM `{table}`"))
                        cols = {row[0].lower(): row[0] for row in cols_result.fetchall()}
                        
                        name_col = cols.get('camp_name') or cols.get('name') or cols.get('title')
                        url_col = cols.get('prettyurl') or cols.get('filename') or cols.get('url') or cols.get('website')
                        location_col = cols.get('location') or cols.get('region') or cols.get('province')
                        type_col = cols.get('elistingtype') or cols.get('listingclass') or cols.get('type') or cols.get('category')
                        price_col = cols.get('price') or cols.get('cost') or cols.get('fee')
                        age_min_col = cols.get('age_min') or cols.get('min_age')
                        age_max_col = cols.get('age_max') or cols.get('max_age')
                        day_col = cols.get('listingclass') or cols.get('day_overnight') or cols.get('camp_style')
                        desc_col = cols.get('description') or cols.get('details')
                        id_col = cols.get('cid') or cols.get('id') or cols.get('camp_id')
                        
                        if not name_col:
                            continue
                        
                        select_cols = [f"`{name_col}`"]
                        col_map = {'name': 0}
                        idx = 1
                        
                        for key, col in [
                            ('url', url_col), ('location', location_col), ('type', type_col),
                            ('price', price_col), ('age_min', age_min_col), ('age_max', age_max_col),
                            ('day_overnight', day_col), ('description', desc_col), ('id', id_col)
                        ]:
                            if col:
                                select_cols.append(f"`{col}`")
                                col_map[key] = idx
                                idx += 1
                        
                        query = f"SELECT {', '.join(select_cols)} FROM `{table}` WHERE `{name_col}` IS NOT NULL LIMIT 10000"
                        
                        camps_result = conn.execute(text(query))
                        for row in camps_result.fetchall():
                            name = row[col_map['name']]
                            if not name:
                                continue
                            
                            key = name.lower().strip()
                            camp_id = row[col_map.get('id', 0)] if 'id' in col_map else None
                            existing_url = row[col_map.get('url', 0)] if 'url' in col_map and row[col_map['url']] else None
                            
                            source_site = "ourkids" if existing_url and 'ourkids.net' in existing_url else "camps.ca"
                            full_url, friendly_url = generate_verified_url(name, camp_id, source_site)
                            
                            camp = {
                                "name": name,
                                "url": full_url,
                                "friendly_url": friendly_url,
                                "location": row[col_map.get('location', 0)] if 'location' in col_map else None,
                                "type": row[col_map.get('type', 0)] if 'type' in col_map else None,
                                "price": row[col_map.get('price', 0)] if 'price' in col_map else None,
                                "age_min": row[col_map.get('age_min', 0)] if 'age_min' in col_map else None,
                                "age_max": row[col_map.get('age_max', 0)] if 'age_max' in col_map else None,
                                "day_overnight": row[col_map.get('day_overnight', 0)] if 'day_overnight' in col_map else None,
                                "description": row[col_map.get('description', 0)] if 'description' in col_map else None,
                                "id": camp_id,
                                "database": db_name,
                                "table": table,
                                "source_site": source_site
                            }
                            
                            client_camps[key] = camp
                    
                    except Exception:
                        continue
        
        except Exception:
            continue
    
    return client_camps

# ═════════════════════════════════════════════
# GEMINI API
# ═════════════════════════════════════════════
MODEL = "gemini-2.5-flash-lite"
BASE = "https://generativelanguage.googleapis.com/v1beta"

def call_gemini(system_prompt, user_prompt, api_key, max_tokens=512):
    """Call Gemini API with error handling"""
    import requests
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens}
    }
    try:
        resp = requests.post(
            f"{BASE}/models/{MODEL}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload, timeout=30
        )
        if resp.status_code >= 400:
            return ""
        data = resp.json()
        if data.get("candidates"):
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
    except Exception:
        pass
    return ""


# ═════════════════════════════════════════════
# CORE RAG PIPELINE
# ═════════════════════════════════════════════

def _validate_filters(filters, user_text):
    """Strip filter values not evidenced in the raw message text.
    Prevents Gemini from bleeding prior context into a fresh query.
    """
    import re
    text = user_text.lower()
    validated = dict(filters)

    # Negation check — "not looking for X", "no longer want X", "forget X"
    # If the activity appears in a negation context, strip it
    negation_patterns = [
        r"not looking for (\w+)",
        r"no longer (?:want|looking for|need) (\w+)",
        r"forget (?:the )?(\w+)",
        r"don't want (\w+)",
        r"not (?:interested in|after) (\w+)",
    ]
    negated_words = set()
    for pat in negation_patterns:
        for m in re.finditer(pat, text):
            negated_words.add(m.group(1))

    # Activity: must have a keyword hint in the message AND not be negated
    activity = (filters.get('activity') or '').lower()
    if activity:
        activity_words = set(re.findall(r'\w+', activity))
        is_negated = bool(activity_words & negated_words)
        if is_negated or not any(w in text for w in activity_words if len(w) > 3):
            validated['activity'] = None

    # Gender: must have an explicit gender word
    gender_words = ['girl', 'girls', 'boy', 'boys', 'female', 'male',
                    'daughter', 'son', 'all-girls', 'all-boys']
    if filters.get('gender') and not any(w in text for w in gender_words):
        validated['gender'] = None

    # Style: must have day/overnight keyword
    style_words = ['overnight', 'day camp', 'residential', 'sleepover']
    if filters.get('style') and not any(w in text for w in style_words):
        validated['style'] = None

    # Age: must have a number or age word
    if filters.get('age') and not re.search(r'\d+\s*(?:yr|year|yrs)|teen|toddler', text):
        validated['age'] = None

    return validated



# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH ENGINE
# Replaces the activity_codes keyword lookup with embedding-based similarity.
# Embeddings are pre-computed and stored in camp_directory.camp_embeddings.
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: list, b: list) -> float:
    """Pure Python cosine similarity — fast enough for <300 camps."""
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def get_query_embedding(query: str, api_key: str) -> list | None:
    """Embed a search query using Gemini embedding-001."""
    import requests as _req
    # Discover available embedding model (cached implicitly via repeated calls)
    try:
        _list = _req.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10
        )
        _embed_models = [
            m["name"] for m in (_list.json().get("models", []) if _list.ok else [])
            if "embedContent" in m.get("supportedGenerationMethods", [])
        ]
        _model = _embed_models[0] if _embed_models else "models/embedding-001"
    except Exception:
        _model = "models/embedding-001"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{_model}:embedContent?key={api_key}"
    )
    try:
        resp = _req.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "model": _model,
                "content": {"parts": [{"text": query}]},
                "taskType": "RETRIEVAL_QUERY",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_camp_embeddings(_engine) -> dict:
    """
    Load all camp embeddings from DB into memory.
    Cached for 1 hour — refreshes automatically after new camps are embedded.
    Returns: {cid: [float, ...], ...}
    """
    from sqlalchemy import text as _text
    try:
        with _engine.connect() as conn:
            rows = conn.execute(_text(
                "SELECT cid, embedding FROM camp_directory.camp_embeddings"
            )).fetchall()
        result = {}
        for cid, emb_json in rows:
            import json as _json
            result[int(cid)] = _json.loads(emb_json) if isinstance(emb_json, str) else emb_json
        return result
    except Exception:
        return {}


def semantic_score_camps(camps: list, query: str, api_key: str, engine) -> list:
    """
    Score each camp by semantic similarity to the query.
    Adds '_semantic_score' (0.0–1.0) to each camp dict.
    Falls back to 0.5 for camps with no embedding (graceful degradation).
    """
    if not query or not query.strip():
        for c in camps:
            c['_semantic_score'] = 0.5
        return camps

    # Embed the query
    query_vec = get_query_embedding(query, api_key)
    if query_vec is None:
        # Embedding API failed — fall back to neutral scores
        for c in camps:
            c['_semantic_score'] = 0.5
        return camps

    # Load stored embeddings
    embeddings = load_camp_embeddings(engine)

    for c in camps:
        cid       = int(c.get('cid') or 0)
        camp_vec  = embeddings.get(cid)
        if camp_vec:
            c['_semantic_score'] = round(cosine_similarity(query_vec, camp_vec), 4)
        else:
            c['_semantic_score'] = 0.3   # no embedding → lower than average, not zero
    return camps


def embeddings_are_ready(engine) -> bool:
    """Check if camp_embeddings table exists and has data."""
    from sqlalchemy import text as _text
    try:
        with engine.connect() as conn:
            count = conn.execute(_text(
                "SELECT COUNT(*) FROM camp_directory.camp_embeddings"
            )).scalar()
        return (count or 0) > 0
    except Exception:
        return False

def extract_filters(user_text, api_key):
    """Use Gemini to extract structured search filters from natural language"""
    system = """You are a camp search assistant. Extract search filters from the user message.
Return ONLY a valid JSON object with these fields (use null if not mentioned):
{
  "province": "full province name or null",
  "region": "city or region name or null",
  "activity": "main activity or dietary/special need or null",
  "age": integer or null,
  "max_cost": integer or null,
  "style": "day" or "overnight" or null,
  "name": "user first name if mentioned or null",
  "gender": "girls" or "boys" or null
}
Province must be one of: British Columbia, Alberta, Ontario, Quebec, Manitoba, Saskatchewan, Nova Scotia, New Brunswick, Prince Edward Island, Newfoundland and Labrador.
For cities, set region to the city name and infer the province.
For dietary or special needs, put the need in activity field.
Examples:
- "etobicoke" → province: "Ontario", region: "Toronto"
- "nepean" → province: "Ontario", region: "Ottawa Region"
- "mississauga" → province: "Ontario", region: "Halton - Peel"
- "debate camps" → activity: "debate"
- "hockey" → activity: "hockey"
- "cheer" or "cheerleading" → activity: "cheer"
- "fashion" or "fashion design" → activity: "fashion"
- "gluten free" → activity: "gluten"
- "vegetarian" or "kosher" or "halal" → activity: "vegetarian" (dietary keyword)
- "special needs" → activity: "special needs"
- "autism" → activity: "autism"
- "under $500" → max_cost: 500
- "10 year old" → age: 10
- "teens" → age: 15
- "my name is Sarah" → name: "Sarah"
- "all-girls" or "girls only" or "girls camp" or "for girls" → gender: "girls"
- "all-boys" or "boys only" or "boys camp" or "for boys" → gender: "boys"
- no gender mention → gender: null
IMPORTANT: Each query is independent. Do not carry over context from previous queries."""

    result = call_gemini(system, user_text, api_key, max_tokens=200)
    import json, re
    try:
        # Strip markdown code fences if present
        clean = re.sub(r'```json|```', '', result).strip()
        parsed = json.loads(clean)
        # Strip non-province values — 'Canada' etc. are not DB provinces
        prov = (parsed.get('province') or '').strip()
        if prov.lower() in ('canada', 'all', 'any', 'nationwide', 'national'):
            parsed['province'] = None
        return parsed
    except:
        return {}


def search_camps(filters, config, limit=20, named_camp=None, engine=None):
    """Query sessions_clean joined to camps_clean for session-level precision"""
    from sqlalchemy import create_engine, text

    # Region mapping for common cities/districts
    # region_map removed — city/region resolved directly from DB (sessions_clean.city)
    # Minimal fallback for province inference when city not in DB
    region_map = {}
    # city_province_map removed — province inferred from DB via city lookup
    # Kept as minimal fallback for province-only queries
    city_province_map = {
        'montreal': 'Quebec', 'laval': 'Quebec', 'longueuil': 'Quebec',
        'gatineau': 'Quebec', 'sherbrooke': 'Quebec', 'quebec city': 'Quebec',
        'vancouver': 'British Columbia', 'victoria': 'British Columbia',
        'calgary': 'Alberta', 'edmonton': 'Alberta',
        'winnipeg': 'Manitoba', 'ottawa': 'Ontario', 'toronto': 'Ontario',
        'halifax': 'Nova Scotia', 'winnipeg': 'Manitoba',
        'saskatoon': 'Saskatchewan', 'regina': 'Saskatchewan',
    }

    province = filters.get('province')
    region   = filters.get('region', '')
    activity = filters.get('activity', '')
    age      = filters.get('age')
    max_cost = filters.get('max_cost')
    style    = filters.get('style')
    gender   = filters.get('gender')  # 'girls', 'boys', or None

    # Resolve city/region using DB as source of truth
    # city field in sessions_clean comes from extra_locations.city (ground truth)
    region_lower = (region or '').lower().strip()

    # Infer province from city if not explicitly provided (minimal fallback map)
    if not province and region_lower in city_province_map:
        province = city_province_map[region_lower]

    # Treat region input as EITHER a city name OR a region name — DB handles both
    # region_map is now empty; resolved_region = raw region string from filters
    resolved_region = region_lower  # pass raw to DB city search

    conditions = ["sc.status = 1", "sc.province != 'Unknown'", "sc.province != 'Virtual Program'", "sc.is_virtual = 0"]
    params = {}

    if province:
        conditions.append("sc.province = :province")
        params['province'] = province

    if resolved_region:
        # Search city column first (exact source of truth), then region name
        conditions.append("(sc.city LIKE :city OR sc.region LIKE :region)")
        params['city']   = f"%{resolved_region}%"
        params['region'] = f"%{resolved_region}%"

    # Activity: specialty codes go back into SQL for known activities (exact match).
    # Unknown activities (no codes) → SQL returns all, semantic ranking applied after.
    # ── Specialty code SQL filter ────────────────────────────────────────────
    # PHILOSOPHY: Only use SQL specialty codes when the DB code is SPECIFIC
    # enough to be meaningful. Broad umbrella codes (188, 268, 9) excluded.
    #
    # For taxonomy subcategories not listed here — financial literacy, pottery,
    # entrepreneurship, filmmaking, martial arts, basketball, etc. — semantic
    # search handles them. This is intentional and more accurate than forcing
    # them into a coarse parent code.
    #
    # Codes verified against DB (sessions_clean.specialty/specialty2).
    # Taxonomy source: camps.ca onboarding PDF (225 categories, 6 groups).
    ACTIVITY_CODES_SQL = {

        # ── SPORTS ──────────────────────────────────────────────────────────
        # Has dedicated codes; most Ball Sports subcategories fall to semantic
        'hockey':               [29],
        'ice hockey':           [29],
        'ball hockey':          [29],
        'figure skating':       [29],
        'ice skating':          [29],
        'soccer':               [54],
        'tennis':               [66],
        'golf':                 [26],
        'disc golf':            [26],
        'volleyball':           [63],
        'beach volleyball':     [63],
        'swimming':             [56],
        'swim':                 [56],
        'sailing':              [49],
        'board sailing':        [49],
        'marine skills':        [49],
        'canoeing':             [41],
        'canoe':                [41],
        'kayaking':             [41],
        'kayak':                [41],
        'rowing':               [41],
        'paddleboard':          [41],
        'stand up paddle':      [41],
        'waterskiing':          [41],
        'wakeboarding':         [41],
        'whitewater':           [41],
        'equestrian':           [30],
        'horseback riding':     [30],
        'horseback':            [30],
        'horse':                [30],
        'riding':               [30],
        'dance':                [22],
        'ballet':               [22],
        'hip hop':              [22],
        'jazz dance':           [22],
        'breakdancing':         [22],
        'acro dance':           [22],
        'tap dance':            [22],
        'cheer':                [164],
        'cheerleading':         [164],
        'chess':                [278],
        # Semantic handles: basketball, lacrosse, archery, gymnastics, martial arts,
        # fencing, rock climbing, trampoline, cycling, parkour, ultimate frisbee,
        # badminton, cricket, rugby, flag football, skateboarding, skiing, snowboarding,
        # surfing, fishing, paintball, taekwondo, karate, ping pong, zip line, etc.

        # ── COMPUTERS & TECH ────────────────────────────────────────────────
        'coding':               [18, 68, 180, 266, 159],
        'programming':          [18, 68, 180, 266, 159],
        'python':               [18, 180],
        'scratch':              [18, 180],
        'java':                 [18, 180],
        'robotics':             [67, 160],
        'lego robotics':        [67],
        'lego':                 [67],
        'engineering':          [160, 67],
        'ai':                   [302],
        'artificial intelligence': [302],
        'machine learning':     [302],
        'minecraft':            [68, 18],
        'roblox':               [68, 18],
        'game design':          [68],
        'video game design':    [68],
        'video game':           [68],
        'animation':            [180, 18],
        'stem':                 [159, 268, 67, 160, 180, 18],
        'steam':                [159, 268, 67, 160, 180, 18],
        'math':                 [129, 20],
        'mathematics':          [129, 20],
        'aerospace':            [50],
        'space':                [50],
        'aviation':             [50],
        'technology':           [180, 18],
        # Semantic handles: 3D printing, drone, VR, web design, Arduino, Raspberry Pi,
        # mechatronics, micro:bit, gaming (general)

        # ── ARTS ────────────────────────────────────────────────────────────
        'music':                [37],
        'guitar':               [37],
        'piano':                [37],
        'drums':                [37],
        'percussion':           [37],
        'singing':              [37],
        'vocal':                [37],
        'songwriting':          [37],
        'djing':                [37],
        'theatre':              [59],
        'theater':              [59],
        'drama':                [59],
        'musical theatre':      [59, 37],
        'acting':               [59],
        'performing arts':      [59, 37, 22],
        'photography':          [69],
        'filmmaking':           [69],
        'videography':          [69],
        'cooking':              [133],
        'baking':               [133],
        'culinary':             [133],
        'chef':                 [133],
        'fashion':              [71, 172],
        'fashion design':       [71, 172],
        'sewing':               [172],
        'makeup':               [172],
        # Semantic handles: pottery, ceramics, drawing, painting, circus, comedy,
        # knitting, woodworking, magic, puppetry, storytelling, podcasting,
        # YouTube/vlogging, sculpture, cartooning, comic art, mixed media

        # ── EDUCATION ───────────────────────────────────────────────────────
        'debate':               [362, 97],
        'public speaking':      [362, 97],
        'speech':               [362, 97],
        'writing':              [362],
        'creative writing':     [362],
        'essay writing':        [362],
        'academic':             [32, 97, 20, 314],
        'tutoring':             [32, 97],
        'french':               [314],
        'french immersion':     [314],
        'language':             [314],
        'esl':                  [314],
        # Semantic handles: financial literacy, entrepreneurship, journalism,
        # makerspace, board games, reading, forensic science, marine biology,
        # architecture, meteorology, zoology, test prep, urban exploration

        # ── OUTDOOR / TRADITIONAL ────────────────────────────────────────────
        'traditional':          [181, 24, 58, 265],
        'wilderness':           [181, 24, 41],
        'outdoor':              [181, 24, 41, 49, 58, 265],
        'adventure':            [181, 41],
        'nature':               [24, 181],
        'hiking':               [24, 181],
        'survival skills':      [24, 181],
        'wilderness skills':    [24, 181],
        'canoe tripping':       [41],

        # ── LEADERSHIP / HEALTH / WELLNESS ──────────────────────────────────
        'leadership':           [88, 33],
        'wellness':             [91],
        'yoga':                 [91],
        'mindfulness':          [91],
        'meditation':           [91],
        'fitness':              [91],
        # Semantic handles: nutrition, pilates, weight loss, first aid,
        # strength & conditioning, behavioral therapy, bronze cross

        # ── SPECIAL NEEDS ────────────────────────────────────────────────────
        'special needs':        [252],
        'autism':               [252],
        'learning disability':  [252],
        'behavioral therapy':   [252],
    }
    # Generic umbrella codes — too broad to mean a specific activity
    GENERIC_CODES_SQL = {188, 268, 9, 10, 79, 33}
    # Location/region codes leaked into specialty column — exclude from all activity filters
    LOCATION_CODES_SQL = {288, 150, 347, 170, 81, 256, 130, 11, 135, 176, 287, 315, 317,
                          318, 342, 347}
    _activity_has_codes = False

    if activity:
        act_lower_sql = activity.lower().strip()
        sql_codes = ACTIVITY_CODES_SQL.get(act_lower_sql, [])
        sql_codes = [c for c in sql_codes
                     if c not in GENERIC_CODES_SQL and c not in LOCATION_CODES_SQL] or sql_codes
        if sql_codes:
            code_list = ','.join(str(c) for c in sql_codes)
            conditions.append(f'(sc.specialty IN ({code_list}) OR sc.specialty2 IN ({code_list}))')
            _activity_has_codes = True
        else:
            # Unknown activity — no SQL filter, semantic ranking handles it post-query
            pass

    if age:
        conditions.append("sc.age_from <= :age AND sc.age_to >= :age")
        params['age'] = age

    if max_cost:
        conditions.append("(sc.cost_from <= :cost OR sc.cost_from = 0)")
        params['cost'] = max_cost

    if style:
        conditions.append("sc.camp_style = :style")
        params['style'] = style

    if gender == 'girls':
        conditions.append("sc.gender = 2")
    elif gender == 'boys':
        conditions.append("sc.gender = 3")

    # If a specific camp was named in a correction, force-include it as an OR
    named_camp_condition = ""
    if named_camp:
        params['named_camp'] = f"%{named_camp}%"
        named_camp_condition = f" OR sc.camp_name LIKE :named_camp"

    where = " AND ".join(conditions)
    # Wrap entire WHERE in parens then OR with named_camp so it always appears
    if named_camp_condition:
        where = f"({where}){named_camp_condition}"
    sql = f"""SELECT
            sc.cid, sc.camp_name, sc.province,
            COALESCE(MIN(CASE WHEN sc.region != 'Virtual Program' THEN sc.region END), MIN(sc.region)) AS region,
            sc.camp_style, sc.listing_tier, sc.camp_url,
            MIN(sc.session_url)             AS session_url,
            MIN(sc.age_from)                AS age_min,
            MAX(sc.age_to)                  AS age_max,
            MIN(NULLIF(sc.cost_from,0))     AS cost_min,
            MAX(NULLIF(sc.cost_to,0))       AS cost_max,
            COUNT(DISTINCT sc.session_id)   AS session_count,
            GROUP_CONCAT(DISTINCT sc.specialty_label
                ORDER BY sc.specialty_label SEPARATOR ', ') AS activities,
            GROUP_CONCAT(
                DISTINCT CONCAT(
                    '#', sc.session_id, ': ', sc.class_name,
                    ' (ages ', sc.age_from, '-', sc.age_to, ')',
                    COALESCE(
                        IF(NULLIF(TRIM(s.mini_description),'') IS NOT NULL,
                           CONCAT(' — ', LEFT(TRIM(s.mini_description), 120)), NULL),
                        IF(NULLIF(TRIM(s.description),'') IS NOT NULL,
                           CONCAT(' — ', LEFT(TRIM(s.description), 120)), NULL),
                        ''
                    )
                )
                ORDER BY sc.listing_tier SEPARATOR ' ||| '
            )                               AS matching_programs,
            cc.description
        FROM camp_directory.sessions_clean sc
        JOIN camp_directory.camps_clean cc
            ON cc.cid = sc.cid AND cc.province = sc.province
        LEFT JOIN camp_directory.sessions s
            ON s.id = sc.session_id
        WHERE {where}
        GROUP BY sc.cid, sc.camp_name, sc.province, sc.camp_style,
                 sc.listing_tier, sc.camp_url, cc.description
        ORDER BY FIELD(sc.listing_tier, 'gold', 'silver', 'bronze'), sc.camp_name
        LIMIT {limit}"""

    try:
        engine = create_engine(get_db_uri(config, config["DB_CAMP_DIR"]), pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows   = result.fetchall()
            cols   = list(result.keys())

            if rows:
                return [dict(zip(cols, row)) for row in rows], province, resolved_region, None, _activity_has_codes

            # Fallbacks 1 & 2 removed — activity is no longer a SQL filter.
            # Semantic search handles activity ranking after SQL returns candidates.
            # If structural filters (province/city/age/style/gender) yield no results,
            # proceed to province-only and global fallbacks below.

            # Fallback 3: province only — drop both activity and region
            if province:
                r3 = conn.execute(text(
                    "SELECT sc.cid, sc.camp_name, sc.province, MIN(sc.region) AS region, "
                    "sc.camp_style, sc.listing_tier, sc.camp_url, "
                    "MIN(sc.session_url) AS session_url, "
                    "MIN(sc.age_from) AS age_min, MAX(sc.age_to) AS age_max, "
                    "MIN(NULLIF(sc.cost_from,0)) AS cost_min, MAX(NULLIF(sc.cost_to,0)) AS cost_max, "
                    "COUNT(DISTINCT sc.session_id) AS session_count, "
                    "GROUP_CONCAT(DISTINCT sc.specialty_label ORDER BY sc.specialty_label SEPARATOR ', ') AS activities, "
                    "GROUP_CONCAT(DISTINCT CONCAT(sc.session_id,':',sc.class_name,' (ages ',sc.age_from,'-',sc.age_to,')') ORDER BY sc.listing_tier SEPARATOR ' | ') AS matching_programs, "
                    "cc.description "
                    "FROM camp_directory.sessions_clean sc "
                    "JOIN camp_directory.camps_clean cc ON cc.cid = sc.cid AND cc.province = sc.province "
                    "WHERE sc.status=1 AND sc.province=:p AND sc.province != 'Virtual Program' AND sc.is_virtual=0 "
                    "GROUP BY sc.cid, sc.camp_name, sc.province, sc.camp_style, sc.listing_tier, sc.camp_url, cc.description "
                    "ORDER BY FIELD(sc.listing_tier,'gold','silver','bronze') LIMIT :lim"
                ), {"p": province, "lim": limit})
                rows3 = r3.fetchall()
                if rows3:
                    return [dict(zip(list(r3.keys()), row)) for row in rows3], province, resolved_region, 'province_only', _activity_has_codes

            # Fallback 4: top camps overall
            r4 = conn.execute(text(
                "SELECT sc.cid, sc.camp_name, sc.province, MIN(sc.region) AS region, "
                "sc.camp_style, sc.listing_tier, sc.camp_url, "
                "MIN(sc.session_url) AS session_url, "
                "MIN(sc.age_from) AS age_min, MAX(sc.age_to) AS age_max, "
                "MIN(NULLIF(sc.cost_from,0)) AS cost_min, MAX(NULLIF(sc.cost_to,0)) AS cost_max, "
                "COUNT(DISTINCT sc.session_id) AS session_count, "
                "GROUP_CONCAT(DISTINCT sc.specialty_label ORDER BY sc.specialty_label SEPARATOR ', ') AS activities, "
                "GROUP_CONCAT(DISTINCT CONCAT(sc.session_id,':',sc.class_name,' (ages ',sc.age_from,'-',sc.age_to,')') ORDER BY sc.listing_tier SEPARATOR ' | ') AS matching_programs, "
                "cc.description "
                "FROM camp_directory.sessions_clean sc "
                "JOIN camp_directory.camps_clean cc ON cc.cid = sc.cid AND cc.province = sc.province "
                "WHERE sc.status=1 AND sc.province != 'Unknown' AND sc.province != 'Virtual Program' AND sc.is_virtual=0 "
                "GROUP BY sc.cid, sc.camp_name, sc.province, sc.camp_style, sc.listing_tier, sc.camp_url, cc.description "
                "ORDER BY FIELD(sc.listing_tier,'gold','silver','bronze') LIMIT :lim"
            ), {"lim": limit})
            rows4 = r4.fetchall()
            return [dict(zip(list(r4.keys()), row)) for row in rows4], province, resolved_region, 'no_match', _activity_has_codes

    except Exception as e:
        return [], province, resolved_region, f"error: {str(e)[:200]}", False


def clean_activities(activities_str):
    """Remove duplicate activity labels e.g. 'Basketball, Basketball' → 'Basketball'"""
    if not activities_str:
        return ''
    seen = set()
    result = []
    for a in activities_str.split(', '):
        a = a.strip()
        if a and a not in seen:
            seen.add(a)
            result.append(a)
    return ', '.join(result)

def dedupe_camps(camps):
    """Deduplicate camps by cid, keeping first (best-tier) occurrence."""
    seen, deduped = set(), []
    for c in camps:
        cid = c.get('cid')
        if cid not in seen:
            seen.add(cid)
            deduped.append(c)
    return deduped



def score_and_rank(camps, filters, fallback):
    """
    Hybrid relevancy scorer combining:
      1. Semantic similarity (_semantic_score set by semantic_score_camps())
      2. Structural signals: location, age, style, gender

    Weights (rescaled to active filters only):
      Semantic (activity/topic) : 0.50
      Location                  : 0.20
      Age                       : 0.15
      Style (day/overnight)     : 0.08
      Gender                    : 0.07

    Sort: combined_score DESC, then tier ASC (gold > silver > bronze)
    """
    TIER_RANK = {'gold': 1, 'silver': 2, 'bronze': 3}

    searched_age      = filters.get('age')
    searched_style    = (filters.get('style') or '').lower().strip()
    searched_gender   = (filters.get('gender') or '').lower().strip()
    searched_region   = (filters.get('region') or '').lower().strip()
    searched_province = (filters.get('province') or '').lower().strip()
    searched_activity = (filters.get('activity') or '').lower().strip()

    W = {}
    if searched_activity:                     W['semantic']  = 0.50
    if searched_region or searched_province:  W['location']  = 0.20
    if searched_age:                          W['age']       = 0.15
    if searched_style:                        W['style']     = 0.08
    if searched_gender:                       W['gender']    = 0.07

    total_w = sum(W.values()) or 1.0
    norm    = {k: v / total_w for k, v in W.items()}

    def score_camp(c):
        s = 0.0

        # Semantic — cosine similarity from embedding comparison
        if 'semantic' in norm:
            s += norm['semantic'] * c.get('_semantic_score', 0.5)

        # Location — city (ground truth) > region > province
        if 'location' in norm:
            camp_city     = (c.get('city') or '').lower()
            camp_region   = (c.get('region') or '').lower()
            camp_province = (c.get('province') or '').lower()
            if searched_region and searched_region in camp_city:
                s += norm['location']
            elif searched_region and searched_region in camp_region:
                s += norm['location'] * 0.85
            elif searched_province and searched_province in camp_province:
                s += norm['location'] * 0.60
            elif fallback in ('province_only', 'no_match'):
                s += norm['location'] * 0.20

        # Age — within range, with tightness bonus
        if 'age' in norm:
            age_min = c.get('age_min')
            age_max = c.get('age_max')
            if age_min is not None and age_max is not None:
                if age_min <= searched_age <= age_max:
                    tightness = max(0.0, 1.0 - (max(age_max - age_min, 1) - 1) / 18)
                    s += norm['age'] * (0.7 + 0.3 * tightness)

        # Style
        if 'style' in norm:
            if searched_style == (c.get('camp_style') or '').lower():
                s += norm['style']

        # Gender
        if 'gender' in norm:
            cg = c.get('gender')
            if searched_gender == 'girls' and cg == 2:
                s += norm['gender']
            elif searched_gender == 'boys' and cg == 3:
                s += norm['gender']

        # Tightness bonus when age not searched
        if 'age' not in norm:
            age_min = c.get('age_min')
            age_max = c.get('age_max')
            if age_min is not None and age_max is not None:
                tightness = max(0.0, 1.0 - (max(age_max - age_min, 1) - 1) / 18)
                s += 0.03 * tightness

        return round(s, 4)

    for c in camps:
        c['_relevancy'] = score_camp(c)

    camps.sort(key=lambda c: (
        -c['_relevancy'],
        TIER_RANK.get(c.get('listing_tier'), 4)
    ))
    return camps


def build_camp_url(c):
    """Return session-level URL when exactly 1 session matched, else camp URL."""
    session_count = int(c.get('session_count') or 0)
    session_url   = c.get('session_url') or ''
    camp_url      = c.get('camp_url') or ''
    return session_url if (session_count == 1 and session_url) else (camp_url or session_url)


def generate_blurbs(deduped, user_text, api_key):
    """Single Gemini call: given N camps, return N one-sentence Why-it-fits blurbs."""
    camp_snippets = []
    for i, c in enumerate(deduped, 1):
        name     = c.get('camp_name', '')
        programs = (c.get('matching_programs') or '').strip()
        desc     = (c.get('description') or '')[:120]
        camp_snippets.append(
            f"CAMP {i}: {name}\n"
            f"Programs: {programs[:300] if programs else 'N/A'}\n"
            f"Description: {desc}"
        )

    system = (
        "You are a warm, knowledgeable Canadian camp consultant. "
        "For each numbered camp below, write exactly ONE sentence (max 25 words) explaining "
        "what makes this camp a great fit. Draw from Programs text first, then Description. "
        "Be specific — mention the actual program name or a standout detail. "
        "CRITICAL: Always stay positive. Never say what a camp does NOT offer. "
        "Never use words like 'not', 'doesn't', 'no hockey', 'lacks', 'only', 'just'. "
        "Focus on what the camp DOES have that makes it worth considering. "
        "No corporate language. Sound like a trusted friend.\n\n"
        "Reply with ONLY this format — one line per camp, nothing else:\n"
        "1: <blurb>\n2: <blurb>\n..."
    )
    user_prompt = f"User searched for: {user_text}\n\n" + "\n\n".join(camp_snippets)

    raw = call_gemini(system, user_prompt, api_key, max_tokens=1500)

    # Parse "1: blurb" lines into a dict
    blurbs = {}
    if raw:
        import re
        for m in re.finditer(r'^(\d+):\s*(.+)$', raw, re.MULTILINE):
            blurbs[int(m.group(1))] = m.group(2).strip()
    return blurbs


def format_camp_context(camps):
    """Legacy — kept for dietary/fallback paths that still pass context to Gemini."""
    deduped = dedupe_camps(camps)
    lines = []
    for c in deduped:
        name      = c.get('camp_name', '') or ''
        url       = build_camp_url(c)
        region    = c.get('region', '')
        province  = c.get('province', '')
        age_min   = c.get('age_min', '')
        age_max   = c.get('age_max', '')
        cost_min  = c.get('cost_min', '')
        cost_max  = c.get('cost_max', '')
        style     = 'Day Camp' if c.get('camp_style') == 'day' else 'Overnight Camp'
        matching  = (c.get('matching_programs') or '').strip()
        desc      = (c.get('description') or '')[:120]
        age_str   = f"Ages {age_min}-{age_max}" if age_min and age_max else "Ages vary"
        cost_str  = f"${cost_min:,}-${cost_max:,}/week" if cost_min and cost_max else "Contact for pricing"
        lines.append(
            f"CAMP: {name} | {region}, {province} | {style} | {age_str} | {cost_str}\n"
            f"URL: {url}\nPrograms: {matching[:200]}\nDescription: {desc}"
        )
    return "\n---\n".join(lines)


def render_results(deduped, blurbs, user_text, filters, fallback, province, region):
    """Python renders the guaranteed-complete camp list — no Gemini dropping camps."""
    if not deduped:
        return None   # caller handles empty case

    # Opening line — only label filters that were explicitly in THIS search
    style_label  = filters.get('style') or ''
    gender_label = filters.get('gender') or ''
    age_val      = filters.get('age')
    act_label    = filters.get('activity') or ''

    descriptors = []
    if gender_label == 'girls': descriptors.append('all-girls')
    elif gender_label == 'boys': descriptors.append('all-boys')
    if style_label:  descriptors.append(style_label)
    if act_label:    descriptors.append(act_label)
    descriptors.append('camps')
    camp_type_str = ' '.join(d for d in descriptors if d)

    loc_str = region or province or 'Canada'
    age_str = f' for age {age_val}' if age_val else ''
    intro   = f"Here are {len(deduped)} {camp_type_str} in {loc_str}{age_str}:\n\n"

    if fallback and fallback != 'exact':
        fallback_notes = {
            'no_activity_in_region':   f"*Showing the best {act_label} camps available near {region or province}.*\n\n",
            'no_activity_in_province': f"*Showing the best {act_label} camps available in {province}.*\n\n",
            'province_only':           f"*Showing camps across {province} — you may want to filter by city.*\n\n",
            'no_match':                "*Showing the closest available matches.*\n\n",
        }
        intro += fallback_notes.get(fallback, '')

    # Build each camp block — Python guarantees all N appear
    blocks = []
    for i, c in enumerate(deduped, 1):
        name      = c.get('camp_name', '') or ''
        url       = build_camp_url(c)
        region_c  = c.get('region', '')
        province_c= c.get('province', '')
        age_min   = c.get('age_min', '')
        age_max   = c.get('age_max', '')
        cost_min  = c.get('cost_min', '')
        cost_max  = c.get('cost_max', '')
        camp_style= 'Overnight Camp' if c.get('camp_style') == 'overnight' else 'Day Camp'

        age_display  = f"Ages {age_min}–{age_max}" if age_min and age_max else "Ages vary"
        cost_display = f"${cost_min:,}–${cost_max:,}/week" if cost_min and cost_max else "Contact for pricing"

        blurb = blurbs.get(i, '')
        why_line = f"   * **Why it fits:** {blurb}" if blurb else ""

        block = (
            f"* **[{name}]({url})**\n"
            f"   * **Location:** {region_c}, {province_c}\n"
        )
        if why_line:
            block += why_line + "\n"
        block += (
            f"   * **Ages:** {age_display} | **Cost:** {cost_display}\n"
            f"   * **Type:** {camp_style}"
        )
        blocks.append(block)

    return intro + "\n".join(blocks)


def process_query(user_text, config, client_camps, chat_history=None, last_filters=None):
    """Main RAG pipeline — Gemini extracts filters, SQL fetches camps, Gemini writes response"""
    import time
    start = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Classify this message as SAME_CHILD, DIFFERENT_CHILD, or REFINE
    # ─────────────────────────────────────────────────────────────────────────
    text_lower_check = user_text.lower().strip()

    # Get last assistant message (used to detect if AI asked a question)
    last_assistant_msg = ''
    if chat_history:
        for m in reversed(chat_history):
            if m['role'] == 'assistant':
                last_assistant_msg = m['content'].strip()
                break
    # Check if AI's last message contained a question — look at last non-empty line
    last_lines = [l.strip() for l in last_assistant_msg.splitlines() if l.strip()]
    last_meaningful_line = last_lines[-1] if last_lines else ''
    ai_asked_question = last_meaningful_line.endswith('?')

    # Pure single-word affirmatives — no new info, reuse filters wholesale
    affirmatives = ['sure', 'yes', 'ok', 'okay', 'show me', 'show more', 'more', 'yep', 'please']
    is_affirmative = text_lower_check.rstrip('!.') in affirmatives

    # Purely additive refinement phrases
    refinement_only_phrases = [
        'show more', 'more options', 'any others', 'what else',
        'day camps only', 'overnight only', 'just day', 'just overnight',
        'cheaper', 'less expensive', 'more affordable',
        'closer', 'nearer', 'same area',
    ]
    is_pure_refinement = any(p in text_lower_check for p in refinement_only_phrases)

    # Location-only reply signals (province/city being added to existing search)
    location_words = [
        'ontario', 'quebec', 'bc', 'british columbia', 'alberta', 'manitoba',
        'saskatchewan', 'nova scotia', 'new brunswick', 'pei', 'newfoundland',
        'toronto', 'vancouver', 'calgary', 'edmonton', 'ottawa', 'montreal',
        'laval', 'longueuil', 'gatineau', 'sherbrooke', 'quebec city', 'levis',
        'hamilton', 'waterloo', 'kitchener', 'london', 'barrie', 'kingston',
        'peterborough', 'sudbury', 'thunder bay', 'windsor', 'niagara',
        'burnaby', 'surrey', 'richmond', 'victoria', 'kelowna', 'abbotsford',
        'winnipeg', 'saskatoon', 'regina', 'halifax', 'moncton',
        'i am from', "i'm from", 'we are in', "we're in", 'located in', 'in ',
    ]
    is_location_reply = any(w in text_lower_check for w in location_words)

    # Detect correction/challenge messages — user pointing out a missing result
    # These should always merge into last_filters, never start fresh
    correction_patterns = [
        "isn't ", "is not ", "what about ", "you missed ", "you forgot ",
        "should be ", "also overnight", "also a ", "that's also", "that is also",
        "aren't they", "are they not", "isn't that", "isn't it",
        "how about ", "didn't you", "did you miss",
    ]
    is_correction = any(p in text_lower_check for p in correction_patterns)

    # If this is a correction, try to extract any camp name mentioned in the message
    # e.g. "isn't Teen Ranch overnight too?" → named_camp_override = "Teen Ranch"
    # We do this by checking if any known camp-like proper nouns appear (title case words)
    import re as _re2
    named_camp_override = None
    if is_correction:
        # Extract sequences of title-case words as potential camp names
        title_words = _re2.findall(r'(?:[A-Z][a-z]+(?:[\s]+[A-Z][a-z]+)*)', user_text)
        if title_words:
            # Take the longest title-case sequence as the most likely camp name
            named_camp_override = max(title_words, key=len)

    # ── Determine search intent: FRESH or REFINE ────────────────────────────
    # Extract fresh filters from new message first, then decide whether to
    # use standalone (fresh) or merge with last_filters (refine).
    import re as _re
    new_filters_peek = extract_filters(user_text, config['GEMINI_API_KEY'])

    # Count independent filter signals present in the new message
    def _count_signals(f, text):
        n = 0
        if f.get('activity'):                    n += 1
        if f.get('gender'):                      n += 1
        if f.get('style'):                       n += 1
        if f.get('province') or f.get('region'): n += 1
        if f.get('age') or _re.search(r'\d+\s*(?:year|yr)', text.lower()): n += 1
        return n

    new_signal_count = _count_signals(new_filters_peek, user_text)

    # Gender flip — previous was girls, new message mentions boy (or vice versa)
    boy_words  = ['son', 'boy', 'for him', 'my boy', 'brother']
    girl_words = ['daughter', 'girl', 'for her', 'my girl', 'sister']
    prev_gender   = (last_filters.get('gender') or '').lower() if last_filters else ''
    gender_flip   = (
        (prev_gender == 'girls' and any(w in text_lower_check for w in boy_words)) or
        (prev_gender == 'boys'  and any(w in text_lower_check for w in girl_words))
    )
    is_new_child  = gender_flip  # used later for acknowledgement text

    # Activity change — new message explicitly requests a different activity
    prev_activity = (last_filters.get('activity') or '').lower() if last_filters else ''
    new_activity  = (new_filters_peek.get('activity') or '').lower()
    activity_changed = bool(new_activity and prev_activity and new_activity != prev_activity)

    # ── Decision tree ────────────────────────────────────────────────────────
    # FRESH  — message is self-contained (2+ own signals), activity changed,
    #          gender flipped, or no prior context exists
    # MERGE  — short additive reply to AI question or known refinement phrase
    # REUSE  — pure affirmative ("yes", "sure", "ok")

    if is_affirmative and last_filters:
        # Pure yes/sure — reuse last filters exactly
        filters = {k: v for k, v in last_filters.items()
                   if k in ('province', 'region', 'activity', 'style', 'gender', 'age')}

    elif (not last_filters or
          new_signal_count >= 2 or
          activity_changed or
          gender_flip):
        # Self-contained or clearly new — fresh filters only, no inheritance
        filters = _validate_filters(new_filters_peek, user_text)

        # CRITICAL: If the new message adds location/age/style but no new activity,
        # and the previous turn had an activity, inherit it.
        # e.g. "hockey camps" → "this for age-10 in Toronto" should keep activity=hockey
        if (last_filters and
            not new_activity and
            prev_activity and
            not activity_changed):
            filters['activity'] = last_filters['activity']

    elif ai_asked_question or is_pure_refinement or is_location_reply or is_correction:
        # Short additive reply — merge new detail into existing search
        clean_last = {k: v for k, v in last_filters.items()}
        if (clean_last.get('province') or '').lower() in ('canada', 'all', 'any', 'nationwide'):
            clean_last['province'] = None
        filters = {**clean_last, **{k: v for k, v in new_filters_peek.items() if v is not None}}

    else:
        # Default: treat as fresh
        filters = new_filters_peek
    # Step 3: Fetch matching camps from camps_clean
    # Build engine once — reused for embeddings and search
    from sqlalchemy import create_engine as _ce
    _search_engine = _ce(get_db_uri(config, config["DB_CAMP_DIR"]), pool_pre_ping=True)
    camps, province, region, fallback, _activity_has_codes = search_camps(
        filters, config,
        named_camp=named_camp_override if 'named_camp_override' in locals() else None,
        engine=_search_engine
    )
    # DEBUG — remove after confirming fix
    import streamlit as _st
    _st.session_state['_debug_last'] = {
        'filters': filters,
        'fallback': fallback,
        'activity_has_codes': _activity_has_codes,
        'camp_count': len(camps),
        'top5_camps': [c.get('camp_name') for c in camps[:5]],
    }

    # Step 4: Build context string for Gemini
    # For dietary/niche keywords with no results, return a camps.ca search URL
    from urllib.parse import quote_plus as qp

    # Keywords that are ambiguous — could mean a camp type OR a dietary/lifestyle need
    # For these we check the DB first, and if no results, ask a clarifying question
    # rather than assuming one interpretation
    ambiguous_keywords = [
        'vegetarian', 'vegan', 'kosher', 'halal', 'organic',
        'plant-based', 'gluten', 'gluten-free', 'gluten free',
        'nut-free', 'nut free', 'peanut', 'dairy-free', 'dairy free',
        'allergy', 'allergen',
    ]
    activity_raw = (filters.get('activity') or '').lower().strip()
    query_lower  = user_text.lower()

    # Detect if query contains an ambiguous keyword
    matched_ambiguous = next(
        (kw for kw in ambiguous_keywords if kw in query_lower or kw == activity_raw),
        None
    )

    # Check if this looks like a bare ambiguous search (no location, age, or other context)
    has_location  = bool(filters.get('province') or filters.get('region'))
    has_age       = bool(filters.get('age'))
    has_style     = bool(filters.get('style'))
    is_bare_ambiguous = (
        matched_ambiguous and
        not has_location and not has_age and not has_style and
        len(user_text.split()) <= 4  # short query like "vegetarian camps"
    )

    # If bare ambiguous query with no DB results → ask clarifying question
    if not camps and is_bare_ambiguous:
        elapsed = time.time() - start
        search_url = f"https://www.camps.ca/camp-site-search.php?keywrds={qp(matched_ambiguous + ' camps')}"
        response = (
            f"I want to make sure I find the right camps for you! When you say **'{matched_ambiguous} camps'**, do you mean:\n\n"
            f"1. 🏕️ **Camps that specifically identify as {matched_ambiguous}** (e.g. a camp with a {matched_ambiguous} philosophy or program)\n"
            f"2. 🥗 **Camps that accommodate {matched_ambiguous} dietary needs** for your child\n\n"
            f"In the meantime, you can also browse our full directory here:\n"
            f"🔍 [Search camps.ca for '{matched_ambiguous} camps']({search_url})\n\n"
            f"*Just reply with 1 or 2, or add more details like location or your child's age and I'll search our member network!*"
        )
        return response, elapsed, filters

    # If ambiguous keyword WITH context (location/age etc.) and no results → show search URL
    if not camps and matched_ambiguous:
        elapsed = time.time() - start
        search_url = f"https://www.camps.ca/camp-site-search.php?keywrds={qp(matched_ambiguous + ' camps')}"
        response = (
            f"I couldn't find camps in our verified member network specifically matching **{matched_ambiguous}**"
            f"{' in ' + filters.get('region', filters.get('province', '')) if has_location else ''}.\n\n"
            f"Many camps accommodate {matched_ambiguous} needs — we recommend contacting camps directly.\n\n"
            f"You can also browse our full directory:\n"
            f"🔍 [Search camps.ca for '{matched_ambiguous} camps']({search_url})\n\n"
            f"💬 *Try searching by location, age, or activity type and I'll find verified member camps for you!*"
        )
        return response, elapsed, filters

    # Step 4: Deduplicate then enforce gold-first cap — gold always included
    raw_deduped = dedupe_camps(camps) if camps else []

    # ── Relevancy thresholds ────────────────────────────────────────────────
    # Rule 1: top score < 10%  → ask a clarifying question, show nothing
    # Rule 2: score ≤ 15%      → silently drop from results
    # Rule 3: everything shown  → only camps scoring > 15%
    activity_searched = (filters.get('activity') or '').strip()
    from urllib.parse import quote_plus as _qp

    # Hard fallback guard:
    # If specialty codes were used in SQL but fallback fired, the camps returned
    # are NOT specialty matches (they're from province/global fallback).
    # In this case, fall through to semantic search as backup rather than showing
    # irrelevant camps with _semantic_score=0.9 applied incorrectly.
    is_activity_fallback = (
        activity_searched and
        fallback in ('no_activity_in_region', 'no_activity_in_province', 'province_only', 'no_match')
    )

    # If specialty codes matched but SQL still fell back, clear the code flag
    # so semantic search runs as backup
    if _activity_has_codes and is_activity_fallback:
        _activity_has_codes = False

    if not raw_deduped or (is_activity_fallback and not raw_deduped):
        elapsed = time.time() - start
        loc_hint = filters.get('region') or filters.get('province') or 'Canada'
        if activity_searched:
            search_url = f"https://www.camps.ca/camp-site-search.php?keywrds={_qp(activity_searched + ' camps')}"
            return (
                f"I couldn't find **{activity_searched} camps** in {loc_hint} in our verified member network.\n\n"
                f"Try a broader location, or search our full directory:\n"
                f"🔍 [Search camps.ca for {activity_searched} camps]({search_url})\n\n"
                f"💬 *Want me to search all of Ontario, or try a different activity?*"
            ), elapsed, filters
        return (
            "I couldn't find any camps matching those criteria in our verified member network. "
            "Try broadening your search — remove a filter or ask me to widen the location."
        ), elapsed, filters

    # ── Activity ranking: specialty codes first, semantic for the long tail ──────
    #
    # ARCHITECTURE:
    #   Pass 1 (specialty codes): SQL-level exact match for known activities.
    #          Fast, precise. Returns only camps with that specialty in the DB.
    #          Used when activity maps to at least one non-generic code.
    #
    #   Pass 2 (semantic):  For activities with no specialty code match
    #          (e.g. "financial literacy", "entrepreneurship", "mindfulness").
    #          Uses elbow detection on cosine similarity scores rather than
    #          a fixed threshold — because embedding-001 compresses all scores
    #          into a narrow band (0.82–0.89), making fixed thresholds useless.
    #
    # NOTE on embedding-001 score compression:
    #   Related camps score ~0.87–0.91, unrelated camps ~0.82–0.86.
    #   The gap is ~0.05 — too small for a fixed floor.
    #   Elbow detection finds the LARGEST drop in the sorted score list
    #   and cuts there, separating signal from noise dynamically.

    activity_query = (filters.get('activity') or '').strip()

    # ── Known activity: SQL already filtered by specialty code ──────────────────
    # If SQL used specialty codes, camps in raw_deduped are confirmed matches.
    # Give them high semantic scores and skip semantic search.
    code_match_used = _activity_has_codes  # set during SQL build phase

    if code_match_used:
        for c in raw_deduped:
            c['_semantic_score'] = 0.9  # confirmed specialty match

    # ── Semantic search (long-tail activities with no code match) ─────────────
    if not code_match_used and activity_query and embeddings_are_ready(_search_engine):
        raw_deduped = semantic_score_camps(
            raw_deduped, activity_query, config['GEMINI_API_KEY'], _search_engine
        )
        # Elbow detection: find meaningful score drop, cut below it.
        #
        # embedding-001 compresses scores into a narrow band (~0.82–0.91).
        # Adjacent score gaps are typically 0.001–0.006 (noise).
        # A real relevance boundary shows a gap of 0.015+ and only appears
        # after the top cluster of genuinely relevant camps.
        #
        # Rules:
        #   - Never cut before position 8 (always return at least 8 results)
        #   - Only cut on gaps > 0.015 (above noise floor)
        #   - Never cut after position 25 (cap at 25 semantic results)
        sorted_camps = sorted(raw_deduped, key=lambda c: -c.get('_semantic_score', 0))
        scores = [c.get('_semantic_score', 0) for c in sorted_camps]

        MIN_RESULTS  = 8
        MAX_RESULTS  = 25
        GAP_THRESHOLD = 0.015   # meaningful drop for embedding-001

        cut_idx = min(len(scores), MAX_RESULTS)  # default: keep up to 25
        if len(scores) > MIN_RESULTS:
            # Only look for elbow after MIN_RESULTS position
            for i in range(MIN_RESULTS, min(len(scores)-1, MAX_RESULTS)):
                if scores[i] - scores[i+1] >= GAP_THRESHOLD:
                    cut_idx = i + 1
                    break

        raw_deduped = sorted_camps[:cut_idx]

        if not raw_deduped:
            elapsed = time.time() - start
            loc_hint = filters.get('region') or filters.get('province') or 'Canada'
            from urllib.parse import quote_plus as _qp2
            search_url = f"https://www.camps.ca/camp-site-search.php?keywrds={_qp2(activity_query + ' camps')}"
            return (
                f"I couldn't find **{activity_query} camps** in {loc_hint} in our verified network.\n\n"
                f"Try searching our full directory:\n"
                f"🔍 [Search camps.ca for {activity_query} camps]({search_url})\n\n"
                f"💬 *Or tell me a different activity or location and I'll search again.*"
            ), elapsed, filters

    elif not code_match_used:
        for c in raw_deduped:
            c['_semantic_score'] = 0.5  # no activity searched — neutral

    scored = score_and_rank(raw_deduped, filters, fallback)

    # Drop structurally weak matches — but always show at least top 5
    deduped = [c for c in scored if c.get('_relevancy', 0) > 0.15] or scored[:5]

    # New-child acknowledgement prefix
    new_child_prefix = ""
    if is_new_child:
        new_child_prefix = "Switching gears for your other child! " if filters.get('gender') not in ('girls','boys') else (
            "Switching gears for your son! " if filters.get('gender') == 'boys' else "Switching gears for your daughter! "
        )

    # Step 5: Single Gemini call — blurbs only (one sentence per camp)
    blurbs = generate_blurbs(deduped, user_text, config["GEMINI_API_KEY"])

    # Step 6: Python renders guaranteed-complete list
    rendered = render_results(deduped, blurbs, user_text, filters, fallback, province, region)

    # Closing question — one Gemini sentence or hardcoded fallback
    closing_system = "You are a camp consultant. Write ONE short friendly question (max 12 words, ending with ?) to help the user narrow down their camp search. No preamble."
    closing_q = call_gemini(closing_system, f"User searched: {user_text}. Filters: {filters}", config["GEMINI_API_KEY"], max_tokens=40)
    if not closing_q or '?' not in closing_q:
        closing_q = "Want me to filter by age range, cost, or specific activities?"

    if new_child_prefix:
        response = new_child_prefix + "\n\n" + rendered + "\n\n" + closing_q.strip()
    else:
        response = rendered + "\n\n" + closing_q.strip()

    elapsed = time.time() - start
    return response, elapsed, filters


# ═════════════════════════════════════════════
# STREAMLIT UI — camps.ca look & feel
# ═════════════════════════════════════════════
st.set_page_config(
    page_title="Camp Finder | camps.ca",
    page_icon="🏕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&family=Source+Sans+3:wght@400;500;600&display=swap');

/* ── Reset & base ─────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background: #f5f6f8;
    color: #2c3e50;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Topbar ───────────────────────────────── */
.topbar {
    background: #1b5e20;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem;
    position: sticky;
    top: 0;
    z-index: 1000;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.topbar-logo {
    font-family: 'Montserrat', sans-serif;
    font-weight: 900;
    font-size: 1.4rem;
    color: #fff;
    letter-spacing: -0.5px;
    text-decoration: none;
}
.topbar-logo em {
    color: #f9a825;
    font-style: normal;
}
.topbar-badge {
    background: #f9a825;
    color: #1b5e20;
    font-size: 0.6rem;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 3px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.topbar-right {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.7);
    font-weight: 500;
}
.topbar-right strong { color: #fff; }

/* ── Hero ─────────────────────────────────── */
.hero {
    background: linear-gradient(160deg, #2e7d32 0%, #388e3c 45%, #43a047 100%);
    padding: 2rem 2rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0; right: 0;
    height: 28px;
    background: #f5f6f8;
    clip-path: ellipse(55% 100% at 50% 100%);
}
.hero-inner { position: relative; z-index: 1; }
.hero h1 {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.9rem;
    font-weight: 900;
    color: #fff;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
    text-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.hero-sub {
    color: rgba(255,255,255,0.88);
    font-size: 0.95rem;
    font-weight: 500;
    margin: 0 0 1rem;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.4rem;
    margin-top: 0.6rem;
}
.pill {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    color: #fff;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s, transform 0.15s;
    text-decoration: none;
    display: inline-block;
}
.pill:hover {
    background: rgba(255,255,255,0.32);
    transform: translateY(-1px);
    text-decoration: none;
    color: #fff;
}
.pill:visited { color: #fff; }

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid #e0e4ea;
}
.sb-head {
    background: #1b5e20;
    color: #fff;
    padding: 0.9rem 1rem;
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 0.95rem;
    border-radius: 0 0 10px 10px;
    margin: 0 -0.5rem 1rem;
    text-align: center;
    letter-spacing: 0.2px;
}
.sb-section {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    color: #1b5e20;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 1rem 0 0.35rem;
    padding-bottom: 0.2rem;
    border-bottom: 2px solid #e8f5e9;
}

/* ── Streamlit widgets override ─────────────── */
.stTextInput > label,
.stSelectbox > label,
.stRadio > label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #37474f !important;
}
.stTextInput input {
    border: 1.5px solid #c8e6c9 !important;
    border-radius: 7px !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 0.7rem !important;
}
.stTextInput input:focus {
    border-color: #2e7d32 !important;
    box-shadow: 0 0 0 2px rgba(46,125,50,0.12) !important;
    outline: none !important;
}
.stSelectbox > div > div {
    border: 1.5px solid #c8e6c9 !important;
    border-radius: 7px !important;
    font-size: 0.88rem !important;
}
div[data-testid="stRadio"] > div {
    gap: 0.3rem !important;
}

/* ── Find button ────────────────────────────── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #2e7d32, #388e3c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 25px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 3px 10px rgba(46,125,50,0.35) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 16px rgba(46,125,50,0.45) !important;
}

/* ── Clear / secondary buttons ─────────────── */
div.stButton > button {
    background: #fff !important;
    color: #2e7d32 !important;
    border: 1.5px solid #2e7d32 !important;
    border-radius: 20px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 0.35rem 0.9rem !important;
    transition: all 0.18s !important;
}
div.stButton > button:hover {
    background: #e8f5e9 !important;
    border-color: #1b5e20 !important;
    color: #1b5e20 !important;
}

/* ── Metric cards ───────────────────────────── */
[data-testid="stMetric"] {
    background: #e8f5e9 !important;
    border: 1px solid #c8e6c9 !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.7rem !important;
    text-align: center !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: #558b2f !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 900 !important;
    color: #1b5e20 !important;
    font-size: 1.4rem !important;
}

/* ── Chat messages ───────────────────────────── */
.main-chat {
    max-width: 860px;
    margin: 1.5rem auto 6rem;
    padding: 0 1.2rem;
}
[data-testid="stChatMessage"] {
    background: #fff !important;
    border: 1px solid #e5eaef !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.8rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
/* Assistant message — left green accent */
[data-testid="stChatMessage"][data-testid*="assistant"],
div[class*="stChatMessage"]:has(img[alt="assistant"]) {
    border-left: 4px solid #2e7d32 !important;
}

/* ── Chat input bar ──────────────────────────── */
[data-testid="stChatInput"] textarea {
    border: 2px solid #c8e6c9 !important;
    border-radius: 25px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 1.1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2e7d32 !important;
    box-shadow: 0 0 0 3px rgba(46,125,50,0.12) !important;
    outline: none !important;
}

/* ── Links ───────────────────────────────────── */
a { color: #2e7d32 !important; font-weight: 600 !important; }
a:hover { color: #1b5e20 !important; text-decoration: underline !important; }

/* ── Caption / footnote ─────────────────────── */
.stCaption {
    color: #78909c !important;
    font-size: 0.72rem !important;
}

/* ── Divider ─────────────────────────────────── */
hr { border-color: #e8edf2 !important; margin: 0.8rem 0 !important; }

/* ── Verified badge ─────────────────────────── */
.verified-banner {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 6px;
    padding: 0.45rem 0.7rem;
    font-size: 0.73rem;
    color: #2e7d32;
    font-weight: 600;
    text-align: center;
    line-height: 1.5;
    margin-top: 0.5rem;
}

/* ── Spinner ─────────────────────────────────── */
.stSpinner > div { border-top-color: #2e7d32 !important; }
</style>
""", unsafe_allow_html=True)

# ── Topbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <span class="topbar-logo">🏕️ camps<em>.ca</em></span>
        <span class="topbar-badge">AI Powered</span>
    </div>
    <div class="topbar-right">
        Canada's Camp Discovery Platform &nbsp;|&nbsp;
        <strong>Verified Member Camps Only</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <h1>Find Your Perfect Canadian Camp 🍁</h1>
    <p class="hero-sub">
      Search thousands of verified day &amp; overnight camps across Canada
    </p>
    <div class="hero-pills">
      <a class="pill" href="https://www.camps.ca/hockey_schools_camps.php" target="_blank">🏒 Hockey</a>
      <a class="pill" href="https://www.camps.ca/stem-camps.php" target="_blank">💻 STEM</a>
      <a class="pill" href="https://www.camps.ca/fine_art_camps.php" target="_blank">🎨 Arts</a>
      <a class="pill" href="https://www.camps.ca/toronto_camps.php" target="_blank">⚽ Sports</a>
      <a class="pill" href="https://www.camps.ca/outdoor-education.php" target="_blank">🌲 Outdoor</a>
      <a class="pill" href="https://www.camps.ca/musical-theatre-camps.php" target="_blank">🎭 Theatre</a>
      <a class="pill" href="https://www.camps.ca/swimming-camps.php" target="_blank">🏊 Swimming</a>
      <a class="pill" href="https://www.camps.ca/music-lessons.php" target="_blank">🎸 Music</a>
      <a class="pill" href="https://www.camps.ca/horseback-riding-lessons.php" target="_blank">🐴 Equestrian</a>
      <a class="pill" href="https://www.camps.ca/robotics-camp-kids.php" target="_blank">🤖 Robotics</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Config & data ─────────────────────────────────────────────────────────────
config = get_config()
required = ["GEMINI_API_KEY", "PINECONE_API_KEY", "DB_HOST", "DB_USER", "DB_PASS", "INDEX_HOST"]
missing = [k for k in required if not config.get(k)]
if missing:
    st.error(f"⚠️ Missing configuration: {', '.join(missing)}")
    st.stop()

with st.spinner("Loading member camps..."):
    client_camps = load_client_camps(config)

# ── Sidebar ───────────────────────────────────────────────────────────────────

def _show_embedding_admin(config):
    """Sidebar widget for building / refreshing the semantic search index."""
    import requests as _r, json as _j, time as _t
    from sqlalchemy import create_engine as _ce, text as _tx

    try:
        _eng = _ce(get_db_uri(config, config["DB_CAMP_DIR"]), pool_pre_ping=True)
        _ready = embeddings_are_ready(_eng)
    except Exception as _e:
        st.error(f"DB error: {_e}")
        return

    if _ready:
        with _eng.connect() as _c:
            _n = _c.execute(_tx("SELECT COUNT(*) FROM camp_directory.camp_embeddings")).scalar()
        st.success(f"Search index ready — {_n} camps indexed")
    else:
        st.warning("Search index not built — results use basic matching only")

    # Test API connectivity before committing to full build
    col1, col2 = st.columns(2)
    _run_test  = col1.button("Test API", use_container_width=True)
    _run_build = col2.button("Build Index", use_container_width=True)

    if _run_test:
        import requests as _tr
        _api_key = config['GEMINI_API_KEY']

        # Step 1: List available models to find working embedding model
        st.write("**Checking available models...**")
        try:
            _list_resp = _tr.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={_api_key}",
                timeout=15
            )
            if _list_resp.ok:
                _models = _list_resp.json().get("models", [])
                _embed_models = [
                    m["name"] for m in _models
                    if "embedContent" in m.get("supportedGenerationMethods", [])
                ]
                if _embed_models:
                    st.success(f"Found {len(_embed_models)} embedding model(s):")
                    for _m in _embed_models:
                        st.code(_m)
                    # Auto-test the first available one
                    _best = _embed_models[0]
                    _model_id = _best.split("/")[-1]  # e.g. "embedding-001"
                    st.write(f"Testing `{_best}`...")
                    _test_resp = _tr.post(
                        f"https://generativelanguage.googleapis.com/v1beta/{_best}:embedContent?key={_api_key}",
                        headers={"Content-Type": "application/json"},
                        json={"model": _best,
                              "content": {"parts": [{"text": "test"}]},
                              "taskType": "RETRIEVAL_DOCUMENT"},
                        timeout=15
                    )
                    if _test_resp.ok:
                        _dims = len(_test_resp.json()["embedding"]["values"])
                        st.success(f"✅ Working model found: `{_best}` — {_dims} dimensions")
                        st.info(f"Update EMBED_MODEL in the code to: `{_best}`")
                    else:
                        st.error(f"Model listed but embedContent failed: {_test_resp.text[:300]}")
                else:
                    st.error("No embedding models available for this API key")
                    st.write("All models:")
                    for _m in _models[:10]:
                        st.code(f"{_m['name']}: {_m.get('supportedGenerationMethods', [])}")
            else:
                st.error(f"ListModels failed {_list_resp.status_code}: {_list_resp.text[:300]}")
        except Exception as _te:
            st.error(f"Connection failed: {_te}")
        return

    if not _run_build:
        return

    def _fingerprint(camp):
        parts = []
        if camp.get("camp_name"):  parts.append(f"Camp: {camp['camp_name']}.")
        if camp.get("description"):parts.append(f"About: {camp['description']}")
        if camp.get("activities"): parts.append(f"Activities: {camp['activities']}.")
        if camp.get("programs"):
            ps = [p.strip() for p in camp["programs"].split("|||") if p.strip()]
            if ps: parts.append("Programs: " + " | ".join(ps[:20]))
        if camp.get("camp_style"): parts.append(f"Type: {camp['camp_style']} camp.")
        if camp.get("city") and camp.get("province"):
            parts.append(f"Location: {camp['city']}, {camp['province']}.")
        if camp.get("age_min") is not None:
            parts.append(f"Ages: {camp['age_min']} to {camp['age_max']}.")
        return " ".join(parts)

    # Auto-discover working embedding model
    import requests as _disc
    _api_key = config['GEMINI_API_KEY']
    _embed_model = None
    try:
        _lr = _disc.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={_api_key}",
            timeout=10
        )
        if _lr.ok:
            _avail = [
                m["name"] for m in _lr.json().get("models", [])
                if "embedContent" in m.get("supportedGenerationMethods", [])
            ]
            if _avail:
                _embed_model = _avail[0]
    except Exception:
        pass

    if not _embed_model:
        st.error("No embedding models found for this API key. Run 'Test API' to diagnose.")
        return

    EMBED_URL = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{_embed_model}:embedContent?key={_api_key}"
    )
    CAMP_SQL = _tx("""
        SELECT cc.cid, cc.camp_name, cc.description, cc.camp_style,
               cc.province, MIN(sc.city) AS city,
               GROUP_CONCAT(DISTINCT sc.specialty_label
                   ORDER BY sc.specialty_label SEPARATOR ', ') AS activities,
               GROUP_CONCAT(
                   DISTINCT CONCAT(sc.class_name, COALESCE(
                       IF(NULLIF(TRIM(s.mini_description),'') IS NOT NULL,
                          CONCAT(' --- ', TRIM(s.mini_description)), NULL),
                       IF(NULLIF(TRIM(s.description),'') IS NOT NULL,
                          CONCAT(' --- ', LEFT(TRIM(s.description),200)), NULL), ''))
                   ORDER BY sc.listing_tier SEPARATOR ' ||| ') AS programs,
               MIN(sc.age_from) AS age_min, MAX(sc.age_to) AS age_max
        FROM camp_directory.camps_clean cc
        JOIN camp_directory.sessions_clean sc
            ON sc.cid=cc.cid AND sc.province=cc.province
        LEFT JOIN camp_directory.sessions s ON s.id=sc.session_id
        WHERE cc.status=1 AND sc.status=1
        GROUP BY cc.cid, cc.camp_name, cc.description, cc.camp_style, cc.province
    """)

    with st.spinner("Building search index... (~2 min)"):
        try:
            with _eng.connect() as _c:
                _res = _c.execute(CAMP_SQL)
                _cols = list(_res.keys())
                _camps = [dict(zip(_cols, r)) for r in _res.fetchall()]

            # Test first camp before running all 335 to surface errors early
            _test_fp   = _fingerprint(_camps[0])
            _test_resp = _r.post(EMBED_URL,
                                 headers={"Content-Type": "application/json"},
                                 json={"model": _embed_model,
                                       "content": {"parts": [{"text": _test_fp}]},
                                       "taskType": "RETRIEVAL_DOCUMENT"},
                                 timeout=20)
            if not _test_resp.ok:
                st.error(
                    f"Embedding API error {_test_resp.status_code}: "
                    f"{_test_resp.text[:500]}"
                )
                st.stop()

            _ok, _fail, _errors = 0, 0, []
            _prog = st.progress(0, text="Indexing camps...")
            for _i, _camp in enumerate(_camps):
                try:
                    _fp   = _fingerprint(_camp)
                    _resp = _r.post(EMBED_URL,
                                    headers={"Content-Type": "application/json"},
                                    json={"model": _embed_model,
                                          "content": {"parts": [{"text": _fp}]},
                                          "taskType": "RETRIEVAL_DOCUMENT"},
                                    timeout=20)
                    _resp.raise_for_status()
                    _vec = _resp.json()["embedding"]["values"]

                    with _eng.connect() as _c2:
                        _c2.execute(_tx("""
                            INSERT INTO camp_directory.camp_embeddings
                                (cid, embedding, fingerprint, updated_at)
                            VALUES (:cid, :emb, :fp, NOW())
                            ON DUPLICATE KEY UPDATE
                                embedding=VALUES(embedding),
                                fingerprint=VALUES(fingerprint),
                                updated_at=NOW()
                        """), {"cid": _camp["cid"], "emb": _j.dumps(_vec), "fp": _fp[:2000]})
                        _c2.commit()
                    _ok += 1
                except Exception as _camp_err:
                    _fail += 1
                    if len(_errors) < 3:  # capture first 3 errors for diagnosis
                        _errors.append(f"cid={_camp.get('cid')} {_camp.get('camp_name','?')}: {_camp_err}")

                _prog.progress((_i + 1) / len(_camps),
                               text=f"Indexed {_i+1}/{len(_camps)} camps...")
                _t.sleep(0.1)

            load_camp_embeddings.clear()
            if _ok > 0:
                st.success(f"Done — {_ok} camps indexed, {_fail} failed")
            else:
                st.error(f"All {_fail} camps failed. First errors:")
                for _e in _errors:
                    st.code(_e)
        except Exception as _e:
            st.error(f"Index build failed: {_e}")

with st.sidebar:
    st.markdown('<div class="sb-head">🔍 Camp Search Consultant</div>', unsafe_allow_html=True)

    # Admin: Embedding Management
    with st.expander("⚙️ Admin: Search Index", expanded=False):
        _show_embedding_admin(config)

    with st.form("consultation_form"):
        st.markdown('<div class="sb-section">👤 About You</div>', unsafe_allow_html=True)
        contact_name = st.text_input("Your name", placeholder="e.g. Sarah")
        region_camp  = st.text_input("City or region", placeholder="e.g. Toronto, Ottawa, Vancouver")

        st.markdown('<div class="sb-section">🏕️ Camp Preferences</div>', unsafe_allow_html=True)
        category_options = [
            "Any", "STEM / Science / Technology", "Sports",
            "Arts / Music / Dance", "Outdoor / Adventure",
            "Academic / Tutoring", "Language / French",
            "Equestrian / Riding", "Cooking / Chef",
            "Leadership", "Traditional / Multi-Activity",
            "Special Needs / Adapted"
        ]
        category_camp = st.selectbox("Camp category", category_options)
        camp_type     = st.text_input("Specific activity", placeholder="e.g. robotics, hockey, painting")
        age_child     = st.text_input("Child's age(s)", placeholder="e.g. 10, or 8 and 12")
        costs_camp    = st.text_input("Max weekly budget", placeholder="e.g. $500")
        date_camp     = st.text_input("Preferred dates", placeholder="e.g. July, Summer 2026")
        overnight     = st.radio("Camp style", ["Either", "Day camp", "Overnight camp"], horizontal=True)
        submitted     = st.form_submit_button("🔍  Find My Camp!", use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Camps", f"{len(client_camps):,}")
    with col2:
        st.metric("Provinces", "13")

    if st.button("🗑️  New Search", use_container_width=True):
        st.session_state.messages = []
        st.session_state.consultation_done = False
        st.session_state.last_filters = None
        st.session_state.suppress_form = True   # prevent form re-firing on rerun
        st.rerun()

    st.markdown("""
    <div class="verified-banner">
        ✅ All results are verified members of<br>
        <strong>camps.ca</strong> &amp; <strong>ourkids.net</strong>
    </div>
    """, unsafe_allow_html=True)



# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "consultation_done" not in st.session_state:
    st.session_state.consultation_done = False
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None
if "suppress_form" not in st.session_state:
    st.session_state.suppress_form = False

# ── Consultation form handler ─────────────────────────────────────────────────
if submitted and contact_name and region_camp and not st.session_state.get('suppress_form', False):
    style_text  = "" if overnight == "Either" else overnight.lower()
    budget_text = f"under {costs_camp} per week" if costs_camp else ""
    age_text    = f"for a {age_child}-year-old" if age_child else ""
    type_text   = camp_type if camp_type else (category_camp if category_camp != "Any" else "general")
    date_text   = f"around {date_camp}" if date_camp else "this summer"

    structured_query = (
        f"My name is {contact_name} and I'm looking for {style_text} {type_text} camps "
        f"in {region_camp} {age_text} {budget_text} {date_text}."
    ).replace("  ", " ").strip()

    welcome = (
        f"Hi **{contact_name}**! 👋 Great to have you here.\n\n"
        f"I'm your personal camp consultant at **camps.ca** — Canada's #1 verified camp network.\n\n"
        f"Let me search our member database for the best **{type_text}** camps in **{region_camp}** for you... 🔍"
    )
    st.session_state.messages = [
        {"role": "assistant", "content": welcome},
        {"role": "user",      "content": structured_query}
    ]
    st.session_state.consultation_done = False
    st.rerun()

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 **Welcome to camps.ca Camp Finder!**\n\n"
            "I'm your personal Canadian camp consultant. Use the panel on the left for guided search, "
            "or type your question below.\n\n"
            "**Try asking:**\n"
            "- *Hockey camps in Toronto for a 10-year-old*\n"
            "- *STEM day camps in Ottawa under $500/week*\n"
            "- *Traditional overnight camps in Ontario*\n"
            "- *Arts camps in Vancouver for teens*\n"
            "- *Gluten-free camps in Mississauga*"
        )
    }]

# ── Render chat messages ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Process pending assistant response ───────────────────────────────────────
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if last_msg and last_msg["role"] == "user" and not st.session_state.consultation_done:
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching verified member camps..."):
            try:
                response, elapsed, filters = process_query(
                    last_msg["content"], config, client_camps,
                    chat_history=st.session_state.messages,
                    last_filters=st.session_state.last_filters
                )
                follow_up = (
                    "\n\n---\n"
                    "💬 *Want to refine? I can filter by age, budget, dates, style, or activity — just ask!*"
                )
                full_response = response + follow_up
                st.markdown(full_response)
                st.caption(f"⚡ {elapsed:.1f}s · Verified camps.ca member network")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.consultation_done = True
                st.session_state.last_filters = filters
            except Exception as e:
                err = f"❌ Something went wrong: {str(e)[:300]}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.session_state.consultation_done = True

# ── Chat input ────────────────────────────────────────────────────────────────
# DEBUG — always visible in main area
if st.session_state.get('_debug_last'):
    d = st.session_state['_debug_last']
    st.info(
        f"🐛 **DEBUG** | "
        f"activity=`{d['filters'].get('activity')}` | "
        f"has_codes=`{d['activity_has_codes']}` | "
        f"fallback=`{d['fallback']}` | "
        f"sql_camps=`{d['camp_count']}` | "
        f"top3={d['top5_camps'][:3]}"
    )

st.session_state.suppress_form = False  # user is engaging — re-allow form
if prompt := st.chat_input("🔍  Search camps... e.g. 'hockey camps in Toronto for a 10-year-old'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.consultation_done = False
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching verified member camps..."):
            try:
                response, elapsed, filters = process_query(
                    prompt, config, client_camps,
                    chat_history=st.session_state.messages,
                    last_filters=st.session_state.last_filters
                )
                st.markdown(response)
                st.caption(f"⚡ {elapsed:.1f}s · Verified camps.ca member network")
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.last_filters = filters
            except Exception as e:
                err = f"❌ Something went wrong: {str(e)[:300]}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
