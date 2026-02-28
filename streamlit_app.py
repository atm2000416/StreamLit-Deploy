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
        return json.loads(clean)
    except:
        return {}


def search_camps(filters, config, limit=100):
    """Query sessions_clean joined to camps_clean for session-level precision"""
    from sqlalchemy import create_engine, text

    # Region mapping for common cities/districts
    region_map = {
        'toronto': 'Toronto', 'etobicoke': 'Toronto', 'scarborough': 'Toronto',
        'north york': 'Toronto', 'east york': 'Toronto',
        'thornhill': 'York', 'richmond hill': 'York', 'markham': 'York',
        'oakville': 'Halton - Peel', 'mississauga': 'Halton - Peel',
        'brampton': 'Halton - Peel', 'burlington': 'Halton - Peel',
        'ajax': 'Durham', 'pickering': 'Durham', 'oshawa': 'Durham', 'whitby': 'Durham',
        'ottawa': 'Ottawa Region', 'nepean': 'Ottawa Region',
        'kanata': 'Ottawa Region', 'gloucester': 'Ottawa Region',
        'vancouver': 'Lower Mainland', 'burnaby': 'Lower Mainland',
        'surrey': 'Lower Mainland', 'richmond': 'Lower Mainland',
        'victoria': 'Vancouver Island', 'kelowna': 'Okanagan',
        'calgary': 'Calgary', 'edmonton': 'Edmonton',
        'winnipeg': 'Manitoba', 'montreal': 'Montreal',
        'halifax': 'Halifax', 'hamilton': 'City of Hamilton',
        'waterloo': 'Waterloo - Wellington', 'kitchener': 'Waterloo - Wellington',
        'kingston': 'Kingston - Prince Edward', 'barrie': 'Barrie - Orillia - Midland',
    }
    city_province_map = {
        'toronto': 'Ontario', 'etobicoke': 'Ontario', 'scarborough': 'Ontario',
        'north york': 'Ontario', 'east york': 'Ontario', 'thornhill': 'Ontario',
        'richmond hill': 'Ontario', 'markham': 'Ontario', 'oakville': 'Ontario',
        'mississauga': 'Ontario', 'brampton': 'Ontario', 'burlington': 'Ontario',
        'ajax': 'Ontario', 'pickering': 'Ontario', 'oshawa': 'Ontario',
        'whitby': 'Ontario', 'ottawa': 'Ontario', 'nepean': 'Ontario',
        'kanata': 'Ontario', 'gloucester': 'Ontario', 'hamilton': 'Ontario',
        'waterloo': 'Ontario', 'kitchener': 'Ontario', 'kingston': 'Ontario',
        'barrie': 'Ontario', 'london': 'Ontario',
        'vancouver': 'British Columbia', 'burnaby': 'British Columbia',
        'surrey': 'British Columbia', 'richmond': 'British Columbia',
        'victoria': 'British Columbia', 'kelowna': 'British Columbia',
        'calgary': 'Alberta', 'edmonton': 'Alberta',
        'winnipeg': 'Manitoba', 'montreal': 'Quebec',
        'halifax': 'Nova Scotia',
    }

    province = filters.get('province')
    region   = filters.get('region', '')
    activity = filters.get('activity', '')
    age      = filters.get('age')
    max_cost = filters.get('max_cost')
    style    = filters.get('style')
    gender   = filters.get('gender')  # 'girls', 'boys', or None

    # Resolve region/city to province if not set
    region_lower = (region or '').lower().strip()
    if not province and region_lower in city_province_map:
        province = city_province_map[region_lower]
    resolved_region = region_map.get(region_lower, region)

    conditions = ["sc.status = 1", "sc.province != 'Unknown'", "sc.is_virtual = 0"]
    params = {}

    if province:
        conditions.append("sc.province = :province")
        params['province'] = province

    if resolved_region:
        conditions.append("sc.region LIKE :region")
        params['region'] = f"%{resolved_region}%"

    if activity:
        # Map common activities to specialty codes for reliable matching
        activity_codes = {
            # Sports
            'hockey':       [29, 188],
            'skating':      [51],
            'ice skating':  [51],
            'sports':       [188, 12, 54, 66, 63, 29],
            'soccer':       [54],
            'basketball':   [11, 12],
            'tennis':       [66],
            'golf':         [26],
            'volleyball':   [63],
            'swimming':     [56],
            'swim':         [56],
            'sailing':      [49],
            'canoe':        [41],
            'canoeing':     [41],
            'lacrosse':     [188],
            'baseball':     [188],
            # STEM
            'stem':         [268, 18, 67, 160, 180, 50, 159, 266, 332],
            'science':      [268, 50, 18],
            'technology':   [268, 18, 180],
            'coding':       [18, 68, 159, 180, 266, 268, 332],
            'programming':  [18, 68, 159, 180, 266, 332],
            'robotics':     [67, 268, 160],
            'engineering':  [160, 268, 50],
            'math':         [20, 129],
            'ai':           [159, 302, 268],
            'game design':  [68],
            # Arts
            'arts':         [9, 10, 69, 173, 178, 355],
            'art':          [9, 10, 69, 173, 355],
            'music':        [37],
            'dance':        [22],
            'theatre':      [59, 172],
            'drama':        [59, 172],
            'animation':    [178],
            'photography':  [69],
            'cooking':      [133],
            'chef':         [133],
            # Outdoor / Traditional
            'outdoor':      [24, 41, 49, 58, 181, 265],
            'adventure':    [41, 181],
            'traditional':  [24, 58, 181, 265],
            'nature':       [24, 181],
            'canoe':        [41],
            # Academic
            'academic':     [20, 32, 97, 196, 314],
            'french':       [314],
            'language':     [314],
            'tutoring':     [32, 97, 314],
            'debate':       [302],
            'writing':      [362],
            'chess':        [278],
            # Other
            'equestrian':   [30],
            'riding':       [30],
            'horse':        [30],
            'leadership':   [33, 79, 88],
            'special needs':[252],
            'wellness':     [91],
            # Cheer & Fashion
            'cheer':        [164],
            'cheerleading': [164],
            'cheering':     [164],
            'fashion':      [71, 172, 264],
            'fashion design':[71, 264],
            'beauty':       [172],
        }
        act_lower = activity.lower().strip()
        codes     = activity_codes.get(act_lower, [])

        # Generic umbrella codes that should NOT qualify a session for a specific
        # activity search — e.g. 188=Sports should NOT match "hockey",
        # 268=STEM should NOT match "coding", 9/10=Arts should NOT match "dance"
        generic_codes = {188, 268, 9, 10, 79, 33}

        if codes:
            # Primary = specific codes only (strips umbrella generics)
            primary = [c for c in codes if c not in generic_codes]
            use     = primary if primary else codes   # if ALL codes are generic, use all
            code_list = ",".join(str(c) for c in use)
            conditions.append(f"(sc.specialty IN ({code_list}) OR sc.specialty2 IN ({code_list}))")
        else:
            # Unknown activity — text search on class_name
            conditions.append(
                "(sc.specialty_label LIKE :act OR sc.class_name LIKE :act2)"
            )
            params['act']  = f"%{act_lower}%"
            params['act2'] = f"%{act_lower}%"

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

    where = " AND ".join(conditions)
    sql = f"""SELECT
            sc.cid, sc.camp_name, sc.province,
            MIN(sc.region)                  AS region,
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
                    sc.session_id, ':', sc.class_name,
                    ' (ages ', sc.age_from, '-', sc.age_to, ')',
                    IF(COALESCE(NULLIF(TRIM(s.mini_description),''), NULLIF(TRIM(s.description),'')) IS NOT NULL,
                       CONCAT(' -- ', LEFT(COALESCE(NULLIF(TRIM(s.mini_description),''), TRIM(s.description)), 200)),
                       '')
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
                return [dict(zip(cols, row)) for row in rows], province, resolved_region, None

            # Fallback 1: drop activity, keep region + province
            if resolved_region and province and activity:
                params_no_act = {k:v for k,v in params.items() if k not in ('act','act2')}
                conds_no_act  = [c for c in conditions if 'specialty' not in c and 'class_name' not in c]
                sql_f1 = f"""SELECT sc.cid, sc.camp_name, sc.province,
                        MIN(sc.region) AS region, sc.camp_style, sc.listing_tier, sc.camp_url,
                        MIN(sc.session_url) AS session_url,
                        MIN(sc.age_from) AS age_min, MAX(sc.age_to) AS age_max,
                        MIN(NULLIF(sc.cost_from,0)) AS cost_min, MAX(NULLIF(sc.cost_to,0)) AS cost_max,
                        COUNT(DISTINCT sc.session_id) AS session_count,
                        GROUP_CONCAT(DISTINCT sc.specialty_label ORDER BY sc.specialty_label SEPARATOR ', ') AS activities,
                        GROUP_CONCAT(DISTINCT CONCAT(sc.session_id,':', sc.class_name,' (ages ',sc.age_from,'-',sc.age_to,')') ORDER BY sc.listing_tier SEPARATOR ' | ') AS matching_programs,
                        cc.description
                    FROM camp_directory.sessions_clean sc
                    JOIN camp_directory.camps_clean cc ON cc.cid = sc.cid AND cc.province = sc.province
                    WHERE {' AND '.join(conds_no_act)}
                    GROUP BY sc.cid, sc.camp_name, sc.province, sc.camp_style,
                             sc.listing_tier, sc.camp_url, cc.description
                    ORDER BY FIELD(sc.listing_tier,'gold','silver','bronze')
                    LIMIT {limit}"""
                r1 = conn.execute(text(sql_f1), params_no_act)
                rows1 = r1.fetchall()
                if rows1:
                    return [dict(zip(list(r1.keys()), row)) for row in rows1], province, resolved_region, 'no_activity_in_region'

            # Fallback 2: drop region, keep province + activity
            if resolved_region and province:
                params_no_reg = {k:v for k,v in params.items() if k != 'region'}
                conds_no_reg  = [c for c in conditions if 'region' not in c]
                sql_f2 = f"""SELECT sc.cid, sc.camp_name, sc.province,
                        MIN(sc.region) AS region, sc.camp_style, sc.listing_tier, sc.camp_url,
                        MIN(sc.session_url) AS session_url,
                        MIN(sc.age_from) AS age_min, MAX(sc.age_to) AS age_max,
                        MIN(NULLIF(sc.cost_from,0)) AS cost_min, MAX(NULLIF(sc.cost_to,0)) AS cost_max,
                        COUNT(DISTINCT sc.session_id) AS session_count,
                        GROUP_CONCAT(DISTINCT sc.specialty_label ORDER BY sc.specialty_label SEPARATOR ', ') AS activities,
                        GROUP_CONCAT(DISTINCT CONCAT(sc.session_id,':', sc.class_name,' (ages ',sc.age_from,'-',sc.age_to,')') ORDER BY sc.listing_tier SEPARATOR ' | ') AS matching_programs,
                        cc.description
                    FROM camp_directory.sessions_clean sc
                    JOIN camp_directory.camps_clean cc ON cc.cid = sc.cid AND cc.province = sc.province
                    WHERE {' AND '.join(conds_no_reg)}
                    GROUP BY sc.cid, sc.camp_name, sc.province, sc.camp_style,
                             sc.listing_tier, sc.camp_url, cc.description
                    ORDER BY FIELD(sc.listing_tier,'gold','silver','bronze')
                    LIMIT {limit}"""
                r2 = conn.execute(text(sql_f2), params_no_reg)
                rows2 = r2.fetchall()
                if rows2:
                    return [dict(zip(list(r2.keys()), row)) for row in rows2], province, resolved_region, 'no_activity_in_province'

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
                    "WHERE sc.status=1 AND sc.province=:p AND sc.is_virtual=0 "
                    "GROUP BY sc.cid, sc.camp_name, sc.province, sc.camp_style, sc.listing_tier, sc.camp_url, cc.description "
                    "ORDER BY FIELD(sc.listing_tier,'gold','silver','bronze') LIMIT :lim"
                ), {"p": province, "lim": limit})
                rows3 = r3.fetchall()
                if rows3:
                    return [dict(zip(list(r3.keys()), row)) for row in rows3], province, resolved_region, 'province_only'

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
                "WHERE sc.status=1 AND sc.province != 'Unknown' AND sc.is_virtual=0 "
                "GROUP BY sc.cid, sc.camp_name, sc.province, sc.camp_style, sc.listing_tier, sc.camp_url, cc.description "
                "ORDER BY FIELD(sc.listing_tier,'gold','silver','bronze') LIMIT :lim"
            ), {"lim": limit})
            rows4 = r4.fetchall()
            return [dict(zip(list(r4.keys()), row)) for row in rows4], province, resolved_region, 'no_match'

    except Exception as e:
        return [], province, resolved_region, f"error: {str(e)[:200]}"


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

def format_camp_context(camps):
    """Format camp data as readable context for Gemini — deduplicated by cid"""
    # Deduplicate: keep one row per camp (best region match = first occurrence)
    seen_cids = set()
    deduped = []
    for c in camps:
        cid = c.get('cid')
        if cid not in seen_cids:
            seen_cids.add(cid)
            deduped.append(c)
    lines = []
    for c in deduped:
        name          = c.get('camp_name', '') or ''
        session_count = int(c.get('session_count') or 0)
        session_url   = c.get('session_url') or ''
        camp_url      = c.get('camp_url') or ''
        # Link directly to the specific session when only 1 matched
        url           = session_url if (session_count == 1 and session_url) else (camp_url or session_url)
        province  = c.get('province', '')
        region    = c.get('region', '')
        style     = 'Day Camp' if c.get('camp_style') == 'day' else 'Overnight Camp'
        tier      = c.get('listing_tier', '')
        age_min   = c.get('age_min', '')
        age_max   = c.get('age_max', '')
        cost_min  = c.get('cost_min', '')
        cost_max  = c.get('cost_max', '')
        activities= clean_activities(c.get('activities', ''))
        desc      = (c.get('description') or '')[:300]

        age_str  = f"Ages {age_min}-{age_max}" if age_min and age_max else "Ages vary"
        cost_str = f"${cost_min:,}-${cost_max:,}/week" if cost_min and cost_max else "Contact for pricing"

        # matching_programs: pipe-separated "sessionID:class_name (ages X-Y)"
        matching_programs = (c.get('matching_programs') or '').strip()

        lines.append(
            f"CAMP: {name}\n"
            f"MARKDOWN_LINK: [{name}]({url})\n"
            f"SESSION_URL: {session_url}\n"
            f"SESSION_COUNT: {session_count}\n"
            f"MATCHING_PROGRAMS: {matching_programs}\n"
            f"Location: {region}, {province}\n"
            f"Type: {style} | Tier: {tier}\n"
            f"Activities: {activities}\n"
            f"{age_str} | {cost_str}\n"
            f"Description: {desc}\n"
        )
    return "\n---\n".join(lines)


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
    ai_asked_question = last_assistant_msg.rstrip().endswith('?')

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
        'i am from', "i'm from", 'we are in', "we're in", 'located in',
    ]
    is_location_reply = any(w in text_lower_check for w in location_words)

    # ── Layer 1: Fast conflict signals ──────────────────────────────────────
    # Detect signals that suggest a NEW child / NEW search
    new_search_signals = []

    if last_filters:
        prev_gender  = (last_filters.get('gender') or '').lower()
        prev_age     = last_filters.get('age')

        # Gender flip signals
        boy_words  = ['son', 'boy', 'his camp', 'for him', 'my boy', 'male', 'brother']
        girl_words = ['daughter', 'girl', 'her camp', 'for her', 'my girl', 'female', 'sister']

        if prev_gender == 'girls' and any(w in text_lower_check for w in boy_words):
            new_search_signals.append('gender_flip')
        if prev_gender == 'boys' and any(w in text_lower_check for w in girl_words):
            new_search_signals.append('gender_flip')

        # Age conflict signals — extract age from new message
        import re as _re
        age_match = _re.search(r'(\d+)[\s-]?year', text_lower_check)
        new_age = int(age_match.group(1)) if age_match else None
        if new_age and prev_age:
            # Flag if ages are far apart (>4 years) suggesting different child
            if abs(new_age - prev_age) > 4:
                new_search_signals.append('age_conflict')

        # Explicit new-search phrases
        explicit_new = [
            'different camp', 'separate search', 'also looking for', 'my other child',
            'another child', 'my other kid', 'different child', 'for my son', 'for my daughter',
            'find me a camp for my', 'looking for a camp for my',
        ]
        if any(p in text_lower_check for p in explicit_new):
            new_search_signals.append('explicit_new')

    # ── Layer 2: Gemini conflict check (only when signals are ambiguous) ────
    # Fires when we have SOME signals but not a clear-cut case
    search_intent = 'SAME'  # default

    if last_filters and new_search_signals:
        # Strong enough signals — classify as DIFFERENT without extra Gemini call
        if 'gender_flip' in new_search_signals or 'explicit_new' in new_search_signals:
            search_intent = 'DIFFERENT'
        elif len(new_search_signals) >= 2:
            search_intent = 'DIFFERENT'
        else:
            # Ambiguous — use a fast Gemini classification call
            classify_system = (
                "You are classifying a camp search conversation. "
                "Reply with exactly one word: SAME, DIFFERENT, or CLARIFY."
            )
            classify_prompt = (
                f"Previous search filters: {last_filters}\n"
                f"New user message: \"{user_text}\"\n\n"
                "Is the new message searching for the SAME child (refinement/follow-up), "
                "a DIFFERENT child (new search), or is it UNCLEAR which child?"
            )
            classification = call_gemini(
                classify_system, classify_prompt,
                config['GEMINI_API_KEY'], max_tokens=10
            ).strip().upper()
            if classification in ('DIFFERENT', 'SAME', 'CLARIFY'):
                search_intent = classification

    # ── Route based on intent ────────────────────────────────────────────────
    is_new_child = (search_intent == 'DIFFERENT')
    needs_clarification = (search_intent == 'CLARIFY')

    if needs_clarification:
        # Return a clarification question immediately — don't search yet
        elapsed = time.time() - start
        prior_summary = []
        if last_filters.get('gender'):   prior_summary.append(last_filters['gender'] + '-only')
        if last_filters.get('style'):    prior_summary.append(last_filters['style'])
        if last_filters.get('age'):      prior_summary.append(f"age {last_filters['age']}")
        if last_filters.get('activity'): prior_summary.append(last_filters['activity'])
        prior_str = ', '.join(prior_summary) if prior_summary else 'your previous search'
        return (
            f"Just to make sure I find the right fit — is this for the same child "
            f"({prior_str}), or are you searching for a different child?",
            elapsed,
            last_filters
        )

    # Step 2: Extract / merge filters
    # ─────────────────────────────────────────────────────────────────────────
    if is_affirmative and last_filters and not is_new_child:
        # Pure "yes/sure" — reuse filters as-is
        filters = {k: v for k, v in last_filters.items() if k in ('province', 'region', 'activity', 'style', 'gender', 'age')}
        combined = user_text

    elif is_new_child or not last_filters:
        # New child / fresh search — extract clean filters, no inheritance
        filters = extract_filters(user_text, config['GEMINI_API_KEY'])
        combined = user_text

    elif ai_asked_question or is_pure_refinement or is_location_reply:
        # AI asked a follow-up OR user is adding detail to SAME search
        # Merge: new filters override old, but old filters fill any gaps
        new_filters = extract_filters(user_text, config['GEMINI_API_KEY'])
        filters = {**last_filters, **{k: v for k, v in new_filters.items() if v is not None}}
        combined = user_text

    else:
        # Default: fresh extraction, no context
        filters = extract_filters(user_text, config['GEMINI_API_KEY'])
        combined = user_text

    # Step 3: Fetch matching camps from camps_clean
    camps, province, region, fallback = search_camps(filters, config)

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

    if not camps:
        camp_context = "No camps found in the database matching these criteria."
    else:
        camp_context = format_camp_context(camps)

    # Step 5: Build fallback note for Gemini
    fallback_note = ""
    activity_label = filters.get('activity','') or ''
    if fallback == 'no_activity_in_region':
        fallback_note = (f"IMPORTANT: No camps specifically offering '{activity_label}' were found near {region}. "
                        f"The camps shown are in {region} but do NOT specifically offer '{activity_label}'. "
                        f"Tell the user honestly there are no '{activity_label}' camps in this area "
                        f"and suggest they contact these local camps to ask about accommodations, "
                        f"or suggest broadening their search.")
    elif fallback == 'no_activity_in_province':
        fallback_note = (f"IMPORTANT: No camps specifically offering '{activity_label}' were found in {province}. "
                        f"The camps shown are in {province} but do NOT offer '{activity_label}'. "
                        f"Be honest about this — do not suggest unrelated camps 'might' offer it. "
                        f"Suggest the user contact camps directly or broaden their search.")
    elif fallback == 'province_only':
        fallback_note = (f"IMPORTANT: No '{activity_label}' camps found in {province}. "
                        f"Showing other camps in {province}. Be honest about the mismatch.")
    elif fallback == 'no_match':
        fallback_note = "Note: No exact matches found, showing top available member camps."

    # Step 6: Gemini writes the full consultant response
    user_name  = filters.get('name', '')
    greeting   = f"The user's name is {user_name}. Address them by name." if user_name else ""
    new_child_note = (
        "IMPORTANT: The user has switched to searching for a DIFFERENT child. "
        "Naturally acknowledge this at the start of your response — e.g. 'Switching gears for your younger one!' or 'Got it, a new search for your son!' — keep it brief and warm, then go straight into results."
    ) if is_new_child else ""

    system_prompt = f"""You are a trusted Canadian camp consultant at camps.ca and ourkids.net — the kind who has personally visited these camps and knows what makes each one special.
You speak to parents like a knowledgeable friend: warm, direct, and confident. No filler phrases, no corporate tone.
Every recommendation feels personal and specific — never generic.
{greeting}
{new_child_note}
STRICT RULES:
1. Only recommend camps from the provided database list — never invent or assume details
2. Use ONLY the description text provided for each camp — do not add information from your training
3. Always format camp names as clickable markdown links: [Camp Name](url)
4. If the user's query is ambiguous (e.g. "vegetarian camps" could mean a camp type OR dietary need),
   and no camps were found, ask a clarifying follow-up question rather than assuming one interpretation.
5. Never show gender-filtered results as co-ed. If gender=girls, all results are girls-only camps.
4. If fallback note says camps don't match request, say so honestly — never force relevance
5. All camps are verified members of camps.ca network
6. Never say a camp "might" offer something unless it's in the provided description"""

    user_prompt = f"""User request: {user_text}

Extracted search criteria: {filters}

{fallback_note}

Available camps from our verified member database:
{camp_context}

Please provide a personalized consultant response.
CRITICAL RULES:
- Only recommend camps that genuinely match what the user asked for
- If the fallback note says camps do NOT match the request, say so honestly
- Never suggest an unrelated camp "might" offer something it clearly doesn't
- For ambiguous searches (vegetarian, vegan, kosher etc): ask a clarifying question — don't assume it means dietary need vs. camp type
- Include for each RELEVANT camp:
- Format EVERY camp exactly like this — no exceptions:
  * **[Camp Name](url)**
     * Location: Region, Province
     * Why it fits: [1-2 sentences max — warm, specific, written like a trusted friend recommending this program. Draw from the program description in MATCHING_PROGRAMS (after the ' -- '). Lead with what makes this program stand out for the child, not a dry list of features. If no description available, use the camp's Description field.]
     * Ages: X-Y | Cost: $min-$max/week
     * Type: Day Camp or Overnight Camp
- URL RULES — use in this priority order:
  1. If SESSION_COUNT = 1: use SESSION_URL as the link (links directly to the specific program)
  2. If SESSION_COUNT > 1: use MARKDOWN_LINK (links to the camp's main page)
- MATCHING_PROGRAMS format is: "sessionID:program name (ages X-Y) -- program description"
  The ' -- ' separator divides the program label from its description text
- Use the description text (after ' -- ') as your source for Why it fits — rewrite it in your own warm voice, don't copy it verbatim
- If multiple programs match, pick the strongest description and mention the count naturally: "3 cheerleading programs including..."
- Keep Why it fits to 1-2 punchy sentences. No corporate language. Sound like you've seen these programs firsthand.
- EXCLUSION RULE: If a camp's MATCHING_PROGRAMS is empty or null, exclude it from results entirely
- List ALL camps that have matching programs
- After the full list, end with ONE short question to help narrow down further — e.g. "Want me to filter by age or location?" This question MUST end with a single '?' and nothing after it.
- CRITICAL: Never ask multiple questions. One question maximum, at the very end."""

    response = call_gemini(system_prompt, user_prompt, config["GEMINI_API_KEY"], max_tokens=4000)

    if not response:
        # Fallback to simple formatted list if Gemini fails
        lines = []
        for c in camps[:5]:
            url  = c.get('camp_url','')
            name = c.get('camp_name','')
            region_str = f"{c.get('region','')}, {c.get('province','')}"
            style = 'Day Camp' if c.get('camp_style') == 'day' else 'Overnight Camp'
            tier  = c.get('listing_tier','')
            acts  = c.get('activities','')
            cost  = f"${c.get('cost_min','')}–${c.get('cost_max','')}/wk" if c.get('cost_min') else ""
            lines.append(f"- **[{name}]({url})** — {style}, {region_str} [{tier}] | {acts} | {cost}")
        response = f"Here are matching camps:\n" + "\n".join(lines)

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
with st.sidebar:
    st.markdown('<div class="sb-head">🔍 Camp Search Consultant</div>', unsafe_allow_html=True)

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

# ── Consultation form handler ─────────────────────────────────────────────────
if submitted and contact_name and region_camp:
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
