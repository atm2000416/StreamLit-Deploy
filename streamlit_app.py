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
# QUERY CLASSIFICATION
# ═════════════════════════════════════════════
def classify_query(user_text):
    """Route queries — default to Case1 (SQL) for most camp searches"""
    text_lower = user_text.lower()

    # Case2: only for detailed descriptive questions about a specific camp
    case2_keywords = ['describe', 'tell me about', 'amenities', 'facilities', 'what is it like']
    if any(kw in text_lower for kw in case2_keywords):
        return "Case2"

    # Everything else goes to Case1 (SQL keyword search)
    return "Case1"

# ═════════════════════════════════════════════
# CASE 1: SQL AGENT
# ═════════════════════════════════════════════
def run_case1(user_text, _config):
    """Query camps using sessions table with specialty codes and proper province/age/cost filtering"""
    from sqlalchemy import create_engine, text

    text_lower = user_text.lower()

    # Specialty code mapping based on actual sessions data
    specialty_map = {
        'stem': [268, 18, 50, 67, 160, 180],
        'science': [268, 50, 18],
        'technology': [268, 18, 180],
        'coding': [18, 180, 268],
        'robotics': [67, 268, 160],
        'computer': [18, 180, 268],
        'engineering': [160, 268, 50],
        'math': [129, 268, 18],
        'arts': [9, 172],
        'art': [9, 172],
        'music': [37],
        'guitar': [37],
        'dance': [22],
        'theatre': [172],
        'drama': [172],
        'film': [172],
        'soccer': [54],
        'basketball': [12],
        'tennis': [66],
        'racquet': [66],
        'equestrian': [30],
        'riding': [30],
        'horse': [30],
        'cooking': [133],
        'chef': [133],
        'aerospace': [50],
        'leadership': [79, 181],
        'outdoor': [181, 79],
        'adventure': [181, 79],
        'hockey': [188],
        'fitness': [188],
        'sports': [188, 12, 54, 66],
        'french': [314],
        'language': [314],
        'academic': [314, 129],
    }

    # City to province mapping
    city_to_province = {
        'vancouver': 'British Columbia', 'victoria': 'British Columbia',
        'kelowna': 'British Columbia', 'surrey': 'British Columbia',
        'toronto': 'Ontario', 'ottawa': 'Ontario', 'hamilton': 'Ontario',
        'mississauga': 'Ontario', 'brampton': 'Ontario', 'london': 'Ontario',
        'montreal': 'Quebec', 'quebec city': 'Quebec', 'laval': 'Quebec',
        'calgary': 'Alberta', 'edmonton': 'Alberta',
        'winnipeg': 'Manitoba', 'saskatoon': 'Saskatchewan',
        'regina': 'Saskatchewan', 'halifax': 'Nova Scotia',
        'fredericton': 'New Brunswick', 'moncton': 'New Brunswick',
    }

    # City/region to specific mapping MIDs for more precise filtering
    city_to_mids = {
        'ottawa': [172, 173, 174, 175, 176, 177, 341, 6280, 6281],
        'toronto': [100, 6084, 6085, 6086, 6087, 6088, 6089, 6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098],
        'vancouver': [281, 340, 1030, 1215],
        'calgary': [390, 391, 392],
        'edmonton': [400, 401, 402],
        'montreal': [285, 286, 287, 288, 289, 290, 291],
        'hamilton': [62, 63, 64],
        'waterloo': [44, 45, 46],
        'kingston': [191, 192, 193],
        'barrie': [131, 132, 133],
    }
    provinces = {
        'british columbia': 'British Columbia', 'bc': 'British Columbia',
        'ontario': 'Ontario', 'quebec': 'Quebec', 'alberta': 'Alberta',
        'nova scotia': 'Nova Scotia', 'new brunswick': 'New Brunswick',
        'manitoba': 'Manitoba', 'saskatchewan': 'Saskatchewan',
        'newfoundland': 'Newfoundland and Labrador',
        'prince edward island': 'Prince Edward Island', 'pei': 'Prince Edward Island',
        'yukon': 'Yukon', 'northwest territories': 'Northwest Territories',
        'nunavut': 'Nunavut'
    }
    location_filter = None
    city_mids = []
    for keyword, value in provinces.items():
        if keyword in text_lower:
            location_filter = value
            break
    if not location_filter:
        for city, province in city_to_province.items():
            if city in text_lower:
                location_filter = province
                city_mids = city_to_mids.get(city, [])
                break
    else:
        # Also check for specific city within a named province
        for city, mids in city_to_mids.items():
            if city in text_lower:
                city_mids = mids
                break

    # Specialty codes from keywords
    specialty_codes = []
    for keyword, codes in specialty_map.items():
        if keyword in text_lower:
            specialty_codes.extend(codes)
    specialty_codes = list(set(specialty_codes))

    # Age filter
    age_filter = None
    import re
    age_match = re.search(r'(\d+)\s*(?:year|yr|yo|-year)', text_lower)
    if age_match:
        age_filter = int(age_match.group(1))
    elif 'teenager' in text_lower or 'teen' in text_lower:
        age_filter = 14
    elif 'toddler' in text_lower:
        age_filter = 4
    elif 'child' in text_lower or 'kid' in text_lower:
        age_filter = 8

    # Cost filter — require $ sign to avoid matching age numbers
    cost_filter = None
    cost_match = re.search(r'\$(\s*\d+)', text_lower)
    if cost_match:
        cost_filter = int(cost_match.group(1).strip())

    # Day/overnight filter
    style_filter = None
    if 'overnight' in text_lower or 'sleepaway' in text_lower or 'residential' in text_lower:
        style_filter = 'double'
    elif 'day camp' in text_lower:
        style_filter = 'single'

    def build_query(province, codes, age, cost, style, limit=10, mids=None):
        conditions = ["c.status = 1", "s.specialty != 0"]
        params = {}

        if codes:
            placeholders = ", ".join([f":sc{i}" for i, _ in enumerate(codes)])
            conditions.append(f"(s.specialty IN ({placeholders}) OR s.specialty2 IN ({placeholders}))")
            for i, code in enumerate(codes):
                params[f'sc{i}'] = code

        if age:
            conditions.append("s.age_from <= :age AND s.age_to >= :age")
            params['age'] = age

        if cost:
            conditions.append("s.cost_from <= :cost")
            params['cost'] = cost

        if style:
            conditions.append("c.listingClass = :style")
            params['style'] = style

        if province:
            conditions.append("m.`grouping` = :province")
            params['province'] = province
            if mids:
                mid_placeholders = ", ".join([f":mid{i}" for i, _ in enumerate(mids)])
                conditions.append(f"m.mid IN ({mid_placeholders})")
                for i, mid in enumerate(mids):
                    params[f'mid{i}'] = mid
            from_clause = """camp_directory.camps c
                JOIN camp_directory.sessions s ON s.cid = c.cid
                JOIN camp_directory.extra_locations el ON el.cid = c.cid
                JOIN common_update.mapping m ON m.mid = el.locations"""
        else:
            from_clause = """camp_directory.camps c
                JOIN camp_directory.sessions s ON s.cid = c.cid"""

        where = " AND ".join(conditions)
        sql = f"""SELECT DISTINCT c.cid, c.camp_name, c.listingClass, c.eListingType, c.prettyURL,
                s.age_from, s.age_to, s.cost_from, s.cost_to,
                {'m.p_region, m.`grouping` as province' if province else 'NULL as p_region, NULL as province'}
            FROM {from_clause}
            WHERE {where}
            ORDER BY c.eListingType DESC
            LIMIT {limit}"""
        return sql, params

    def format_rows(rows, col_names):
        camp_list = []
        seen = set()
        for row in rows:
            d = dict(zip(col_names, row))
            name = d.get("camp_name", "Unknown")
            if name in seen:
                continue
            seen.add(name)
            province = d.get("province") or ""
            region = d.get("p_region") or ""
            location_str = f"{region}, {province}".strip(", ") if (region or province) else "N/A"
            style = "Day Camp" if d.get("listingClass") == "single" else "Overnight Camp"
            tier = d.get("eListingType", "")
            pretty_url = d.get("prettyURL", "")
            cid = d.get("cid", "")
            age_from = d.get("age_from", "")
            age_to = d.get("age_to", "")
            cost_from = d.get("cost_from", "")
            cost_to = d.get("cost_to", "")
            age_str = f"Ages {age_from}-{age_to}" if age_from and age_to else ""
            cost_str = f"${cost_from}-${cost_to}/wk" if cost_from and cost_to else (f"${cost_from}" if cost_from else "")
            details = " | ".join(filter(None, [age_str, cost_str]))
            if pretty_url and cid:
                full_url = f"https://www.camps.ca/{pretty_url}/{cid}"
                camp_list.append(f"- **{name}** ([camps.ca/{pretty_url}/{cid}]({full_url})) — {style}, {location_str} [{tier}]{(' | ' + details) if details else ''}")
            else:
                camp_list.append(f"- **{name}** — {style}, {location_str} [{tier}]{(' | ' + details) if details else ''}")
        return camp_list

    try:
        engine = create_engine(get_db_uri(_config, _config["DB_CAMP_DIR"]), pool_pre_ping=True)
        with engine.connect() as conn:

            # Try 1: Full search
            sql, params = build_query(location_filter, specialty_codes, age_filter, cost_filter, style_filter, mids=city_mids)
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            col_names = list(result.keys())

            if rows:
                camp_list = format_rows(rows, col_names)
                filters = []
                if location_filter: filters.append(location_filter)
                if specialty_codes: filters.append(", ".join([k for k in specialty_map if any(c in specialty_codes for c in specialty_map[k])][:3]))
                if age_filter: filters.append(f"age {age_filter}")
                if cost_filter: filters.append(f"under ${cost_filter}")
                summary = f" (filters: {', '.join(filters)})" if filters else ""
                return f"Found {len(camp_list)} camp(s){summary}:\n" + "\n".join(camp_list)

            # Try 2: Drop cost filter
            if cost_filter:
                sql, params = build_query(location_filter, specialty_codes, age_filter, None, style_filter, mids=city_mids)
                result = conn.execute(text(sql), params)
                rows = result.fetchall()
                col_names = list(result.keys())
                if rows:
                    camp_list = format_rows(rows, col_names)
                    return f"No camps found under ${cost_filter} — here are the closest matches:\n" + "\n".join(camp_list)

            # Try 3: Drop age + cost, keep specialty + province
            if age_filter or cost_filter:
                sql, params = build_query(location_filter, specialty_codes, None, None, style_filter, mids=city_mids)
                result = conn.execute(text(sql), params)
                rows = result.fetchall()
                col_names = list(result.keys())
                if rows:
                    camp_list = format_rows(rows, col_names)
                    return f"Here are matching camps in {location_filter or 'Canada'}:\n" + "\n".join(camp_list)

            # Try 4: Province only, no specialty
            if specialty_codes and location_filter:
                sql, params = build_query(location_filter, [], None, None, style_filter, mids=city_mids)
                result = conn.execute(text(sql), params)
                rows = result.fetchall()
                col_names = list(result.keys())
                if rows:
                    camp_list = format_rows(rows, col_names)
                    return f"No exact specialty match found in {location_filter} — here are other camps in {location_filter}:\n" + "\n".join(camp_list)

            # Try 5: Specialty across all Canada, no province
            if specialty_codes and location_filter:
                sql, params = build_query(None, specialty_codes, age_filter, None, style_filter)
                result = conn.execute(text(sql), params)
                rows = result.fetchall()
                col_names = list(result.keys())
                if rows:
                    camp_list = format_rows(rows, col_names)
                    return f"No matching camps found in {location_filter} — here are similar camps across Canada:\n" + "\n".join(camp_list)

            # Try 6: Top camps overall
            sql, params = build_query(None, [], None, None, None)
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            col_names = list(result.keys())
            if rows:
                camp_list = format_rows(rows, col_names)
                return f"Here are our top member camps:\n" + "\n".join(camp_list)

            return "No camps found in our database."

    except Exception as e:
        return f"Database query error: {str(e)[:500]}"

# ═════════════════════════════════════════════
# CASE 2: VECTOR SEARCH
# ═════════════════════════════════════════════

def run_case2(user_text, _config):
    """Search Pinecone vector DB directly — no LangChain wrapper"""
    from pinecone import Pinecone

    try:
        pc = Pinecone(api_key=_config["PINECONE_API_KEY"])
        index = pc.Index(_config["INDEX_NAME"], host=_config["INDEX_HOST"])

        # Embed query directly via Pinecone inference API
        embed_response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[user_text],
            parameters={"input_type": "query", "truncate": "END"}
        )
        query_vector = embed_response[0]["values"]

        # Query Pinecone directly
        results = index.query(
            vector=query_vector,
            top_k=5,
            namespace=_config["NAMESPACE"],
            include_metadata=True
        )

        matches = results.get("matches", [])
        if not matches:
            return ""

        texts = []
        for match in matches:
            meta = match.get("metadata", {})
            text = meta.get("text", "") or meta.get("content", "") or str(meta)
            if text:
                texts.append(text)

        return "\n\n".join(texts) if texts else ""

    except Exception as e:
        return ""

# ═════════════════════════════════════════════
# CLIENT-ONLY FILTER WITH URL VALIDATION
# ═════════════════════════════════════════════
def filter_to_clients_only(text, client_camps, user_query):
    """
    Filter response to show only paying client camps
    Add verified URLs beside each camp name
    Replace non-client camps with best alternatives
    """
    patterns = [
        r'\b(?:Camp\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Camp)\b',
    ]
    
    mentioned_camps = set()
    for pattern in patterns:
        mentioned_camps.update(re.findall(pattern, text))
    
    if not mentioned_camps:
        return text
    
    client_camps_found = {}
    non_client_camps = []
    
    for camp in mentioned_camps:
        key = camp.lower().strip()
        if key in client_camps:
            client_camps_found[camp] = client_camps[key]
        else:
            non_client_camps.append(camp)
    
    if non_client_camps:
        prefs = extract_preferences(user_query)
        
        for non_client in non_client_camps:
            alternative = find_best_alternative(non_client, prefs, client_camps, client_camps_found)
            
            if alternative:
                friendly_url = alternative.get('friendly_url', alternative.get('url', ''))
                text = text.replace(
                    non_client,
                    f"**{alternative['name']}** ([{friendly_url}]({alternative['url']})) _(recommended alternative)_"
                )
                client_camps_found[alternative['name']] = alternative
    
    for camp_name, camp_data in client_camps_found.items():
        url = camp_data.get('url', '')
        friendly_url = camp_data.get('friendly_url', url)
        
        if url:
            formatted = f"**{camp_name}** ([{friendly_url}]({url}))"
            text = re.sub(
                rf'\b{re.escape(camp_name)}\b(?!\s*\()',
                formatted,
                text,
                count=1
            )
    
    return text

def extract_preferences(user_query):
    """Extract search preferences from user query"""
    text_lower = user_query.lower()
    
    prefs = {
        'location': None,
        'type': None,
        'age': None,
        'price': None,
        'day_overnight': None
    }
    
    provinces = ['ontario', 'quebec', 'bc', 'british columbia', 'alberta', 
                 'saskatchewan', 'manitoba', 'nova scotia', 'new brunswick']
    for prov in provinces:
        if prov in text_lower:
            prefs['location'] = prov.title()
            break
    
    types = ['stem', 'science', 'sports', 'arts', 'music', 'outdoor', 'adventure', 
             'hockey', 'soccer', 'basketball', 'tech', 'coding', 'robotics']
    for camp_type in types:
        if camp_type in text_lower:
            prefs['type'] = camp_type.upper() if camp_type == 'stem' else camp_type.title()
            break
    
    age_match = re.search(r'(\d+)[\s-]?year', text_lower)
    if age_match:
        prefs['age'] = int(age_match.group(1))
    
    price_match = re.search(r'\$?(\d+)', text_lower)
    if price_match and 'under' in text_lower:
        prefs['price'] = int(price_match.group(1))
    
    if 'overnight' in text_lower or 'sleep' in text_lower:
        prefs['day_overnight'] = 'overnight'
    elif 'day camp' in text_lower or 'day' in text_lower:
        prefs['day_overnight'] = 'day'
    
    return prefs

def find_best_alternative(non_client_name, prefs, client_camps, already_suggested):
    """Find best alternative client camp"""
    def score_camp(camp_data):
        score = 0
        
        if camp_data['name'] in already_suggested:
            return -1
        
        if prefs.get('location') and camp_data.get('location'):
            if prefs['location'].lower() in camp_data['location'].lower():
                score += 50
        
        if prefs.get('type') and camp_data.get('type'):
            if prefs['type'].lower() in camp_data['type'].lower():
                score += 30
        
        if prefs.get('age') and camp_data.get('age_min') and camp_data.get('age_max'):
            age = prefs['age']
            if camp_data['age_min'] <= age <= camp_data['age_max']:
                score += 20
        
        if prefs.get('price') and camp_data.get('price'):
            if camp_data['price'] <= prefs['price']:
                score += 15
        
        if prefs.get('day_overnight') and camp_data.get('day_overnight'):
            if prefs['day_overnight'].lower() in str(camp_data['day_overnight']).lower():
                score += 10
        
        return score
    
    scored_camps = []
    for key, camp in client_camps.items():
        score = score_camp(camp)
        if score > 0:
            scored_camps.append((score, camp))
    
    if scored_camps:
        scored_camps.sort(reverse=True, key=lambda x: x[0])
        return scored_camps[0][1]
    
    return None

# ═════════════════════════════════════════════
# MAIN QUERY PROCESSOR
# ═════════════════════════════════════════════
def process_query(user_text, config, client_camps, chat_history=None):
    """Main query processing pipeline"""
    start_time = time.time()

    # Combine recent user messages for better context across conversation turns
    if chat_history:
        recent_user_msgs = [m["content"] for m in chat_history[-6:] if m["role"] == "user"]
        combined_text = " ".join(recent_user_msgs + [user_text])
    else:
        combined_text = user_text

    case = classify_query(combined_text)

    if case == "Case1":
        answer = run_case1(combined_text, config)
    elif case == "Case2":
        answer = run_case2(combined_text, config)
    else:
        answer = run_case1(combined_text, config)

    elapsed = time.time() - start_time
    return answer, elapsed

# ═════════════════════════════════════════════
# STREAMLIT UI
# ═════════════════════════════════════════════
st.set_page_config(page_title="Camp Discovery", page_icon="🏕️", layout="wide")

st.markdown("""
<style>
    .camp-header {
        text-align: center; padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem; margin-bottom: 2rem; color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="camp-header"><h1>🏕️ Camp Discovery</h1><p>Verified Member Camps</p></div>', unsafe_allow_html=True)

config = get_config()
required = ["GEMINI_API_KEY", "PINECONE_API_KEY", "DB_HOST", "DB_USER", "DB_PASS", "INDEX_HOST"]
missing = [k for k in required if not config.get(k)]

if missing:
    st.error(f"⚠️ Missing configuration: {', '.join(missing)}")
    st.stop()

with st.spinner("Loading member camps..."):
    client_camps = load_client_camps(config)

with st.sidebar:
    st.header("📋 Search Our Members")
    st.markdown("All results include verified links to **camps.ca** or **ourkids.net**")
    st.divider()
    
    with st.expander("📊 Database Stats"):
        st.metric("Member Camps", f"{len(client_camps):,}")
        st.caption("✅ Only paying clients shown")
        st.caption("🔗 All camps have verified URLs")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hi! 🏕️ I'll help you find camps from our **verified member network**.\n\n"
            "💡 *All results include verified links to camps.ca or ourkids.net*\n\n"
            "**Example format:**\n"
            "**Camp Sunshine** ([camps.ca/camp-sunshine](https://camps.ca/camp-sunshine))\n\n"
            "Try: *Show me STEM camps in Ontario for 12-year-olds under $500*"
        )
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Search member camps..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            response, elapsed = process_query(prompt, config, client_camps, chat_history=st.session_state.messages)
            st.markdown(response)
            if elapsed < 3:
                st.caption(f"⚡ {elapsed:.1f}s • Member camps only")
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error = f"❌ Error: {str(e)[:300]}"
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})
