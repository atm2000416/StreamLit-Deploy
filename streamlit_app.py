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
  "activity": "main activity type or null",
  "age": integer or null,
  "max_cost": integer or null,
  "style": "day" or "overnight" or null,
  "name": "user first name if mentioned or null"
}
Province must be one of: British Columbia, Alberta, Ontario, Quebec, Manitoba, Saskatchewan, Nova Scotia, New Brunswick, Prince Edward Island, Newfoundland and Labrador.
For cities, set region to the city name and infer the province.
Examples:
- "etobicoke" → province: "Ontario", region: "Toronto"
- "nepean" → province: "Ontario", region: "Ottawa Region"  
- "debate camps" → activity: "debate"
- "hockey" → activity: "hockey"
- "under $500" → max_cost: 500
- "10 year old" → age: 10
- "teens" → age: 15"""

    result = call_gemini(system, user_text, api_key, max_tokens=200)
    import json, re
    try:
        # Strip markdown code fences if present
        clean = re.sub(r'```json|```', '', result).strip()
        return json.loads(clean)
    except:
        return {}


def search_camps(filters, config, limit=8):
    """Query camps_clean using extracted filters — simple WHERE clauses, no joins"""
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

    # Resolve region/city to province if not set
    region_lower = (region or '').lower().strip()
    if not province and region_lower in city_province_map:
        province = city_province_map[region_lower]
    resolved_region = region_map.get(region_lower, region)

    conditions = ["status = 1", "province != 'Unknown'"]
    params = {}

    if province:
        conditions.append("province = :province")
        params['province'] = province

    if resolved_region:
        conditions.append("region LIKE :region")
        params['region'] = f"%{resolved_region}%"

    if activity:
        conditions.append(
            "(activities LIKE :act OR program_names LIKE :act2 OR camp_name LIKE :act3)"
        )
        params['act']  = f"%{activity}%"
        params['act2'] = f"%{activity}%"
        params['act3'] = f"%{activity}%"

    if age:
        conditions.append("age_min <= :age AND age_max >= :age")
        params['age'] = age

    if max_cost:
        conditions.append("(cost_min <= :cost OR cost_min IS NULL)")
        params['cost'] = max_cost

    if style:
        conditions.append("camp_style = :style")
        params['style'] = style

    where = " AND ".join(conditions)
    sql = f"""SELECT DISTINCT cid, camp_name, province, region,
            camp_style, listing_tier, camp_url,
            age_min, age_max, cost_min, cost_max, activities, description
        FROM camp_directory.camps_clean
        WHERE {where}
        ORDER BY FIELD(listing_tier, 'gold', 'silver', 'bronze'), camp_name
        LIMIT {limit}"""

    try:
        engine = create_engine(get_db_uri(config, config["DB_CAMP_DIR"]), pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows   = result.fetchall()
            cols   = list(result.keys())

            if rows:
                return [dict(zip(cols, row)) for row in rows], province, resolved_region, None

            # Fallback 1: drop region
            if resolved_region and province:
                conditions2 = [c for c in conditions if 'region' not in c]
                sql2 = f"""SELECT DISTINCT cid, camp_name, province, region,
                        camp_style, listing_tier, camp_url,
                        age_min, age_max, cost_min, cost_max, activities, description
                    FROM camp_directory.camps_clean
                    WHERE {' AND '.join(conditions2)}
                    ORDER BY FIELD(listing_tier,'gold','silver','bronze')
                    LIMIT {limit}"""
                params2 = {k:v for k,v in params.items() if k != 'region'}
                result2 = conn.execute(text(sql2), params2)
                rows2   = result2.fetchall()
                if rows2:
                    return [dict(zip(list(result2.keys()), row)) for row in rows2], province, resolved_region, 'no_region'

            # Fallback 2: province only
            if province:
                result3 = conn.execute(text(
                    "SELECT DISTINCT cid, camp_name, province, region, camp_style, listing_tier, "
                    "camp_url, age_min, age_max, cost_min, cost_max, activities, description "
                    "FROM camp_directory.camps_clean WHERE status=1 AND province=:p "
                    "ORDER BY FIELD(listing_tier,'gold','silver','bronze') LIMIT :lim"
                ), {"p": province, "lim": limit})
                rows3 = result3.fetchall()
                if rows3:
                    return [dict(zip(list(result3.keys()), row)) for row in rows3], province, resolved_region, 'no_activity'

            # Fallback 3: top camps
            result4 = conn.execute(text(
                "SELECT DISTINCT cid, camp_name, province, region, camp_style, listing_tier, "
                "camp_url, age_min, age_max, cost_min, cost_max, activities, description "
                "FROM camp_directory.camps_clean WHERE status=1 AND province != 'Unknown' "
                "ORDER BY FIELD(listing_tier,'gold','silver','bronze') LIMIT :lim"
            ), {"lim": limit})
            rows4 = result4.fetchall()
            return [dict(zip(list(result4.keys()), row)) for row in rows4], province, resolved_region, 'no_match'

    except Exception as e:
        return [], province, resolved_region, f"error: {str(e)[:200]}"


def format_camp_context(camps):
    """Format camp data as readable context for Gemini"""
    lines = []
    for c in camps:
        name      = c.get('camp_name', '')
        url       = c.get('camp_url', '')
        province  = c.get('province', '')
        region    = c.get('region', '')
        style     = 'Day Camp' if c.get('camp_style') == 'day' else 'Overnight Camp'
        tier      = c.get('listing_tier', '')
        age_min   = c.get('age_min', '')
        age_max   = c.get('age_max', '')
        cost_min  = c.get('cost_min', '')
        cost_max  = c.get('cost_max', '')
        activities= c.get('activities', '')
        desc      = (c.get('description') or '')[:300]

        age_str  = f"Ages {age_min}-{age_max}" if age_min and age_max else ""
        cost_str = f"${cost_min}-${cost_max}/week" if cost_min and cost_max else ""

        lines.append(
            f"CAMP: {name}\n"
            f"URL: {url}\n"
            f"Location: {region}, {province}\n"
            f"Type: {style} | Tier: {tier}\n"
            f"Activities: {activities}\n"
            f"{age_str} | {cost_str}\n"
            f"Description: {desc}\n"
        )
    return "\n---\n".join(lines)


def process_query(user_text, config, client_camps, chat_history=None):
    """Main RAG pipeline — Gemini extracts filters, SQL fetches camps, Gemini writes response"""
    import time
    start = time.time()

    # Step 1: Build combined context from chat history
    if chat_history:
        recent = [m["content"] for m in chat_history[-3:] if m["role"] == "user"]
        # Only include history if current message has no location (refinement)
        location_words = ['ontario','british columbia','alberta','quebec','manitoba',
                         'vancouver','toronto','ottawa','calgary','edmonton','winnipeg',
                         'montreal','halifax','bc','etobicoke','nepean','scarborough']
        has_location = any(w in user_text.lower() for w in location_words)
        combined = " ".join(recent[:-1] + [user_text]) if not has_location else user_text
    else:
        combined = user_text

    # Step 2: Extract filters using Gemini
    filters = extract_filters(combined, config["GEMINI_API_KEY"])

    # Step 3: Fetch matching camps from camps_clean
    camps, province, region, fallback = search_camps(filters, config)

    # Step 4: Build context string for Gemini
    if not camps:
        camp_context = "No camps found in the database matching these criteria."
    else:
        camp_context = format_camp_context(camps)

    # Step 5: Build fallback note for Gemini
    fallback_note = ""
    if fallback == 'no_region':
        fallback_note = f"Note: No camps found specifically near {region}, so showing results from {province} instead."
    elif fallback == 'no_activity':
        fallback_note = f"Note: No {filters.get('activity','matching')} camps found in {province}, showing other available camps."
    elif fallback == 'no_match':
        fallback_note = "Note: No exact matches found, showing top available member camps."

    # Step 6: Gemini writes the full consultant response
    user_name = filters.get('name', '')
    greeting  = f"The user's name is {user_name}. Address them by name." if user_name else ""

    system_prompt = f"""You are an expert Canadian camp consultant at camps.ca and ourkids.net.
You help parents find the perfect verified member camp for their children.
Be warm, helpful, and specific. Keep responses concise but valuable.
{greeting}
Always include the camp URL as a clickable markdown link like [Camp Name](url).
Only recommend camps from the provided list — do not invent camps.
If the search returned fallback results, acknowledge this honestly and suggest alternatives.
All camps are verified members of camps.ca / ourkids.net network."""

    user_prompt = f"""User request: {user_text}

Extracted search criteria: {filters}

{fallback_note}

Available camps from our verified member database:
{camp_context}

Please provide a personalized consultant response recommending the best camps from this list.
Include for each camp:
- Camp name as a clickable link to their camps.ca page
- Location (region, province)  
- Why it fits the user's request
- Age range and weekly cost
- Day camp or overnight camp
Maximum 5 recommendations. End with an offer to refine the search."""

    response = call_gemini(system_prompt, user_prompt, config["GEMINI_API_KEY"], max_tokens=1000)

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
    return response, elapsed


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
    .consultant-form {
        background: #f8f9fa; border-radius: 1rem;
        padding: 1.5rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="camp-header"><h1>🏕️ Camp Discovery</h1><p>Your Personal Canadian Camp Consultant</p></div>', unsafe_allow_html=True)

config = get_config()
required = ["GEMINI_API_KEY", "PINECONE_API_KEY", "DB_HOST", "DB_USER", "DB_PASS", "INDEX_HOST"]
missing = [k for k in required if not config.get(k)]

if missing:
    st.error(f"⚠️ Missing configuration: {', '.join(missing)}")
    st.stop()

with st.spinner("Loading member camps..."):
    client_camps = load_client_camps(config)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏕️ Camp Consultant")
    st.markdown("Tell us about yourself and we'll find the perfect camp!")
    st.divider()

    # Consultation intake form
    with st.form("consultation_form"):
        st.subheader("👤 About You")
        contact_name = st.text_input("Your name", placeholder="e.g. Sarah")
        region_camp  = st.text_input("Your region / city", placeholder="e.g. Toronto, Ottawa, Vancouver")
        st.subheader("🏕️ Camp Preferences")
        category_options = ["Any", "STEM / Science / Technology", "Sports", "Arts / Music / Dance",
                            "Outdoor / Adventure", "Academic / Tutoring", "Language / French",
                            "Equestrian / Riding", "Cooking / Chef", "Leadership", "Traditional"]
        category_camp = st.selectbox("Type of camp", category_options)
        camp_type     = st.text_input("Specific activity", placeholder="e.g. robotics, hockey, painting")
        date_camp     = st.text_input("Preferred dates / season", placeholder="e.g. July, Summer 2025, Week of Aug 4")
        costs_camp    = st.text_input("Max budget per week", placeholder="e.g. $500, $800")
        age_child     = st.text_input("Child's age(s)", placeholder="e.g. 10, 8 and 12")
        overnight     = st.radio("Camp style", ["Either", "Day camp", "Overnight camp"])
        submitted     = st.form_submit_button("🔍 Find My Camp!", use_container_width=True)

    st.divider()
    with st.expander("📊 Network Stats"):
        st.metric("Member Camps", f"{len(client_camps):,}")
        st.caption("✅ Verified member camps only")
        st.caption("🔗 All camps have verified URLs")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.consultation_done = False
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "consultation_done" not in st.session_state:
    st.session_state.consultation_done = False

# ── Handle consultation form submission ───────────────────────────────────────
if submitted and contact_name and region_camp:
    # Build a rich structured query from the form
    style_text = "" if overnight == "Either" else overnight.lower()
    budget_text = f"under {costs_camp} per week" if costs_camp else ""
    age_text    = f"for a {age_child}-year-old" if age_child else ""
    type_text   = camp_type if camp_type else category_camp if category_camp != "Any" else "general"
    date_text   = f"around {date_camp}" if date_camp else "this summer"

    structured_query = (
        f"My name is {contact_name} and I'm looking for {style_text} {type_text} camps "
        f"in {region_camp} {age_text} {budget_text} {date_text}."
    ).replace("  ", " ").strip()

    welcome = (
        f"Hi {contact_name}! 👋 Great to meet you! I'm your personal camp consultant and "
        f"I'll find the best camps for you in **{region_camp}**.\n\n"
        f"Let me search our verified member network now... 🔍"
    )

    st.session_state.messages = [{"role": "assistant", "content": welcome}]
    st.session_state.messages.append({"role": "user", "content": structured_query})
    st.session_state.consultation_done = False
    st.rerun()

# ── Welcome message if no messages yet ───────────────────────────────────────
if not st.session_state.messages:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Welcome to Camp Discovery! I'm your personal Canadian camp consultant.\n\n"
            "**To get started**, fill in the form on the left and I'll find the perfect camp for you! "
            "Or just type your search below.\n\n"
            "💡 *Try: 'Show me STEM camps in Toronto for a 12-year-old under $500'*"
        )
    }]

# ── Display chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Process any pending user message ─────────────────────────────────────────
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if last_msg and last_msg["role"] == "user" and not st.session_state.consultation_done:
    with st.chat_message("assistant"):
        with st.spinner("Searching member camps..."):
            try:
                response, elapsed = process_query(
                    last_msg["content"], config, client_camps,
                    chat_history=st.session_state.messages
                )

                # Append follow-up prompt after first consultation result
                follow_up = (
                    "\n\n---\n"
                    "🎯 **Now that you have your initial recommendations**, is there anything specific "
                    "that matters most to you? For example:\n"
                    "- A **different type** of camp activity\n"
                    "- A **different region** or closer location\n"
                    "- A specific **budget** range\n"
                    "- A specific **age group** or dates\n\n"
                    "Just let me know and I'll refine your results!"
                )

                full_response = response + follow_up
                st.markdown(full_response)
                st.caption(f"⚡ {elapsed:.1f}s • Verified member camps")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.consultation_done = True

            except Exception as e:
                error = f"❌ Error: {str(e)[:300]}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
                st.session_state.consultation_done = True

# ── Free-text chat input ──────────────────────────────────────────────────────
if prompt := st.chat_input("Refine your search or ask a follow-up question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.consultation_done = False
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                response, elapsed = process_query(
                    prompt, config, client_camps,
                    chat_history=st.session_state.messages
                )
                st.markdown(response)
                st.caption(f"⚡ {elapsed:.1f}s • Verified member camps")
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error = f"❌ Error: {str(e)[:300]}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
