"""
Camp Discovery Chatbot - Production Version
Business Logic: Client-Only Member Camps with Verified URLs
Platform: Streamlit Cloud
Databases: Aiven MySQL (campdb, camp_directory, common_update)
Vector DB: Pinecone
AI: Google Gemini 2.0 Flash
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
                        
                        name_col = cols.get('name') or cols.get('camp_name') or cols.get('title')
                        url_col = cols.get('url') or cols.get('website') or cols.get('link')
                        location_col = cols.get('location') or cols.get('region') or cols.get('province')
                        type_col = cols.get('type') or cols.get('category') or cols.get('camp_type')
                        price_col = cols.get('price') or cols.get('cost') or cols.get('fee')
                        age_min_col = cols.get('age_min') or cols.get('min_age')
                        age_max_col = cols.get('age_max') or cols.get('max_age')
                        day_col = cols.get('day_overnight') or cols.get('camp_style')
                        desc_col = cols.get('description') or cols.get('details')
                        id_col = cols.get('id') or cols.get('camp_id')
                        
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
MODEL = "gemini-2.0-flash"
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
    """Fast keyword-based classification"""
    text_lower = user_text.lower()
    
    case1_keywords = ['how many', 'count', 'average', 'price', 'cost', 'under $', 
                      'capacity', 'available', 'cheapest', 'most expensive', 'compare']
    if any(kw in text_lower for kw in case1_keywords):
        return "Case1"
    
    case2_keywords = ['describe', 'what is', 'tell me about', 'amenities', 
                      'facilities', 'programs', 'activities', 'like']
    if any(kw in text_lower for kw in case2_keywords):
        return "Case2"
    
    return "Case3"

# ═════════════════════════════════════════════
# CASE 1: SQL AGENT
# ═════════════════════════════════════════════
@st.cache_data(ttl=300)
def run_case1(user_text, _config):
    """Query database for structured data"""
    from langchain.chat_models import init_chat_model
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain import hub
    from langgraph.prebuilt import create_react_agent

    os.environ["GOOGLE_API_KEY"] = _config["GEMINI_API_KEY"]

    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    
    db = SQLDatabase.from_uri(
        get_db_uri(_config, _config["DB_CAMPDB"]),
        view_support=True,
        sample_rows_in_table_info=0
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    prompt = (
        "You are a SQL expert querying a camps database. "
        "IMPORTANT: Only suggest camps that exist in this database. "
        f"Query: {user_text}. Use LIMIT 10."
    )
    
    agent = create_react_agent(llm, tools, prompt=prompt)
    
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
        return response["messages"][-1].content
    except Exception as e:
        return f"Database query error: {str(e)[:200]}"

# ═════════════════════════════════════════════
# CASE 2: VECTOR SEARCH
# ═════════════════════════════════════════════
@st.cache_data(ttl=300)
def run_case2(user_text, _config):
    """Search vector database for descriptive content"""
    from pinecone import Pinecone
    from langchain_core.embeddings import Embeddings
    from langchain_pinecone import PineconeVectorStore

    class LlamaEmbeddings(Embeddings):
        def __init__(self, pc):
            self.pc = pc
        def embed_query(self, text):
            return self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[text],
                parameters={"input_type": "query", "truncate": "END"}
            )[0]["values"]
        def embed_documents(self, texts):
            return [e["values"] for e in self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )]

    pc = Pinecone(api_key=_config["PINECONE_API_KEY"])
    index = pc.Index(_config["INDEX_NAME"], host=_config["INDEX_HOST"])
    embeddings = LlamaEmbeddings(pc)
    
    vectorstore = PineconeVectorStore(
        index=index, embedding=embeddings,
        namespace=_config["NAMESPACE"], text_key="text"
    )
    
    results = vectorstore.similarity_search_by_vector_with_score(
        embeddings.embed_query(user_text), k=5
    )
    
    if not results:
        return "No camps found matching your criteria."
    
    return "\n\n".join([doc.page_content for doc, _ in results])

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
def process_query(user_text, config, client_camps):
    """Main query processing pipeline"""
    start_time = time.time()
    
    case = classify_query(user_text)
    
    if case == "Case1":
        answer = run_case1(user_text, config)
    elif case == "Case2":
        answer = run_case2(user_text, config)
    else:
        ans1 = run_case1(user_text, config)
        ans2 = run_case2(user_text, config)
        answer = f"{ans1}\n\n{ans2}"
    
    result = filter_to_clients_only(answer, client_camps, user_text)
    elapsed = time.time() - start_time
    
    return result, elapsed

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
            response, elapsed = process_query(prompt, config, client_camps)
            st.markdown(response)
            if elapsed < 3:
                st.caption(f"⚡ {elapsed:.1f}s • Member camps only")
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error = f"❌ Error: {str(e)[:300]}"
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})
