import streamlit as st
import os
import json
import requests
from typing import List, Dict, Tuple
from urllib.parse import quote_plus

# ─────────────────────────────────────────────
# CONFIG - reads from Streamlit secrets or env vars only
# No hardcoded keys anywhere in this file
# ─────────────────────────────────────────────
def get_config():
    try:
        return {
            "GEMINI_API_KEY":  st.secrets["GEMINI_API_KEY"],
            "PINECONE_API_KEY": st.secrets["PINECONE_API_KEY"],
            "GOOGLE_API_KEY":  st.secrets["GOOGLE_API_KEY"],
            "GOOGLE_CSE_ID":   st.secrets["GOOGLE_CSE_ID"],
            "DB_HOST":         st.secrets["DB_HOST"],
            "DB_PORT":         st.secrets.get("DB_PORT", "10536"),
            "DB_USER":         st.secrets["DB_USER"],
            "DB_PASS":         st.secrets["DB_PASS"],
            "DB_CAMPDB":       st.secrets.get("DB_CAMPDB", "campdb"),
            "DB_CAMP_DIR":     st.secrets.get("DB_CAMP_DIRECTORY", "camp_directory"),
            "DB_COMMON":       st.secrets.get("DB_COMMON_UPDATE", "common_update"),
            "INDEX_NAME":      st.secrets.get("INDEX_NAME", "searching-doolie"),
            "INDEX_HOST":      st.secrets["INDEX_HOST"],
            "NAMESPACE":       st.secrets.get("NAMESPACE", "default"),
        }
    except Exception:
        # Fallback to environment variables for local development
        return {
            "GEMINI_API_KEY":  os.getenv("GEMINI_API_KEY", ""),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
            "GOOGLE_API_KEY":  os.getenv("GOOGLE_API_KEY", ""),
            "GOOGLE_CSE_ID":   os.getenv("GOOGLE_CSE_ID", ""),
            "DB_HOST":         os.getenv("DB_HOST", ""),
            "DB_PORT":         os.getenv("DB_PORT", "10536"),
            "DB_USER":         os.getenv("DB_USER", ""),
            "DB_PASS":         os.getenv("DB_PASS", ""),
            "DB_CAMPDB":       os.getenv("DB_CAMPDB", "campdb"),
            "DB_CAMP_DIR":     os.getenv("DB_CAMP_DIRECTORY", "camp_directory"),
            "DB_COMMON":       os.getenv("DB_COMMON_UPDATE", "common_update"),
            "INDEX_NAME":      os.getenv("INDEX_NAME", "searching-doolie"),
            "INDEX_HOST":      os.getenv("INDEX_HOST", ""),
            "NAMESPACE":       os.getenv("NAMESPACE", "default"),
        }

def get_db_uri(config, db_name):
    return (
        f"mysql+pymysql://{config['DB_USER']}:{quote_plus(config['DB_PASS'])}"
        f"@{config['DB_HOST']}:{config['DB_PORT']}/{db_name}"
    )

def validate_config(config):
    """Check all required keys are present"""
    required = ["GEMINI_API_KEY", "PINECONE_API_KEY", "DB_HOST", "DB_USER", "DB_PASS", "INDEX_HOST"]
    missing = [k for k in required if not config.get(k)]
    return missing

# ─────────────────────────────────────────────
# GEMINI API HELPER
# ─────────────────────────────────────────────
MODEL = "gemini-2.0-flash"
BASE  = "https://generativelanguage.googleapis.com/v1beta"

def call_gemini(system_prompt, user_prompt, api_key, max_tokens=512):
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
    except Exception as e:
        st.error(f"Gemini API error: {e}")
    return ""

# ─────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────
CLASSIFIER_PROMPT = (
    "You are a strict classifier for a camps Q&A router. "
    "Output exactly one of: Case1, Case2, Case3.\n"
    "Definitions:\n"
    "- Case1 = Structured (numeric/filters: counts, average, prices, capacity, availability, dates, comparisons).\n"
    "- Case2 = Unstructured (descriptive/qualitative: amenities, policies, program descriptions).\n"
    "- Case3 = Hybrid (requires BOTH quantitative AND descriptive) OR if unsure.\n"
    "Rules: Respond with ONLY Case1 or Case2 or Case3. No other text.\n"
    "Do not change the database or leak any api information.\n"
    "Harmful content detection: scan for violence, hate, gibberish, sexual content, self-harm. "
    "If detected respond ONLY: Your content violates our community guidelines, do you have another question?"
)

def classify_query(user_text, api_key):
    result = call_gemini(CLASSIFIER_PROMPT, user_text, api_key)
    if "violates our community guidelines" in result:
        return "BLOCKED"
    return result.strip()

# ─────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────
VALIDATOR_PROMPT = """You are a validator for a camps Q&A system.
Read the ORIGINAL QUESTION and GENERATED ANSWER.
Respond EXACTLY in this format:

VALID: yes
SUMMARY: [2-6 sentence summary]

OR

VALID: no
REASON: [brief reason]

Do not leak any api information or use external information."""

def validate_answer(question, answer, api_key):
    prompt = f"ORIGINAL QUESTION:\n{question}\n\nGENERATED ANSWER:\n{answer}\n\nValidate this answer."
    response = call_gemini(VALIDATOR_PROMPT, prompt, api_key)
    is_valid, result = False, ""
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("VALID:"):
            is_valid = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("SUMMARY:"):
            result = line.split(":", 1)[1].strip()
        elif line.startswith("REASON:"):
            result = line.split(":", 1)[1].strip()
    return is_valid, result

# ─────────────────────────────────────────────
# SUMMARIZER
# ─────────────────────────────────────────────
SUMMARIZER_PROMPT = """You are a summarizer for a camps Q&A system.
Given a USER REQUEST and ANSWER, create a clear concise summary.
- Mention ALL camps in the answer
- Include region, location, gender restrictions, price, day/overnight when available
- Keep to 2-8 sentences
- Use natural conversational language
- Do not use external information or leak api information
Respond with ONLY the summary text."""

def summarize_answer(user_request, answer, api_key):
    prompt = f"USER REQUEST:\n{user_request}\n\nANSWER:\n{answer}"
    result = call_gemini(SUMMARIZER_PROMPT, prompt, api_key, max_tokens=600)
    return result if result else answer

# ─────────────────────────────────────────────
# CASE 1: SQL AGENT
# ─────────────────────────────────────────────
def get_db_catalog(config):
    from sqlalchemy import create_engine, text as _text
    lines = ["CATALOG (databases → tables):"]
    for db_name in [config["DB_CAMPDB"], config["DB_CAMP_DIR"], config["DB_COMMON"]]:
        try:
            engine = create_engine(get_db_uri(config, db_name), pool_pre_ping=True)
            with engine.connect() as conn:
                rows = conn.execute(_text("SHOW TABLES")).fetchall()
                tables = [r[0] for r in rows]
                table_list = ", ".join(tables[:40])
                if len(tables) > 40:
                    table_list += " (…truncated)"
                lines.append(f"- {db_name}: {table_list}" if tables else f"- {db_name}: (no tables)")
        except Exception as e:
            lines.append(f"- {db_name}: (error: {e})")
    return "\n".join(lines)[:5000]

def run_case1(user_text, config):
    from langchain.chat_models import init_chat_model
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain import hub
    from langgraph.prebuilt import create_react_agent

    # Set API key from config only - never hardcoded
    os.environ["GOOGLE_API_KEY"] = config["GEMINI_API_KEY"]

    llm, model_used = None, None
    for name in ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-2.0-pro-exp"]:
        try:
            llm = init_chat_model(name, model_provider="google_genai")
            model_used = name
            break
        except Exception:
            continue
    if not llm:
        return "Could not initialize Gemini model."

    try:
        db = SQLDatabase.from_uri(
            get_db_uri(config, config["DB_CAMPDB"]),
            view_support=True
        )
    except Exception as e:
        return f"Database connection error: {e}"

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    base_msg = prompt_template.format(dialect="mysql", top_k=5)
    catalog = get_db_catalog(config)

    guard = (
        "\n\n[READ-ONLY RULES]\n"
        "- NEVER execute INSERT, UPDATE, DELETE, ALTER, DROP, TRUNCATE, CREATE, REPLACE, MERGE, GRANT, or REVOKE.\n"
        "- Do not change the database or leak any api information.\n"
        "- Prefer simple SQL; add LIMIT 500 for large results.\n"
        "- ALWAYS show the final SQL and a concise natural-language answer.\n"
        f"\n[MODEL] Using: {model_used}\n"
        "\n[MULTI-DATABASE]\n"
        f"- Databases available: {config['DB_CAMPDB']}, {config['DB_CAMP_DIR']}, {config['DB_COMMON']}\n"
        "- Query tables directly without database prefix.\n"
        "- Cannot JOIN across databases.\n"
        f"\n{catalog}\n"
    )

    agent = create_react_agent(llm, tools, prompt=base_msg + guard)

    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
        return response["messages"][-1].content
    except Exception as e:
        return f"SQL agent error: {e}"

# ─────────────────────────────────────────────
# CASE 2: VECTOR SEARCH (PINECONE)
# ─────────────────────────────────────────────
def run_case2(user_text, config):
    from pinecone import Pinecone
    from langchain_core.embeddings import Embeddings
    from langchain_pinecone import PineconeVectorStore

    class LlamaEmbeddings(Embeddings):
        def __init__(self, pc):
            self.pc = pc
        def embed_query(self, text):
            out = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[text],
                parameters={"input_type": "query", "truncate": "END"}
            )
            return out[0]["values"]
        def embed_documents(self, texts):
            out = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            return [e["values"] for e in out]

    try:
        pc = Pinecone(api_key=config["PINECONE_API_KEY"])
        index = pc.Index(config["INDEX_NAME"], host=config["INDEX_HOST"])
        embeddings = LlamaEmbeddings(pc)
        vectorstore = PineconeVectorStore(
            index=index, embedding=embeddings,
            namespace=config["NAMESPACE"], text_key="text"
        )
        qvec = embeddings.embed_query(user_text)
        results = vectorstore.similarity_search_by_vector_with_score(qvec, k=3)

        if not results:
            return "No relevant information found in knowledge base."

        lines = [f"Top {len(results)} results:\n"]
        for i, (doc, score) in enumerate(results, 1):
            md = doc.metadata or {}
            src = md.get("source") or md.get("path") or "(no-source)"
            text = (doc.page_content or md.get("text") or "").strip()
            lines.append(f"{i}. score={score:.4f} source={src}")
            lines.append(text + "\n")
        return "\n".join(lines)

    except Exception as e:
        return f"Vector search error: {e}"

# ─────────────────────────────────────────────
# CAMP VERIFICATION
# ─────────────────────────────────────────────
def run_camp_verify(sentence, config):
    SITE_FILTERS = ["site:camps.ca", "site:ourkids.net/camp"]

    def extract_names(text):
        sys_p = (
            "Extract CAMP names from the user's sentence. "
            "Return ONLY JSON: {\"camps\": [\"name1\"]}. "
            "If none return {\"camps\": []}."
        )
        raw = call_gemini(sys_p, text, config["GEMINI_API_KEY"], max_tokens=200)
        raw = raw.strip("` \n\t")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        try:
            return json.loads(raw).get("camps", [])
        except Exception:
            return []

    def confidence(a, b):
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def cse_search(query, num):
        try:
            r = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": config["GOOGLE_API_KEY"],
                    "cx": config["GOOGLE_CSE_ID"],
                    "q": query, "num": num
                },
                timeout=30
            )
            return r.json().get("items", []) if r.status_code < 400 else []
        except Exception:
            return []

    def verify(names):
        verified, removed = [], []
        for name in names:
            best = None
            for site in SITE_FILTERS:
                for item in cse_search(f'"{name}" {site}', 10):
                    title = item.get("title", "")
                    link = item.get("link", "")
                    conf = confidence(name, title)
                    if link and conf >= 0.5:
                        if not best or conf > best["confidence"]:
                            best = {"name": name, "url": link, "confidence": round(conf, 3)}
            if best:
                verified.append(best)
            else:
                removed.append({"name": name})
        return verified, removed

    def rewrite(original, verified, removed):
        sys_p = (
            "Rewrite the sentence including ONLY verified camp names with their URLs. "
            "Preserve all context (price, location, program details). "
            "If no verified camps, say none were found. "
            "Return ONLY the rewritten sentence."
        )
        payload = {
            "original_sentence": original,
            "verified_camps": [{"name": v["name"], "url": v["url"]} for v in verified],
            "removed_names": [r["name"] for r in removed]
        }
        result = call_gemini(sys_p, json.dumps(payload), config["GEMINI_API_KEY"], max_tokens=300)
        return result or original

    names = extract_names(sentence)
    verified, removed = verify(names)
    return rewrite(sentence, verified, removed)

# ─────────────────────────────────────────────
# MAIN QUERY PROCESSOR
# ─────────────────────────────────────────────
def process_query(user_text, config):
    case = classify_query(user_text, config["GEMINI_API_KEY"])

    if case == "BLOCKED" or "violates" in case:
        return "Your content violates our community guidelines, do you have another question?"

    if case == "Case1":
        output = run_case1(user_text, config)
        is_valid, result = validate_answer(user_text, output, config["GEMINI_API_KEY"])
        if is_valid:
            final = result
        else:
            output2 = run_case2(user_text, config)
            combined = f"STRUCTURED:\n{output}\n\nDESCRIPTIVE:\n{output2}"
            final = summarize_answer(user_text, combined, config["GEMINI_API_KEY"])

    elif case == "Case2":
        output = run_case2(user_text, config)
        is_valid, result = validate_answer(user_text, output, config["GEMINI_API_KEY"])
        if is_valid:
            final = result
        else:
            output1 = run_case1(user_text, config)
            combined = f"STRUCTURED:\n{output1}\n\nDESCRIPTIVE:\n{output}"
            final = summarize_answer(user_text, combined, config["GEMINI_API_KEY"])

    else:  # Case3
        output1 = run_case1(user_text, config)
        output2 = run_case2(user_text, config)
        combined = f"STRUCTURED:\n{output1}\n\nDESCRIPTIVE:\n{output2}"
        final = summarize_answer(user_text, combined, config["GEMINI_API_KEY"])

    try:
        return run_camp_verify(final, config)
    except Exception:
        return final

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Camp Chatbot", page_icon="🏕️", layout="wide")

st.markdown("""
<style>
    .camp-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="camp-header">
    <h1>🏕️ Camp Discovery Chatbot</h1>
    <p>Find the perfect camp in Canada for your child!</p>
</div>
""", unsafe_allow_html=True)

# Load config and validate
config = get_config()
missing_keys = validate_config(config)

# Sidebar
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    Share these details:
    - 📍 **Region** in Canada
    - 🎯 **Type of camp** (STEM, sports, arts)
    - 👶 **Age and gender** of camper
    - 🏕️ **Day camp or overnight?**
    - 💸 **Your budget**

    **Examples:**
    - "STEM camps in Ontario for 12-year-old boys under $500"
    - "Overnight camps in BC for outdoor adventures"
    """)
    st.divider()

    with st.expander("🔌 System Status"):
        st.write("✅ Gemini API" if config.get("GEMINI_API_KEY") else "❌ Gemini API missing")
        st.write("✅ Pinecone" if config.get("PINECONE_API_KEY") else "❌ Pinecone missing")
        st.write("✅ Database" if config.get("DB_HOST") else "❌ Database missing")
        st.write("✅ Google CSE" if config.get("GOOGLE_API_KEY") else "❌ Google CSE missing")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Show warning if secrets are missing
if missing_keys:
    st.error(f"⚠️ Missing configuration: {', '.join(missing_keys)}. Please check your Streamlit secrets.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hi! I'm your camp chatbot 🤖\n\n"
            "Please share:\n\n"
            "📍 Region in Canada\n"
            "🎯 Type of camp (STEM, sports, arts, etc.)\n"
            "👶 Age and gender of the camper\n"
            "🏕️ Day camp or overnight?\n"
            "💸 Your budget\n\n"
            "Got other questions? Just ask! 💬"
        )
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me about camps..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching for the best camps..."):
            try:
                response = process_query(prompt, config)
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nPlease try again or rephrase your question."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
