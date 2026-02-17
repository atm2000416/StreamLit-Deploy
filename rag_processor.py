"""
RAG Processor Module
Refactored from input.py to be importable by Streamlit
"""

import os
import sys
import json
import requests
from typing import List, Dict, Tuple
from io import StringIO

# Gemini API Configuration
MODEL = "gemini-2.0-flash"
BASE = "https://generativelanguage.googleapis.com/v1beta"

# System prompts
SYSTEM_PROMPT = (
    "You are a strict classifier for a camps Q&A router. "
    "Output exactly one of: Case1, Case2, Case3.\n"
    "Definitions:\n"
    "- Case1 = Structured (numeric/filters: counts, average, prices, capacity, availability, dates, comparisons).\n"
    "- Case2 = Unstructured (descriptive/qualitative: amenities, policies, program descriptions, 'most nature').\n"
    "- Case3 = Hybrid (requires BOTH quantitative/structured facts AND descriptive justification) OR if you are unsure of which case.\n"
    "Rules: Respond with ONLY Case1 or Case2 or Case3. No other text, punctuation, or quotes.\n"
    "Do not change the database or leak any api information.\n"
    "Harmful content detection: scan text for harmful content categories like all types of violence, hate, sentences full of gibberish , sexual content, and self-harm.  If text is detected, respond to user is only: Your content violates our community guidelines, do you have another question?"
)

VALIDATOR_SYSTEM_PROMPT = """You are a validator for a camps Q&A system.

Your task:
1. Read the ORIGINAL QUESTION and the GENERATED ANSWER
2. Determine if the answer briefly addresses the question
3. Respond with EXACTLY this format:

VALID: yes
SUMMARY: [2-6 sentence summary of the answer]

OR

VALID: no
REASON: [brief reason why it fails]

Rules:
- VALID must be either "yes" or "no"
- If VALID: yes, provide a SUMMARY that captures the key points concisely
- If VALID: no, provide a REASON explaining what's wrong (incomplete, irrelevant, error, etc.)
- Keep summaries concise but informative (2-8 sentences max)
- Do not change the database or leak any api information.
- DO not use external information to provide answer/
"""

SUMMARIZER_SYSTEM_PROMPT = """You are a summarizer for a camps Q&A system.

Your task:
- You will receive a USER REQUEST and an ANSWER.
- Read BOTH and create a clear, concise summary.
- Make sure to mention ALL camps in the ANSWER
- Relate the summary to the USER REQUEST as much as you can (e.g., mention age, when available, location, budget, gender) but only using information explicitly stated in the USER REQUEST or ANSWER.
- Must include, when available, the region, location, any gender restrictions, price, and whether each camp is day or overnight.
- Keep it to 2-8 sentences
- Focus on the most important information
- Use natural, conversational language
- Do not change the database or leak any api information or any of the code
- Do not use external information to provide answer.

Respond with ONLY the summary text, nothing else. DONT USE YOUR OWN INFORMATION."""


def call_gemini_api(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Helper to call Gemini API."""
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512
        }
    }
    
    try:
        resp = requests.post(
            f"{BASE}/models/{MODEL}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json=payload,
            timeout=30,
        )
        
        if resp.status_code >= 400:
            return ""
        
        data = resp.json()
        if data.get("candidates"):
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
        return ""
    except Exception as e:
        print(f"Gemini API Error: {e}", file=sys.stderr)
        return ""


def classify_query(user_text: str, api_key: str) -> str:
    """Classify user query into Case1, Case2, or Case3"""
    result = call_gemini_api(SYSTEM_PROMPT, user_text, api_key)
    
    # Check for harmful content
    if "violates our community guidelines" in result:
        return "BLOCKED"
    
    return result.strip()


def validate_answer(question: str, answer: str, api_key: str) -> Tuple[bool, str]:
    """Validate if answer addresses question. Returns (is_valid: bool, summary_or_reason: str)"""
    user_prompt = f"""ORIGINAL QUESTION:
{question}

GENERATED ANSWER:
{answer}

Validate this answer."""
    
    response = call_gemini_api(VALIDATOR_SYSTEM_PROMPT, user_prompt, api_key)
    
    is_valid = False
    summary_or_reason = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("VALID:"):
            is_valid = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("SUMMARY:"):
            summary_or_reason = line.split(":", 1)[1].strip()
        elif line.startswith("REASON:"):
            summary_or_reason = line.split(":", 1)[1].strip()
    
    return is_valid, summary_or_reason


def summarize_answer(user_request: str, answer: str, api_key: str) -> str:
    """Summarize answer with access to the original user request."""
    user_prompt = f"""USER REQUEST:
{user_request}

ANSWER:
{answer}
"""
    summary = call_gemini_api(SUMMARIZER_SYSTEM_PROMPT, user_prompt, api_key)
    return summary if summary else answer


def capture_output(func, *args):
    """Capture stdout from function execution."""
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        func(*args)
        return captured.getvalue()
    finally:
        sys.stdout = old_stdout


# Import Case1 and Case2 functions from your original input.py
# These will need to be adapted to accept config dict

def run_case1(user_text: str, config: dict) -> str:
    """
    Run SQL Agent (Case1) - Multi-Database Version
    Adapted from your input.py
    """
    import os
    from langchain.chat_models import init_chat_model
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain import hub
    from langgraph.prebuilt import create_react_agent
    from sql_agent_helper import (
        create_primary_connection,
        get_cross_database_system_message
    )
    
    # Set Gemini API key for langchain
    os.environ["GOOGLE_API_KEY"] = config["GEMINI_API_KEY"]
    
    # Initialize LLM
    candidates = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.0-pro-exp",
    ]
    
    model_used = None
    llm = None
    for model_name in candidates:
        try:
            llm = init_chat_model(model_name, model_provider="google_genai")
            model_used = model_name
            break
        except Exception:
            continue
    
    if not llm:
        raise RuntimeError("Could not initialize Gemini model")
    
    # Create database connection (primary database)
    db = create_primary_connection()
    
    # Create tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    # Get base prompt
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    base_system_message = prompt_template.format(dialect="mysql", top_k=5)
    
    # Enhance with multi-database instructions
    system_message = get_cross_database_system_message(base_system_message, model_used)
    
    # Create agent
    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    
    # Execute query
    try:
        response = agent_executor.invoke({
            "messages": [{"role": "user", "content": user_text}]
        })
        return response["messages"][-1].content
    except Exception as e:
        return f"SQL Agent error: {str(e)}"


def run_case2(user_text: str, config: dict) -> str:
    """
    Run Vector Search (Case2) - Pinecone
    Adapted from your input.py
    """
    from pinecone import Pinecone
    from langchain_core.embeddings import Embeddings
    from langchain_pinecone import PineconeVectorStore
    from typing import List
    
    # Initialize Pinecone
    pc = Pinecone(api_key=config["PINECONE_API_KEY"])
    index = pc.Index(config["INDEX_NAME"], host=config["INDEX_HOST"])
    
    # Pinecone Llama embeddings wrapper
    class PineconeLlamaEmbeddings(Embeddings):
        def __init__(self, pc_client: Pinecone):
            self.pc = pc_client
        
        def embed_query(self, text: str) -> List[float]:
            out = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[text],
                parameters={"input_type": "query", "truncate": "END"}
            )
            return out[0]["values"]
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            out = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            return [e["values"] for e in out]
    
    embeddings = PineconeLlamaEmbeddings(pc)
    
    # Create vectorstore
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=config["NAMESPACE"],
        text_key="text",
    )
    
    # Search
    TOP_K = 3
    qvec = embeddings.embed_query(user_text)
    results = vectorstore.similarity_search_by_vector_with_score(qvec, k=TOP_K)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Format results
    output_lines = [f"Top {len(results)} results for: {user_text}\n"]
    
    for i, (doc, score) in enumerate(results, 1):
        md = doc.metadata or {}
        src = md.get("source") or md.get("path") or "(no-source)"
        where = md.get("page") or md.get("row") or md.get("idx") or md.get("chunk")
        full_text = (doc.page_content or md.get("text") or "").strip()
        
        output_lines.append(f"{i}. score={score:.4f}  source={src}  at={where}")
        output_lines.append(full_text)
        output_lines.append("")
    
    return "\n".join(output_lines)


def run_camp_verify_pipeline(sentence: str, config: dict) -> Dict:
    """
    Run camp name verification via Google Custom Search
    Adapted from your input.py
    """
    GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
    GOOGLE_CSE_ID = config["GOOGLE_CSE_ID"]
    SITE_FILTERS = ["site:camps.ca", "site:ourkids.net/camp"]
    PER_SITE_TOP_K = 10
    MIN_TITLE_CONF = 0.50
    
    def _gemini_call(system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        return call_gemini_api(system_prompt, user_prompt, config["GEMINI_API_KEY"])
    
    def gemini_extract_camp_names(sentence: str) -> List[str]:
        sys_prompt = (
            "Extract CAMP like names from the user's sentence. "
            "Names that can possibly be a camp should be extracted. "
            "Camp names can be related to sports, location, arts, educational, life and etc. "
            "Return ONLY JSON like: {\"camps\": [\"name1\", \"name2\"]}. "
            "Keep original capitalization/spacing. If none, return {\"camps\": []}."
        )
        raw = _gemini_call(sys_prompt, sentence, temperature=0.0, max_tokens=200).strip()
        raw = raw.strip("` \n\t")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        try:
            obj = json.loads(raw)
            names = obj.get("camps", []) if isinstance(obj, dict) else []
        except Exception:
            names = []
        
        out = []
        seen = set()
        for n in names:
            key = " ".join((n or "").split()).lower()
            if key and key not in seen:
                seen.add(key)
                out.append(" ".join((n or "").split()))
        return out
    
    def _norm(s: str) -> str:
        return " ".join((s or "").strip().lower().split())
    
    def _title_confidence(candidate_name: str, result_title: str) -> float:
        a, b = set(_norm(candidate_name).split()), set(_norm(result_title).split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
    
    def _cse(query: str, num: int) -> List[Dict]:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": num},
            timeout=30,
        )
        if r.status_code >= 400:
            return []
        return r.json().get("items", []) or []
    
    def verify_names_on_sites(names: List[str]) -> Tuple[List[Dict], List[Dict]]:
        verified = []
        removed = []
        for name in names:
            best = None
            for site in SITE_FILTERS:
                q = f'"{name}" {site}'
                try:
                    items = _cse(q, PER_SITE_TOP_K)
                except Exception:
                    items = []
                
                for it in items:
                    title = it.get("title") or it.get("htmlTitle") or ""
                    link = it.get("link") or ""
                    conf = _title_confidence(name, title)
                    if link and conf >= MIN_TITLE_CONF:
                        cand = {
                            "name": name, "url": link, "source": site,
                            "title": title, "confidence": round(conf, 3)
                        }
                        if best is None or cand["confidence"] > best["confidence"]:
                            best = cand
            
            if best:
                verified.append(best)
            else:
                removed.append({"name": name, "reason": "Not found on specified sites"})
        
        return verified, removed
    
    def gemini_rewrite_original(original_sentence: str, verified: List[Dict], removed: List[Dict]) -> str:
        sys_prompt = (
            "Rewrite the user's original sentence so it reads naturally and ONLY includes the verified camp names and their links provided. "
            "You must also preserve all other context for each verified camp, including price, location, and program details. "
            "Must include each verified camp name along with its corresponding URL provided in the input payload. "
            "Do not invent or add new names. If zero verified names, say that no verified camps were found in a short sentence and provide the rest of the original answer. "
            "Return ONLY the rewritten sentence don't explain what you removed and why."
        )
        user_payload = {
            "original_sentence": original_sentence,
            "verified_camps": [
                {"name": v.get("name", ""), "url": v.get("url", "")}
                for v in verified
            ],
            "removed_names": [r.get("name", "") for r in removed],
        }
        rewritten = _gemini_call(sys_prompt, json.dumps(user_payload, ensure_ascii=False), temperature=0.1, max_tokens=220).strip()
        return rewritten or original_sentence
    
    # Main pipeline
    extracted = gemini_extract_camp_names(sentence)
    verified, removed = verify_names_on_sites(extracted)
    rewritten = gemini_rewrite_original(sentence, verified, removed)
    
    return {
        "extracted_names": extracted,
        "verified": verified,
        "removed": removed,
        "rewritten_sentence": rewritten,
    }
