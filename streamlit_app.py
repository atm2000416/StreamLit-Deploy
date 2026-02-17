import streamlit as st
import sys
from io import StringIO
from config import get_config

# Configure page
st.set_page_config(
    page_title="Camp Chatbot",
    page_icon="🏕️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .camp-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="camp-header">
    <h1>🏕️ Camp Discovery Chatbot</h1>
    <p>Find the perfect camp in Canada for your child!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    Share these details to get personalized recommendations:
    
    - 📍 **Region** in Canada
    - 🎯 **Type of camp** (STEM, sports, arts, etc.)
    - 👶 **Age and gender** of camper
    - 🏕️ **Day camp or overnight?**
    - 💸 **Your budget**
    
    **Examples:**
    - "Show me STEM camps in Ontario for 12-year-old boys under $500"
    - "What overnight camps in BC focus on outdoor adventures?"
    """)
    
    st.divider()
    
    # Connection status
    with st.expander("🔌 System Status"):
        config = get_config()
        st.write("✅ Gemini API" if config.get("GEMINI_API_KEY") else "❌ Gemini API")
        st.write("✅ Pinecone" if config.get("PINECONE_API_KEY") else "❌ Pinecone")
        st.write("✅ Database" if config.get("DB_HOST") else "❌ Database")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """Hi! I'm your camp chatbot 🤖

Please share:

📍 Region in Canada you're interested in
🎯 Type of camp (STEM, sports, arts, etc.)
👶 Age and gender of the camper
🏕️ Day camp or overnight?
💸 Your budget

Got other questions? Just ask! 💬"""
    }]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Import RAG functions (from input.py)
def load_rag_system():
    """Load all RAG processing functions"""
    import os
    import requests
    import json
    from typing import List, Dict
    
    # Get configuration
    config = get_config()
    
    # Set environment variables for your existing code
    os.environ["GOOGLE_API_KEY"] = config["GEMINI_API_KEY"]
    
    # Import your existing functions
    # You'll need to refactor input.py into importable functions
    # For now, return config
    return config

def process_user_query(user_text: str):
    """
    Main processing function that calls your RAG system
    Integrates all logic from input.py
    """
    config = get_config()
    
    # Import your functions from rag_processor
    from rag_processor import (
        classify_query,
        run_case1,
        run_case2,
        validate_answer,
        summarize_answer,
        run_camp_verify_pipeline
    )
    
    try:
        # Step 1: Classify query
        case = classify_query(user_text, config["GEMINI_API_KEY"])
        
        # Check for blocked content
        if case == "BLOCKED" or "violates our community guidelines" in case:
            return "Your content violates our community guidelines, do you have another question?"
        
        # Step 2: Process based on case
        if case == "Case1":
            output = run_case1(user_text, config)
            is_valid, result = validate_answer(user_text, output, config["GEMINI_API_KEY"])
            
            if is_valid:
                final_answer = result
            else:
                # Fallback to Case3
                output2 = run_case2(user_text, config)
                combined = f"STRUCTURED: {output}\n\nDESCRIPTIVE: {output2}"
                final_answer = summarize_answer(user_text, combined, config["GEMINI_API_KEY"])
        
        elif case == "Case2":
            output = run_case2(user_text, config)
            is_valid, result = validate_answer(user_text, output, config["GEMINI_API_KEY"])
            
            if is_valid:
                final_answer = result
            else:
                # Fallback to Case3
                output1 = run_case1(user_text, config)
                combined = f"STRUCTURED: {output1}\n\nDESCRIPTIVE: {output}"
                final_answer = summarize_answer(user_text, combined, config["GEMINI_API_KEY"])
        
        else:  # Case3
            output1 = run_case1(user_text, config)
            output2 = run_case2(user_text, config)
            combined = f"STRUCTURED: {output1}\n\nDESCRIPTIVE: {output2}"
            final_answer = summarize_answer(user_text, combined, config["GEMINI_API_KEY"])
        
        # Step 3: Verify camp names
        try:
            camp_result = run_camp_verify_pipeline(final_answer, config)
            return camp_result["rewritten_sentence"]
        except Exception as e:
            # If camp verification fails, return answer without verification
            return final_answer
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ Error processing your query: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."

# Chat input
if prompt := st.chat_input("Ask me about camps..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and respond
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching camps..."):
            try:
                # Load RAG system
                config = load_rag_system()
                
                # Process query
                response = process_user_query(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nPlease check your configuration and try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Clear chat button
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
