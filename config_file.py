import os
import streamlit as st

def get_config():
    """
    Get configuration from Streamlit secrets (production) or environment variables (local)
    """
    # Try Streamlit secrets first (for deployed app)
    try:
        return {
            # API Keys
            "GEMINI_API_KEY": st.secrets.get("GEMINI_API_KEY"),
            "PINECONE_API_KEY": st.secrets.get("PINECONE_API_KEY"),
            "GOOGLE_API_KEY": st.secrets.get("GOOGLE_API_KEY"),
            "GOOGLE_CSE_ID": st.secrets.get("GOOGLE_CSE_ID"),
            
            # Aiven MySQL Database
            "DB_HOST": st.secrets.get("DB_HOST"),
            "DB_PORT": st.secrets.get("DB_PORT", "10536"),
            "DB_USER": st.secrets.get("DB_USER"),
            "DB_PASS": st.secrets.get("DB_PASS"),
            
            # Database names
            "DB_NAME": st.secrets.get("DB_NAME"),
            "DB_CAMPDB": st.secrets.get("DB_CAMPDB", "campdb"),
            "DB_CAMP_DIRECTORY": st.secrets.get("DB_CAMP_DIRECTORY", "camp_directory"),
            "DB_COMMON_UPDATE": st.secrets.get("DB_COMMON_UPDATE", "common_update"),
            
            # Pinecone
            "INDEX_NAME": st.secrets.get("INDEX_NAME", "searching-doolie"),
            "INDEX_HOST": st.secrets.get("INDEX_HOST"),
            "NAMESPACE": st.secrets.get("NAMESPACE", "default"),
        }
    except:
        # Fallback to environment variables (for local development)
        return {
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID"),
            
            "DB_HOST": os.getenv("DB_HOST"),
            "DB_PORT": os.getenv("DB_PORT", "10536"),
            "DB_USER": os.getenv("DB_USER"),
            "DB_PASS": os.getenv("DB_PASS"),
            
            "DB_NAME": os.getenv("DB_NAME"),
            "DB_CAMPDB": os.getenv("DB_CAMPDB", "campdb"),
            "DB_CAMP_DIRECTORY": os.getenv("DB_CAMP_DIRECTORY", "camp_directory"),
            "DB_COMMON_UPDATE": os.getenv("DB_COMMON_UPDATE", "common_update"),
            
            "INDEX_NAME": os.getenv("INDEX_NAME", "searching-doolie"),
            "INDEX_HOST": os.getenv("INDEX_HOST"),
            "NAMESPACE": os.getenv("NAMESPACE", "default"),
        }

def get_database_uri(database_name=None):
    """Get MySQL connection URI for SQLAlchemy"""
    from urllib.parse import quote_plus
    config = get_config()
    
    # Use provided database name or default from config
    db = database_name or config['DB_NAME']
    
    return (
        f"mysql+pymysql://{config['DB_USER']}:{quote_plus(config['DB_PASS'])}"
        f"@{config['DB_HOST']}:{config['DB_PORT']}/{db}"
    )

def get_all_database_uris():
    """Get URIs for all three databases"""
    config = get_config()
    return {
        'campdb': get_database_uri(config['DB_CAMPDB']),
        'camp_directory': get_database_uri(config['DB_CAMP_DIRECTORY']),
        'common_update': get_database_uri(config['DB_COMMON_UPDATE'])
    }

def get_target_databases():
    """Get list of database names to query (replaces TARGET_SCHEMAS in input.py)"""
    config = get_config()
    return [
        config['DB_CAMPDB'],
        config['DB_CAMP_DIRECTORY'],
        config['DB_COMMON_UPDATE']
    ]
