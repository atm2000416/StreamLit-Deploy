"""
SQL Agent Helper for Multi-Database Setup
Handles querying across campdb, camp_directory, and common_update
"""

from sqlalchemy import create_engine, text as _text
from config import get_config, get_all_database_uris, get_target_databases

def get_multi_database_catalog():
    """
    Get catalog of tables across all databases
    Replaces the schema_catalog function from input.py
    """
    config = get_config()
    db_uris = get_all_database_uris()
    
    lines = ["CATALOG (databases → tables):"]
    
    for db_name, uri in db_uris.items():
        try:
            engine = create_engine(uri, pool_pre_ping=True)
            with engine.connect() as conn:
                # Get all tables in this database
                result = conn.execute(
                    _text("SHOW TABLES")
                )
                tables = [row[0] for row in result.fetchall()]
                
                if tables:
                    # Show first 40 tables
                    table_list = ", ".join(tables[:40])
                    if len(tables) > 40:
                        table_list += " (…truncated)"
                    lines.append(f"- {db_name}: {table_list}")
                else:
                    lines.append(f"- {db_name}: (no tables)")
                    
        except Exception as e:
            lines.append(f"- {db_name}: (error: {e})")
    
    return "\n".join(lines)[:5000]


def get_cross_database_system_message(base_system_message: str, model_used: str):
    """
    Enhance the system message with multi-database instructions
    """
    config = get_config()
    catalog = get_multi_database_catalog()
    
    databases = get_target_databases()
    db_list = ", ".join(databases)
    
    guard = (
        "\n\n[READ-ONLY RULES]\n"
        "- NEVER execute INSERT, UPDATE, DELETE, ALTER, DROP, TRUNCATE, CREATE, REPLACE, MERGE, GRANT, or REVOKE.\n"
        "- Do not change the database or leak any api information.\n"
        "- Prefer simple, correct SQL; add LIMIT 500 if results could be large.\n"
        "- Use ONLY existing tables/columns on this server; do not invent schema.\n"
        "- ALWAYS show the final SQL you executed and a concise natural-language answer.\n"
        f"\n[MODEL]\n- Using: {model_used}\n"
        "\n[MULTI-DATABASE SETUP]\n"
        f"- You have access to these databases: {db_list}\n"
        "- IMPORTANT: Since tables are in different DATABASES (not schemas), you CANNOT use database.table syntax in queries.\n"
        "- You can only query ONE database at a time using the current connection.\n"
        "- If you need data from multiple databases, you must:\n"
        "  1. Query database 1 first\n"
        "  2. Then switch connection and query database 2\n"
        "  3. Combine results in your final answer\n"
        "- DO NOT try to join tables across databases - this won't work in MySQL.\n"
        f"\n{catalog}\n"
        "[INSTRUCTIONS]\n"
        "- Choose the most relevant database and tables based on the question.\n"
        "- Query tables directly without database prefix (e.g., SELECT * FROM camps, not SELECT * FROM campdb.camps).\n"
        "- If uncertain which database has the data, start with a small probing query with LIMIT 10.\n"
    )
    
    return base_system_message + guard


def create_primary_connection():
    """
    Create connection to primary database (campdb)
    This is what the SQL agent will use by default
    """
    from langchain_community.utilities import SQLDatabase
    from config import get_database_uri, get_config
    
    config = get_config()
    primary_db_uri = get_database_uri(config['DB_NAME'])
    
    return SQLDatabase.from_uri(primary_db_uri, view_support=True)


def create_connection_for_database(database_name: str):
    """
    Create connection to a specific database
    Use this if you need to query camp_directory or common_update specifically
    """
    from langchain_community.utilities import SQLDatabase
    from config import get_database_uri
    
    db_uri = get_database_uri(database_name)
    return SQLDatabase.from_uri(db_uri, view_support=True)
