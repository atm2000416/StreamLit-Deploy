#!/usr/bin/env python3
"""
generate_embeddings.py
─────────────────────
One-time (and incremental) script to generate semantic embeddings for all camps.
Stores results in camp_directory.camp_embeddings.

Usage:
    python3 generate_embeddings.py                    # generate all missing
    python3 generate_embeddings.py --rebuild          # regenerate everything
    python3 generate_embeddings.py --cid 189          # single camp

Requirements:
    pip install sqlalchemy pymysql requests python-dotenv

Environment (same .env or secrets as the main app):
    DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_CAMP_DIR, GEMINI_API_KEY
"""

import os, sys, json, time, hashlib, argparse
import requests
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "embedding-001:embedContent"
)
EMBED_DIM    = 768
BATCH_DELAY  = 0.1   # seconds between API calls (stay within free tier rate limit)

# ── DB connection ─────────────────────────────────────────────────────────────
def get_engine():
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "3306")
    user = os.environ.get("DB_USER")
    pwd  = os.environ.get("DB_PASS")
    db   = os.environ.get("DB_CAMP_DIR", "camp_directory")
    if not user or not pwd:
        # Try reading from streamlit secrets format
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            host = secrets.get("DB_HOST", host)
            port = secrets.get("DB_PORT", port)
            user = secrets.get("DB_USER", user)
            pwd  = secrets.get("DB_PASS", pwd)
            db   = secrets.get("DB_CAMP_DIR", db)
        except Exception:
            pass
    uri = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(uri, pool_pre_ping=True)


# ── Embedding API ─────────────────────────────────────────────────────────────
def embed_text(text_content: str, api_key: str) -> list[float]:
    """Call Gemini embedding-001 and return 768-dim vector."""
    resp = requests.post(
        f"{GEMINI_EMBED_URL}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={
            "model": "models/embedding-001",
            "content": {"parts": [{"text": text_content}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


# ── Fingerprint builder ───────────────────────────────────────────────────────
def build_fingerprint(row: dict) -> str:
    """
    Combine all meaningful text about a camp into one string for embedding.
    Richer text = better semantic search.
    Structure mirrors how a user would describe an ideal camp.
    """
    parts = []

    name = (row.get("camp_name") or "").strip()
    if name:
        parts.append(f"Camp: {name}.")

    desc = (row.get("description") or "").strip()
    if desc:
        parts.append(f"About: {desc}")

    activities = (row.get("activities") or "").strip()
    if activities:
        parts.append(f"Activities offered: {activities}.")

    programs = (row.get("programs") or "").strip()
    if programs:
        # programs is a pipe-separated list of session names + descriptions
        prog_lines = [p.strip() for p in programs.split("|||") if p.strip()]
        if prog_lines:
            parts.append("Programs: " + " | ".join(prog_lines[:20]))  # cap at 20

    style = (row.get("camp_style") or "").strip()
    if style:
        parts.append(f"Type: {style} camp.")

    province = (row.get("province") or "").strip()
    city     = (row.get("city") or "").strip()
    if city and province:
        parts.append(f"Location: {city}, {province}.")
    elif province:
        parts.append(f"Location: {province}.")

    age_min = row.get("age_min")
    age_max = row.get("age_max")
    if age_min is not None and age_max is not None:
        parts.append(f"Ages: {age_min} to {age_max}.")

    return " ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(rebuild: bool = False, single_cid: int = None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            api_key = secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY not found in environment or .streamlit/secrets.toml")

    engine = get_engine()

    with engine.connect() as conn:
        # ── Fetch camps ───────────────────────────────────────────────────────
        cid_filter = "AND cc.cid = :cid" if single_cid else ""
        existing_filter = "" if rebuild else """
            AND cc.cid NOT IN (SELECT cid FROM camp_directory.camp_embeddings)
        """

        sql = text(f"""
            SELECT
                cc.cid,
                cc.camp_name,
                cc.description,
                cc.camp_style,
                cc.province,
                MIN(sc.city)                                              AS city,
                GROUP_CONCAT(DISTINCT sc.specialty_label
                    ORDER BY sc.specialty_label SEPARATOR ', ')           AS activities,
                GROUP_CONCAT(
                    DISTINCT CONCAT(
                        sc.class_name,
                        COALESCE(
                            IF(NULLIF(TRIM(s.mini_description),'') IS NOT NULL,
                               CONCAT(' — ', TRIM(s.mini_description)), NULL),
                            IF(NULLIF(TRIM(s.description),'') IS NOT NULL,
                               CONCAT(' — ', LEFT(TRIM(s.description), 200)), NULL),
                            ''
                        )
                    )
                    ORDER BY sc.listing_tier SEPARATOR ' ||| '
                )                                                         AS programs,
                MIN(sc.age_from)                                          AS age_min,
                MAX(sc.age_to)                                            AS age_max
            FROM camp_directory.camps_clean cc
            JOIN camp_directory.sessions_clean sc
                ON sc.cid = cc.cid AND sc.province = cc.province
            LEFT JOIN camp_directory.sessions s
                ON s.id = sc.session_id
            WHERE cc.status = 1
              AND sc.status = 1
              AND sc.is_virtual = 0
              {cid_filter}
              {existing_filter}
            GROUP BY cc.cid, cc.camp_name, cc.description, cc.camp_style, cc.province
            ORDER BY cc.cid
        """)

        params = {"cid": single_cid} if single_cid else {}
        rows = conn.execute(sql, params).fetchall()
        cols = list(conn.execute(sql, params).keys()) if not rows else None

    if not rows:
        print("No camps to process. All embeddings are up to date.")
        return

    # Re-fetch with column names
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT
                    cc.cid, cc.camp_name, cc.description, cc.camp_style, cc.province,
                    MIN(sc.city) AS city,
                    GROUP_CONCAT(DISTINCT sc.specialty_label
                        ORDER BY sc.specialty_label SEPARATOR ', ') AS activities,
                    GROUP_CONCAT(
                        DISTINCT CONCAT(
                            sc.class_name,
                            COALESCE(
                                IF(NULLIF(TRIM(s.mini_description),'') IS NOT NULL,
                                   CONCAT(' — ', TRIM(s.mini_description)), NULL),
                                IF(NULLIF(TRIM(s.description),'') IS NOT NULL,
                                   CONCAT(' — ', LEFT(TRIM(s.description), 200)), NULL),
                                ''
                            )
                        )
                        ORDER BY sc.listing_tier SEPARATOR ' ||| '
                    ) AS programs,
                    MIN(sc.age_from) AS age_min,
                    MAX(sc.age_to) AS age_max
                FROM camp_directory.camps_clean cc
                JOIN camp_directory.sessions_clean sc
                    ON sc.cid = cc.cid AND sc.province = cc.province
                LEFT JOIN camp_directory.sessions s
                    ON s.id = sc.session_id
                WHERE cc.status = 1 AND sc.status = 1 AND sc.is_virtual = 0
                {cid_filter} {existing_filter}
                GROUP BY cc.cid, cc.camp_name, cc.description, cc.camp_style, cc.province
                ORDER BY cc.cid
            """),
            params
        )
        cols  = list(result.keys())
        camps = [dict(zip(cols, row)) for row in result.fetchall()]

    total   = len(camps)
    success = 0
    failed  = []

    print(f"\n{'='*60}")
    print(f"Generating embeddings for {total} camps...")
    print(f"{'='*60}\n")

    with engine.connect() as conn:
        for i, camp in enumerate(camps, 1):
            cid  = camp["cid"]
            name = camp["camp_name"]

            try:
                fingerprint = build_fingerprint(camp)
                fp_hash     = hashlib.md5(fingerprint.encode()).hexdigest()

                # Skip if fingerprint unchanged (content hasn't changed)
                if not rebuild and not single_cid:
                    existing = conn.execute(
                        text("SELECT cid FROM camp_directory.camp_embeddings "
                             "WHERE cid = :cid"),
                        {"cid": cid}
                    ).fetchone()
                    if existing:
                        print(f"  [{i}/{total}] SKIP  {name} (already embedded)")
                        success += 1
                        continue

                vector = embed_text(fingerprint, api_key)

                conn.execute(text("""
                    INSERT INTO camp_directory.camp_embeddings
                        (cid, embedding, fingerprint, updated_at)
                    VALUES
                        (:cid, :embedding, :fingerprint, NOW())
                    ON DUPLICATE KEY UPDATE
                        embedding    = VALUES(embedding),
                        fingerprint  = VALUES(fingerprint),
                        updated_at   = NOW()
                """), {
                    "cid":         cid,
                    "embedding":   json.dumps(vector),
                    "fingerprint": fingerprint[:2000],
                })
                conn.commit()

                success += 1
                print(f"  [{i}/{total}] OK    {name}")
                time.sleep(BATCH_DELAY)

            except Exception as e:
                failed.append((cid, name, str(e)))
                print(f"  [{i}/{total}] FAIL  {name} — {e}")

    print(f"\n{'='*60}")
    print(f"Done. {success}/{total} embeddings generated.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for cid, name, err in failed:
            print(f"  cid={cid}  {name}: {err}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate camp embeddings")
    parser.add_argument("--rebuild", action="store_true",
                        help="Regenerate all embeddings even if they exist")
    parser.add_argument("--cid", type=int, default=None,
                        help="Generate embedding for a single camp by cid")
    args = parser.parse_args()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    run(rebuild=args.rebuild, single_cid=args.cid)
