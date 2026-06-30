#!/usr/bin/env python3
"""Safe recovery: rebuild FalkorDB nodes from Qdrant payloads without wiping survivors.

Differs from recover_from_qdrant.py:
  - Does NOT clear the graph first.
  - Uses MERGE semantics: only inserts memory IDs that don't already exist.
  - Re-enriches via POST /memory? No — direct graph write (matches existing script).
    Edges will be rebuilt by the existing AutoMem enrichment scheduler or via a follow-up
    /consolidate run after import.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from falkordb import FalkorDB
from qdrant_client import QdrantClient

load_dotenv()
load_dotenv(Path.home() / ".config" / "automem" / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "memories")
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD")
BATCH = 200


def existing_ids(g) -> Set[str]:
    rows = g.query("MATCH (m:Memory) RETURN m.id").result_set
    return {r[0] for r in rows if r and r[0]}


def fetch_all(qc) -> List[Dict[str, Any]]:
    offset = None
    out: List[Dict[str, Any]] = []
    while True:
        points, offset = qc.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            out.append({"id": str(p.id), "payload": p.payload or {}})
        print(f"  fetched batch, total={len(out)}", flush=True)
        if offset is None:
            break
        time.sleep(0.05)
    return out


def restore(g, mem: Dict[str, Any]) -> bool:
    payload = mem["payload"]
    mid = mem["id"]
    RESERVED = {"type", "confidence", "content", "timestamp", "importance", "tags", "id"}
    extras = []
    meta = payload.get("metadata") or {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            if k in RESERVED:
                continue
            if isinstance(v, (list, dict)):
                v = str(v)
            sv = str(v).replace("'", "\\'")
            extras.append(f"{k}: '{sv}'")
    extras_str = (", " + ", ".join(extras)) if extras else ""

    tags = payload.get("tags") or []
    tag_list = ", ".join(f"'{str(t).replace(chr(39), chr(92)+chr(39))}'" for t in tags) if tags else ""

    def _num(v, default):
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return default
        return default

    importance = _num(payload.get("importance"), 0.5)
    confidence = _num(payload.get("confidence"), 0.6)
    timestamp = payload.get("timestamp") or ""
    if not isinstance(timestamp, str):
        timestamp = str(timestamp)
    timestamp = timestamp.replace("'", "")
    mtype = payload.get("type") or "Memory"
    if not isinstance(mtype, str):
        mtype = "Memory"
    mtype = mtype.replace("'", "")

    query = f"""
    CREATE (m:Memory {{
        id: '{mid}',
        content: $content,
        timestamp: '{timestamp}',
        importance: {importance},
        type: '{mtype}',
        confidence: {confidence},
        tags: [{tag_list}]
        {extras_str}
    }})
    """
    try:
        g.query(query, {"content": payload.get("content", "")})
        return True
    except Exception as e:
        print(f"  ERROR {mid[:8]}: {e}", flush=True)
        return False


def main():
    print("== Safe Qdrant → FalkorDB recovery ==", flush=True)
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    fc = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT,
                  password=FALKORDB_PASSWORD,
                  username="default" if FALKORDB_PASSWORD else None)
    g = fc.select_graph(os.getenv("FALKORDB_GRAPH", "memories"))

    print("Loading existing graph IDs…", flush=True)
    have = existing_ids(g)
    print(f"  graph has {len(have)} memories already", flush=True)

    print("Fetching all Qdrant points…", flush=True)
    qpoints = fetch_all(qc)
    print(f"  qdrant has {len(qpoints)} points", flush=True)

    missing = [p for p in qpoints if p["id"] not in have]
    print(f"  → {len(missing)} need restore (skipping {len(qpoints)-len(missing)} duplicates)", flush=True)

    ok = fail = 0
    for i, mem in enumerate(missing, 1):
        if restore(g, mem):
            ok += 1
        else:
            fail += 1
        if i % 500 == 0:
            print(f"  progress {i}/{len(missing)}  ok={ok} fail={fail}", flush=True)

    print(f"\nDone. restored={ok} failed={fail}", flush=True)
    print(f"Final graph size: {len(existing_ids(g))}", flush=True)


if __name__ == "__main__":
    main()
