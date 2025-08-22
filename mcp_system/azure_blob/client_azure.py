"""
client_azure.py – description‑first workflow for Azure Blob Storage
------------------------------------------------------------------
• List every blob in the container and show previews
• Auto‑describe any blob whose side‑car metadata is missing (uses mcp_system.scanner.core.scan_object)
• Ask the LLM to re‑write empty descriptions, pick ≤3 files, and answer the user’s question
"""
from __future__ import annotations

import io
import os
import logging
import re
from textwrap import dedent
from typing import List, Dict
from datetime import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient
from anthropic import Anthropic
from mcp_system.scanner.metadata_store import SidecarStore, FileMeta
from mcp_system.scanner.core import scan_object, get_blob_client
CHUNK = 256_000

# ────────────────────────── Config / init ────────────────────────────
AZ_CONN: str      = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER: str = os.getenv("AZURE_CONTAINER")
MODEL: str        = "claude-3-5-haiku-20241022"

blob_svc = BlobServiceClient.from_connection_string(AZ_CONN)
client   = Anthropic()
log      = logging.getLogger(__name__)

# ────────────────────────── Regex helpers ────────────────────────────
_HDR1  = re.compile(r"^DESCRIPTIONS$", re.M)
_HDR2  = re.compile(r"\bDESCRIPTIONS\b", re.I)
_SPLIT = re.compile(r"^(?:SELECTED_FILES|ANSWER)$", re.M)
_KV    = re.compile(r"^(.*?)\s*:\s*(.+)$")

# ────────────────────────── Storage helpers ──────────────────────────

def _list_meta() -> List[Dict]:
    """Return minimal metadata for every *data* blob (skip side‑car files)."""
    container = blob_svc.get_container_client(AZ_CONTAINER)
    return [
        {"path": b.name, "title": b.name, "description": ""}
        for b in container.list_blobs() if not b.name.endswith(".meta.json")
    ]

def _read_blob(path: str, length: int = CHUNK) -> str:
    """Download up‑to *length* bytes, decode UTF‑8 (fallback latin‑1)."""
    data = get_blob_client(path).download_blob(offset=0, length=length).readall()
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")

# ────────────────────────── Side‑car prescan ─────────────────────────

def _prescan() -> None:
    """Create sidecar metadata for any new blob"""
    store = SidecarStore()
    for meta in _list_meta():
        key = meta["path"]
        full_path = f"{AZ_CONTAINER}/{key}"

        # Check if metadata exists in cloud
        if not store.get_meta(full_path, cloud="azure"):
            # Create and save metadata
            meta_obj = FileMeta(
                path=full_path,
                cloud="azure",
                title=key,
                description="",  # Will be filled by scan
                mime_type="",  # Will be detected
                size=0,  # Will be updated
                last_scanned=datetime.now()
            )
            store.write_meta(meta_obj, cloud="azure")

            # Now scan to fill description
            scan_object(full_path, cloud="azure")

# ────────────────────────── Doc collector ────────────────────────────

def _collect_docs() -> List[Dict]:
    _prescan()
    docs: List[Dict] = []
    store = SidecarStore()
    for meta in _list_meta():
        full_path = f"{AZ_CONTAINER}/{meta['path']}"
        m = store.get_meta(full_path, cloud="azure")
        desc = (m.description if m else "") or ""
        sample = _read_blob(meta["path"])[:1500]
        try:
            schema = ", ".join(pd.read_csv(io.StringIO(sample), nrows=0).columns)
        except Exception as e:
            schema = f"(Error reading schema: {e})"
        docs.append({
            "path":        meta["path"],
            "title":       meta["title"],
            "description": desc,
            "schema": schema,
            "sample_text": sample,
        })
    return docs

# ────────────────────────── Prompt builder ───────────────────────────
RETURN_SPEC = (
    "Return **exactly** this layout (no extra text):\n\n"
    "DESCRIPTIONS\n<path>: <desc>\n\n"
    "SELECTED_FILES\n<path> ...\n\n"
    "ANSWER\n<answer>"
)


def _build_prompt(question: str, docs: List[Dict]) -> str:
    docs_block = "\n\n".join(
        dedent(f"""
            ### {d['title']}
            path: {d['path']}
            description: {d['description'] or '⧗ (empty)'}
            schema: {d.get('schema','(n/a)')}
            sample_text:
            ```
            {d['sample_text']}
            ```
        """).strip()
        for d in docs
    )
    return f"""
You are an **elite file‑analysis LLM**.
* Rewrite empty descriptions based on sample_text.
* Pick up to **3** files to answer the user's question.
* Answer in ≤120 words.

{RETURN_SPEC}

## User question
{question}

## Files
{docs_block}
""".strip()

# ────────────────────────── Public answer() API ───────────────────────

def answer(question: str, target_path: str = None) -> str:
    docs = _collect_docs()
    if target_path:
        # filter to only the planner-specified file
        docs = [d for d in docs if d["path"] == target_path]
        if not docs:
            return f"No data found for file: {target_path}"
    prompt = _build_prompt(question, docs)
    msg    = client.messages.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 800,
        temperature = 0.1,
    )
    raw = "".join(chunk.text for chunk in msg.content).strip()

    # Cache any new DESCRIPTIONS lines emitted by the model
    if _HDR1.search(raw) or _HDR2.search(raw):
        chunk = _HDR1.split(raw, 1)[1] if _HDR1.search(raw) else _HDR2.split(raw, 1)[1]
        chunk = _SPLIT.split(chunk, 1)[0]
        store = SidecarStore()
        for ln in chunk.splitlines():
            m = _KV.match(ln)
            if m:
                p, desc = m.groups()
                key = p.strip()
                full_path = f"{AZ_CONTAINER}/{key}"
                existing = store.get_meta(full_path, cloud="azure")
                if existing:
                    existing.description = desc.strip()
                    store.write_meta(existing, cloud="azure")
                else:
                    seed = FileMeta(
                        path = full_path,
                        cloud = "azure",
                        title = key,
                        description = desc.strip(),
                        mime_type = "application/octet-stream",
                        size = 0,
                        last_scanned = __import__("datetime").datetime.utcnow(),
                    )
                    store.write_meta(seed, cloud="azure")
    # Strip model's wrapper
    if "ANSWER" in raw:
        raw = raw.split("ANSWER", 1)[1].strip()
    return raw

# ────────────────────────── CLI runner ────────────────────────────────
if __name__ == "__main__":
    import sys
    import textwrap

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("query> ")
    print("\n— thinking —\n")
    print(textwrap.fill(answer(q), 88), end="\n\n")
