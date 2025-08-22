"""
client_s3 – description-first workflow for Amazon S3
─────────────────────────────────────────────────────
• List every object in BUCKET
• Auto-describe new files (PK/FK/date range) with auto_describe()
• Feed file schemas + previews to the LLM
• LLM rewrites empty descriptions, selects ≤3 files, and answers
"""
from __future__ import annotations

import io, os, logging, re, boto3, pandas as pd
from textwrap import dedent
from typing import List, Dict
from anthropic import Anthropic
from mcp_system.scanner.metadata_store import SidecarStore, FileMeta
from mcp_system.scanner.auto_desc import auto_describe

MODEL  = "claude-3-5-haiku-20241022"
CHUNK  = 256_000               # bytes read per object

s3      = boto3.client("s3")
BUCKET  = os.getenv("S3_BUCKET")
client  = Anthropic()
log     = logging.getLogger(__name__)

# ─────────── util regex ───────────
_HDR1  = re.compile(r"^DESCRIPTIONS$", re.M)
_HDR2  = re.compile(r"\bDESCRIPTIONS\b", re.I)
_SPLIT = re.compile(r"^(?:SELECTED_FILES|ANSWER)$", re.M)
_KV    = re.compile(r"^(.*?)\s*:\s*(.+)$")

def get_s3_client():
    return s3
# ─────────── storage helpers ───────────
def _list_meta() -> List[Dict]:
    """List every non-side-car object in the bucket."""
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, PaginationConfig={"PageSize": 100}
    )
    out: List[Dict] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".meta.json"):
                continue
            out.append({"path": key, "title": key})
    return out


def _read(path: str, length: int = CHUNK) -> str:
    body = s3.get_object(
        Bucket=BUCKET, Key=path, Range=f"bytes=0-{length-1}"
    )["Body"].read()
    try:
        return body.decode("utf-8", errors="ignore")
    except Exception:
        return body.decode("latin-1", errors="ignore")

# ─────────── doc collector ───────────
def _collect_docs() -> List[Dict]:
    docs: List[Dict] = []
    store = SidecarStore()
    for meta in _list_meta():
        path = meta["path"]
        full_path = f"{BUCKET}/{path}"
        cached_meta = store.get_meta(full_path, cloud="s3")
        cached = (cached_meta.description if cached_meta else "") or ""
        preview = _read(path)
        schema  = ""
        if path.lower().endswith(".csv"):
            try:
                schema = ", ".join(pd.read_csv(io.StringIO(preview), nrows=0).columns)
            except Exception:
                pass
        docs.append(
            {
                "path": path,
                "title": meta["title"],
                "description": cached,
                "schema": schema or "(unknown)",
                "sample_text": preview[:1_500],
            }
        )
    return docs

# ─────────── prompt builder ───────────
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
            schema: {d['schema']}
            sample_text:
            ```
            {d['sample_text']}
            ```
        """).strip()
        for d in docs
    )
    return f"""
You are an **elite file-analysis LLM**.
* Rewrite empty descriptions based on sample_text.
* Pick up to **3** files to answer the user's question.
* Answer in ≤120 words.

{RETURN_SPEC}

## User question
{question}

## Files
{docs_block}
""".strip()

# ─────────── prescan: auto-describe new files ───────────
def _auto_prescan() -> None:
    """If a file lacks cloud metadata, seed a quick auto description then run the scanner."""
    from mcp_system.scanner.core import scan_object
    store = SidecarStore()
    for meta in _list_meta():
        key = meta["path"]
        full_path = f"{BUCKET}/{key}"
        if store.get_meta(full_path, cloud="s3"):
            continue
        # 1) lightweight description
        raw = _read(key, length=20_000)
        desc = auto_describe(key, raw.encode() if isinstance(raw, str) else raw)
        seed = FileMeta(
          path=full_path,
            cloud="s3",
            title=key,
            description=desc,
            mime_type="application/octet-stream",
            size=0,
            last_scanned=__import__("datetime").datetime.utcnow(),
        )
        store.write_meta(seed, cloud="s3")
        # 2) full scan to fill mime/size and (possibly better) description
        scan_object(full_path, cloud="s3")

# ─────────── public API ───────────
def answer(question: str, target_path: str = None) -> str:
    _auto_prescan()
    docs = _collect_docs()
    if target_path:
        # filter to only the planner-specified file
        docs = [d for d in docs if d["path"] == target_path]
        if not docs:
            return f"No data found for file: {target_path}"

    prompt = _build_prompt(question, docs)
    raw = client.messages.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 800,
        temperature = 0.1,
    ).content[0].text.strip()

    # cache any DESCRIPTIONS the model produced
    if _HDR1.search(raw) or _HDR2.search(raw):
        chunk = _HDR1.split(raw,1)[1] if _HDR1.search(raw) else _HDR2.split(raw,1)[1]
        chunk = _SPLIT.split(chunk,1)[0]
        store = SidecarStore()
        for ln in chunk.splitlines():
            m = _KV.match(ln.strip())
            if m:
                p, desc = m.groups()
                key = p.strip()
                full_path = f"{BUCKET}/{key}"
                existing = store.get_meta(full_path, cloud="s3")

                if existing:
                    existing.description = desc.strip()
                    store.write_meta(existing, cloud="s3")
                else:
                    seed = FileMeta(
                        path=full_path,
                        cloud="s3",
                        title=key,
                        description=desc.strip(),
                        mime_type="application/octet-stream",
                        size=0,
                        last_scanned=__import__("datetime").datetime.now(),
                    )
                    store.write_meta(seed, cloud="s3")

    if "ANSWER" in raw:
        return raw.split("ANSWER",1)[1].strip()
    return raw

# ─────────── CLI test ───────────
if __name__ == "__main__":
    import sys, textwrap
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("query> ")
    print("\n— thinking —\n")
    print(textwrap.fill(answer(q), 88), end="\n\n")
