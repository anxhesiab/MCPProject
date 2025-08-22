"""
client.py – Orchestrator that plans, executes, and fuses multi-cloud datasets
--------------------------------------------------------------------
* Discover available datasets and their descriptions
* Plan which dataset to query first (and with what sub-question)
* Execute each step against the appropriate client
* Merge results sequentially into a cohesive, self-checking answer
"""
from __future__ import annotations
import os, io,sys, textwrap, re, json
from typing import List, Dict
import pandas as pd
from anthropic import Anthropic
from mcp_system.aws_s3.client_s3        import answer as s3_answer, BUCKET, _list_meta as list_s3_meta
from mcp_system.azure_blob.client_azure import answer as az_answer, AZ_CONTAINER, _list_meta as list_az_meta
from mcp_system.scanner.metadata_store  import SidecarStore
from mcp_system.scanner.core           import scan_object, _download_head
# Model for planning and merging
MODEL_MERGE = os.getenv("FINAL_MODEL", "claude-3-5-haiku-20241022")
LLM         = Anthropic()
# Pattern returned when a client has no relevant files
NO_RELEVANT_P = re.compile(r"no relevant files", re.I)

def _build_dataset_and_schema_blocks() -> tuple[str, str, list[tuple[str, str]]]:
    dataset_lines = []
    schema_lines = []
    all_files = []
    all_meta = list_s3_meta()
    all_meta += list_az_meta()
    for meta in all_meta:
        path = meta["path"]
        if meta in list_s3_meta():
            cloud = "s3"
            full_path = path
        else:
            cloud = "azure"
            full_path = path
        all_files.append({"path": full_path, "cloud": cloud})

        try:
            head_text, _ = _download_head(path, cloud)
            df = pd.read_csv(io.StringIO(head_text), nrows=500)
            cols = list(df.columns)
            short_desc = f"columns: {', '.join(cols[:5])}"
            dataset_lines.append(f"- {path} ({cloud}): {short_desc}")
            if len(cols) >= 2:
                schema_lines.append(f"- {path} ({cloud}): columns = {', '.join(cols[:10])}")
        except Exception as e:
            head_preview = head_text[:200].replace("\n", "\\n")  # for debugging
            print(f"⚠️ Could not load {path} from {cloud}: {e} — Preview: {head_preview}")

    return "\n".join(dataset_lines), "\n".join(schema_lines), all_files


def _plan(question: str, dataset_block: str, schema_block: str, metas: list) -> Dict[str, List[Dict[str, str]]]:
    files_list = "\n".join(f"- {m['path']} ({m['cloud']})" for m in metas)
    prompt = f"""You are MCP’s expert **multi-dataset** planner.
    Here are the available datasets with their descriptions:{dataset_block}
    Here are their inferred schemas:{schema_block}
    Here is the list of all dataset files that actually exist:{files_list}
    ## User question{question}
    ## Your task Produce a JSON plan with an array `steps`, each step containing:
      - `dataset`: the file path to query next (MUST match exactly one of the files above)
      - `cloud`: "s3" or "azure" — MUST match the correct cloud for that dataset
      - `sub_question`: the exact question to ask that dataset
    Rules:
    1. Only use files listed above — do not invent file names or paths.
    2. Assign each file to its correct cloud: "s3" or "azure".
    3. When datasets share a field (e.g. ID, code, name), plan JOINs across them.
    4. For analytical queries (counts, sums, averages, rankings):
       - Filter by date/category if needed
       - Group by relevant fields
       - Apply appropriate aggregation (count, avg, sum, etc.)
    5. For questions about records *without matches* (e.g. items lacking entries elsewhere):
       - Compare shared keys
       - Count which are missing in the second dataset
    6. When filtering a dataset by a user condition, return all columns for those filtered rows. 
    7. Reference previous step outputs clearly when needed.
    8. If no matching data exists, return: {{\"steps\": []}}
    9. Output JSON only — no extra text or explanation.
    """.strip()
    msg = LLM.messages.create(
        model=MODEL_MERGE,messages=[{"role": "user", "content": prompt}],temperature=0.0,max_tokens=1000,
    )
    raw = "".join(chunk.text for chunk in msg.content).strip()
    return json.loads(raw)


def _augment(subq: str, history: list[str]) -> str:
    if not history:
        return subq
    ctx = (
        "\n\n----- PREVIOUS ANSWERS -----\n"
        + "\n\n".join(f"[step {i+1}]\n{ans}" for i, ans in enumerate(history))
        + "\n-----------------------------\n\n"
    )
    return ctx + subq

def _execute_plan(plan: dict) -> list[tuple[str, str]]:
    """Run each step sequentially, feeding every answer to the next prompt."""
    results, history = [], []
    for step in plan.get("steps", []):
        cloud = step.get("cloud")
        raw_q = step.get("sub_question", "")
        dataset = step.get("dataset")
        subq = _augment(raw_q, history)
        if not cloud or not dataset:
            results.append(("error", f"⚠️ Skipped step: missing cloud or dataset.\n{subq}"))
            continue
        # Properly format prompt using f-string
        enhanced_prompt = f"""
        You are a data extraction expert.
        Your task is to answer the following question using ONLY the contents of the file (if it exists):
        File: {dataset}
        Cloud: {cloud.upper()}
        QUESTION:{subq}
        Return direct results from the file.
        If the file is missing or not available, respond with:
        No data found for file: {dataset}""".strip()
        out = (
            s3_answer(enhanced_prompt, target_path=dataset) if cloud.lower() == "s3"
            else az_answer(enhanced_prompt, target_path=dataset)
        )
        results.append((cloud, out))
        history.append(out)
    return results




def _merge_step(question:  str,prev_ans:  str,this_ans:  str,prev_cloud:str,this_cloud:str,) -> str:
    prompt = f"""
You have two result fragments:
■ From [{prev_cloud}]:   {prev_ans}
■ From [{this_cloud}]:   {this_ans}
**If one fragment contains no actionable facts
IGNORE that fragment and return an answer based solely on the useful one.**
**Your job**:
1. **Extract all facts**, tagging each with its source ([{prev_cloud}] or [{this_cloud}]).
2. **Deduplicate**: merge any duplicate facts (preserving multiple sources).
3. **Check consistency**: for any key item present in one fragment but missing in the other,
   add a "Data Gap: <item> missing from [cloud]" fact.
4. **Synthesize** a cohesive answer (≤180 words) using only those facts.
5. End with one bold **Executive Takeaway** sentence.
Return **plain Markdown only**—no code fences or extra sections.
""".strip()
    msg = LLM.messages.create(
        model      = MODEL_MERGE,
        messages   = [{"role": "user", "content": prompt}],
        temperature= 0.1,
        max_tokens=1000,
    )
    return "".join(chunk.text for chunk in msg.content).strip()


def answer(question: str) -> str:
    """
    Orchestrate planning, execution, and merging for the user question.
    """
    # Pre-scan all files for missing descriptions
    store = SidecarStore()
    for meta in list_s3_meta():
        key = meta["path"]
        full_path = f"{BUCKET}/{key}"
        if not store.get_meta(full_path, cloud="s3"):
            scan_object(full_path, cloud="s3")
    for meta in list_az_meta():
        key = meta["path"]
        full_path = f"{AZ_CONTAINER}/{key}"
        if not store.get_meta(full_path, cloud="azure"):
            scan_object(full_path, cloud="azure")

    # Build the dataset block from sidecar metadata
    metas = store.read_all()
    "\n".join(
        f"- {m.path} ({m.cloud}): {m.description or '⧗ (empty)'}"
        for m in metas
    )

    # Plan the multi-dataset steps
    dataset_block, schema_block, metas = _build_dataset_and_schema_blocks()
    plan = _plan(question, dataset_block, schema_block, metas)
    print("\n··· EXECUTION PLAN ···\n", plan, "\n···\n", file=sys.stderr)
    steps = plan.get("steps", [])
    if not steps:
        return "I couldn't create an execution plan for your query."

    # Execute the plan
    results = _execute_plan(plan)

    # Fast-path: if only one result, return it directly
    if len(results) == 1:
        return results[0][1]

    print("\n⚙️ Execution Results:")
    for idx, (cloud, result) in enumerate(results):
        print(f"\n[Step {idx + 1} from {cloud}]\n{result}\n")

    # Otherwise, merge sequentially with self-checks
    merged     = results[0][1]
    prev_cloud = results[0][0]
    for cloud, out in results[1:]:
        merged     = _merge_step(question, merged, out, prev_cloud, cloud)
        prev_cloud = cloud
    return merged


# ───────────────────── CLI / interactive runner ─────────────────────────
if __name__ == "__main__":
    def run_one(query: str) -> None:
        print("\n— thinking —\n")
        print(textwrap.fill(answer(query), 88), end="\n\n")

    # one-shot mode
    if len(sys.argv) > 1:
        run_one(" ".join(sys.argv[1:]))
        sys.exit(0)

    # REPL mode
    print("💬  MCP interactive mode (orchestrator).  Type 'exit' to quit.\n")
    try:
        while True:
            q = input("query> ").strip()
            if q.lower() in {"exit", "quit"}:
                print("👋  Bye!")
                break
            if q:
                run_one(q)
    except (EOFError, KeyboardInterrupt):
        print("\n👋  Bye!")
