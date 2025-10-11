#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Level 1 + Level 2 prompts over GoEmotions dataset using Ollama (FREE).
Includes tqdm progress bars, data verification, a model sanity test, caching, raw failure dumps,
and post-write previews of saved files.
"""

import os, json, uuid, time, hashlib, asyncio, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import aiohttp
import pandas as pd

# ---------------- CONFIG ----------------

OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

SOURCE   = "csv"                                   # "hf" or "csv"
CSV_PATH = "resources/csv_files/go_emotions_dataset.csv"

MODEL_NAME = "llama3:instruct"                     # e.g., "llama3.1:8b-instruct"
PROVIDER   = "ollama-local"

CONCURRENCY = 4
PACK_N      = 1
MAX_RETRIES = 4
RETRY_BASE  = 2.0
SEED        = 42

DEBUG_FAIL_DIR = os.path.join(OUT_DIR, "raw_fails")
os.makedirs(DEBUG_FAIL_DIR, exist_ok=True)

RUN_LEVEL1          = True
RUN_LEVEL2_VARIANTS = ["improved_basic"]

# ---------------- PROMPTS ----------------
LEVEL1_PROMPT = r"""
SYSTEM / INSTRUCTIONS (Level 1 – Holistic)
You will receive K text items labeled [0]...[K-1]. For each item, produce ONE JSON object in the same order.
Return a SINGLE JSON ARRAY of length K, where element i corresponds to input [i].
Strictly output JSON only. No prose. Do not wrap in code fences.

Schema for each element:
{
  "id": "<string-or-index>",
  "summary": {
    "emotions": [{"subtype":"<EmotionFromList>","score":<0..1>,"valence":"pos|neg|neu"}],
    "notes": "<optional>"
  },
  "confidence": <0..1>
}

Example output for K=2:
[
  {"id":"0","summary":{"emotions":[{"subtype":"Joy","score":0.7,"valence":"pos"}]},"confidence":0.9},
  {"id":"1","summary":{"emotions":[{"subtype":"Sadness","score":0.5,"valence":"neg"}]},"confidence":0.7}
]

Return ONLY a JSON array of length K. If unsure for an item, return:
{"id":"i","summary":{"emotions":[]},"confidence":0.0}
"""

LEVEL2_PROMPTS: Dict[str, str] = {
    "improved_basic": r"""
SYSTEM / INSTRUCTIONS (Level 2 – Span Awareness: Improved Basic)
You will receive K text items labeled [0]...[K-1]. For EACH item, return ONE object that follows the schema below.
Return a SINGLE JSON ARRAY of length K, element i for input [i]. Strict JSON only; no prose or code fences.

Schema per element:
{
  "id": "<string-or-index>",
  "tokens": ["..."],
  "labels": { "EMO": ["O|B-EMO|I-EMO", ...] },
  "spans": [
    {
      "type": "EMO",
      "subtype": "<emotion_from_list>",
      "start": <int>,
      "end": <int>,
      "text": "<exact token span>",
      "attrs": {
        "valence": "pos|neg|neu",
        "intensity": "low|med|high",
        "certainty": "asserted|hedged|negated|hypothetical",
        "temporality": "past|present|future|ongoing|recent",
        "source": "self|other|situation",
        "target_text": "<optional referenced text>",
        "target_relation": "object|cause|recipient",
        "emotion_group": "<optional category>",
        "sentence_index": <optional int>,
        "clause_index": <optional int>,
        "confidence": <optional float>,
        "emotion_cause": "<optional string>",
        "emotion_result": "<optional string>"
      }
    }
  ]
}

If unsure, return a valid empty structure for that item:
{"id":"i","tokens":[],"labels":{"EMO":[]},"spans":[]}

Return ONLY a JSON array of length K.
"""
}

# ---------------- HELPERS ----------------
def is_level1_valid(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict): return False
    if "summary" not in obj: return False
    em = obj["summary"].get("emotions")
    return isinstance(em, list)

def is_level2_valid(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict): return False
    if "tokens" not in obj or "labels" not in obj or "spans" not in obj: return False
    if not isinstance(obj["spans"], list): return False
    return True

def hash_key(**parts) -> str:
    m = hashlib.sha256()
    for k in sorted(parts.keys()):
        v = str(parts[k]).encode("utf-8")
        m.update(k.encode("utf-8") + b"=" + v + b";")
    return m.hexdigest()

def _print_sample_rows(rows: List[Dict[str, Any]], k: int = 5, title: str = "Samples"):
    print(f"\n🧩 {title} (showing up to {k}):")
    for i, r in enumerate(rows[:k]):
        print(f"[{i}] id={r.get('id','NA')} | text={r.get('text','')}")
    print()

def verify_goemotions_access(source: str = None, csv_path: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    src = source or SOURCE
    if src == "hf":
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("`datasets` not installed. Run: pip install datasets") from e

        print("🔍 Verifying Hugging Face GoEmotions access (config='simplified')…")
        try:
            ds = load_dataset("go_emotions", "simplified")
        except Exception as e:
            raise RuntimeError(f"Failed to load GoEmotions from HF Hub: {e}") from e

        print("✅ Splits:", ", ".join([f"{k}={len(v)}" for k, v in ds.items()]))
        for split in ("train", "validation", "test"):
            if split in ds and len(ds[split]) > 0:
                preview = [{"id": str(x.get("id", i)), "text": x["text"]}
                           for i, x in enumerate(ds[split].select(range(min(limit, len(ds[split])))))]
                _print_sample_rows(preview, k=limit, title=f"HF '{split}'")
                return preview
        raise RuntimeError("Dataset loaded but no non-empty split found.")
    elif src == "csv":
        path = csv_path or CSV_PATH
        print(f"🔍 Verifying CSV access at: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found at: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}") from e
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column.")
        n = len(df)
        print(f"✅ CSV rows: {n} | columns: {list(df.columns)}")
        k = min(limit, n)
        preview = [{"id": str(i), "text": df.iloc[i]["text"]} for i in range(k)]
        _print_sample_rows(preview, k=k, title="CSV")
        return preview
    else:
        raise ValueError(f"Unknown SOURCE '{src}'. Use 'hf' or 'csv'.")

def verify_written_file(path: str, expected_rows: int, preview_n: int = 3):
    abs_path = os.path.abspath(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ File not found after write: {abs_path}")
        return
    n = len(lines)
    print(f"📝 Wrote {n} JSONL lines → {abs_path} (expected ~{expected_rows})")
    if n > 0:
        print("🔎 Preview:")
        for i, L in enumerate(lines[:preview_n]):
            print(f"  [{i}] {L.rstrip()[:300]}")

# ---------------- LOAD DATA ----------------
def load_goemotions_texts(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if SOURCE == "hf":
        from datasets import load_dataset
        ds = load_dataset("go_emotions", "simplified")
        rows = []
        for split in ("train", "validation", "test"):
            if split in ds:
                for rec in ds[split]:
                    rid = str(rec.get("id", uuid.uuid4()))
                    rows.append({"id": rid, "text": rec["text"]})
        if limit: rows = rows[:limit]
        return rows
    else:
        df = pd.read_csv(CSV_PATH)
        if "text" not in df.columns:
            raise ValueError("CSV must have 'text' column")
        if limit: df = df.head(limit)
        return [{"id": str(i), "text": t} for i, t in enumerate(df["text"].tolist())]

# ---------------- USER PAYLOAD BUILDER ----------------
SEP = "\n-----\n"

@dataclass
class Inp:
    id: str
    text: str

def pack_items(items: List[Inp], n: int):
    for i in range(0, len(items), n):
        batch = items[i:i+n]
        yield batch

def build_user_payload(batch: List[Inp], sep: str) -> str:
    K = len(batch)
    header = f"K={K}\nEach input is labeled [0]..[{K-1}]. Return ONLY a JSON array of length {K}."
    body = sep.join([f"[{i}] {b.text}" for i, b in enumerate(batch)])
    return header + "\n\n" + body

# ---------------- OLLAMA CALL ----------------
async def llm_call_ollama(session: aiohttp.ClientSession, model: str, system_prompt: str, user_text: str) -> str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_text}
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
            "top_p": 0.0
        }
    }
    async with session.post(url, json=payload, timeout=180) as r:
        r.raise_for_status()
        data = await r.json()
        content = (data.get("message") or {}).get("content", "")
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

# ---------------- JSON COERCION ----------------
def coerce_json(text: str):
    if not text:
        return None
    s = text.strip().replace("“", '"').replace("”", '"').replace("\u200b", "")
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if isinstance(obj, dict):
        return [obj]
    return obj

# ---------------- CACHE ----------------
CACHE_PATH = os.path.join(OUT_DIR, "cache.jsonl")
_cache: Dict[str, Any] = {}

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    _cache[row["key"]] = row["value"]
                except Exception:
                    pass

def save_cache_item(key: str, value: Any):
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
    _cache[key] = value

# ---------------- PROCESS ----------------
async def process_batch(session, batch, prompt_name, prompt_text, level):
    user_text = build_user_payload(batch, SEP)
    key = hash_key(level=level, prompt=prompt_name, model=MODEL_NAME, texts="||".join([x.text for x in batch]))
    if key in _cache:
        return _cache[key]

    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            raw = await llm_call_ollama(session, MODEL_NAME, prompt_text, user_text)
            obj = coerce_json(raw)
            if not isinstance(obj, list):
                raise ValueError("Model did not return a JSON array.")
            if len(obj) != len(batch):
                if len(obj) < len(batch):
                    obj = obj + [{} for _ in range(len(batch)-len(obj))]
                else:
                    obj = obj[:len(batch)]

            validated = []
            for item, out in zip(batch, obj):
                ok = is_level1_valid(out) if level == "level1" else is_level2_valid(out)
                validated.append({
                    "src_id": item.id,
                    "ok": ok,
                    "data": out if ok else {"error": "invalid_schema", "raw": out}
                })
            save_cache_item(key, validated)
            return validated

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raw_text = locals().get("raw", "")
                raw_dump_path = os.path.join(DEBUG_FAIL_DIR, f"{level}_{prompt_name}_{hash_key(texts='||'.join([x.text for x in batch]))}.txt")
                try:
                    with open(raw_dump_path, "w", encoding="utf-8") as fh:
                        fh.write("PROMPT:\n" + prompt_text + "\n\nUSER_TEXT:\n" + user_text + "\n\nRAW:\n" + raw_text)
                except Exception:
                    pass
                return [{"src_id": x.id, "ok": False, "data": {"error": f"failed:{e}"}} for x in batch]
            await asyncio.sleep(delay)
            delay *= RETRY_BASE

# ---------------- RUNNER ----------------
async def run_level(level: str, items: List[Inp], prompts: Dict[str, str], runid: str):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        for pname, ptext in prompts.items():
            print(f"\n▶ Running {level.upper()} with prompt={pname}, items={len(items)}, batch={PACK_N}")

            batches = list(pack_items(items, PACK_N))
            pbar = tqdm(total=len(batches), desc=f"{level}-{pname}", position=0, ncols=100)

            async def do_one(batch):
                async with sem:
                    out = await process_batch(session, batch, pname, ptext, level)
                    pbar.update(1)
                    return out

            tasks = [do_one(b) for b in batches]
            completed = []
            for i in range(0, len(tasks), 100):
                chunk = tasks[i:i+100]
                completed.extend(await asyncio.gather(*chunk))

            pbar.close()

            flat = [x for batch_out in completed for x in batch_out]
            ok_rows = [r for r in flat if r["ok"]]
            err_rows = [r for r in flat if not r["ok"]]
            tqdm.write(f"  ✓ {len(ok_rows)} OK, {len(err_rows)} failed")

            # 👀 Print a sample OK row before writing
            if ok_rows:
                print("👀 Sample OK row:", json.dumps(ok_rows[0], ensure_ascii=False)[:300])

            out_path = os.path.join(OUT_DIR, f"goemotions_{level}_{pname}_{MODEL_NAME.replace(':','-')}_{runid}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for r in ok_rows:
                    f.write(json.dumps({
                        "src_id": r["src_id"],
                        "model": MODEL_NAME,
                        "provider": PROVIDER,
                        "prompt": pname,
                        "level": level,
                        "data": r["data"]
                    }, ensure_ascii=False) + "\n")
            print(f"Saved → {out_path}")

            # 🧾 Verify file content and print preview lines
            verify_written_file(out_path, expected_rows=len(ok_rows), preview_n=3)

# ---------------- SANITY MODEL TEST ----------------
async def sanity_model_test():
    """
    Minimal connectivity + JSON-format sanity check (K=1).
    Expect a JSON array of length 1 back.
    """
    test_items = [Inp(id="0", text="sample text 0")]  # K=1
    sys_prompt = (
        "SYSTEM\n"
        "You will receive K items labeled [0]..[K-1]. "
        "Return ONLY a JSON array of length K. No prose, no code fences.\n"
        'Each element i must be {"id":"<i>","ok":true}.\n'
    )
    user_text = build_user_payload(test_items, "\n-----\n")  # K=1

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            raw = await llm_call_ollama(session, MODEL_NAME, sys_prompt, user_text)
    except aiohttp.ClientConnectorError:
        print("   Start it with:  ollama serve")
        print("   Then pull a model, e.g.:  ollama pull llama3:instruct")
        return False

    obj = coerce_json(raw)
    print("\n🔎 Sanity model test raw (truncated):", (raw[:200].replace("\n"," ") + ("…" if len(raw) > 200 else "")))
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or len(obj) != 1 or not (isinstance(obj[0], dict) and "id" in obj[0]):
        print("❌ Sanity model test FAILED. The model did not return the required JSON array of length 1.")
        return False
    print("✅ Sanity model test PASSED:", obj)
    return True

# ---------------- MAIN ----------------
def write_meta(runid, items_count):
    meta = {
        "runid": runid,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "provider": PROVIDER,
        "rows": items_count,
        "pack_n": PACK_N,
        "concurrency": CONCURRENCY
    }
    with open(os.path.join(OUT_DIR, f"meta_{runid}.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main(limit=None):
    random.seed(SEED)
    load_cache()

    # Verify data access + preview
    try:
        _ = verify_goemotions_access(SOURCE, CSV_PATH, limit=5)
    except Exception as e:
        print(f"\n❌ Data verification failed: {e}")
        return

    # Sanity test the model/JSON behavior
    loop = asyncio.get_event_loop()
    ok = loop.run_until_complete(sanity_model_test())
    if not ok:
        print("➡️  Tip: keep PACK_N=1 and try model 'llama3.2:3b-instruct' if needed.")
        return

    # Load actual rows for the run (honors `limit`)
    rows = load_goemotions_texts(limit)
    if not rows:
        print("❌ No rows loaded. Check SOURCE/CSV_PATH.")
        return

    print(f"🚚 Will process {len(rows)} rows (limit={limit})")
    _print_sample_rows(rows, k=min(5, len(rows)), title="Run Preview")

    items = [Inp(**r) for r in rows]
    runid = str(uuid.uuid4())[:8]
    write_meta(runid, len(items))

    if RUN_LEVEL1:
        loop.run_until_complete(run_level("level1", items, {"holistic": LEVEL1_PROMPT}, runid))

    if RUN_LEVEL2_VARIANTS:
        prompts = {v: LEVEL2_PROMPTS[v] for v in RUN_LEVEL2_VARIANTS}
        loop.run_until_complete(run_level("level2", items, prompts, runid))

    print("\n✅ All done!")

if __name__ == "__main__":
    # test small subset first
    main(limit=50)
    # main()