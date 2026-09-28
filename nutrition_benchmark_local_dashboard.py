"""Nutrition AI benchmark - local Streamlit dashboard.

Run:  streamlit run nutrition_benchmark_local_dashboard.py
"""
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import anthropic
except ImportError:
    anthropic = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "benchmark.sqlite3"
DEFAULT_VECTOR_STORE = os.environ.get("OPENAI_VECTOR_STORE_ID", "")

DATA_DIR.mkdir(exist_ok=True)

RESPONSE_COLUMNS = [
    "id", "run_id", "question_id", "config_id", "provider", "model", "status",
    "answer", "raw_json", "retrieved_json", "input_tokens", "output_tokens",
    "latency_ms", "attempts", "error", "created_at",
]
SCORE_COLUMNS = [
    "id", "response_id", "method", "earned", "maximum", "normalized",
    "details_json", "created_at",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass  # another (older) process may hold the DB; plain journal mode still works
    c.execute("PRAGMA busy_timeout=30000")
    return c


def _columns(c, table):
    return [r[1] for r in c.execute("PRAGMA table_info(%s)" % table).fetchall()]


def init_db():
    """Create tables, recreating any table whose schema is stale.

    The shipped DB was written by an older version with a different column
    list; CREATE TABLE IF NOT EXISTS silently kept it and every INSERT failed.
    """
    c = conn()
    ddl = {
        "runs": (
            "CREATE TABLE runs (id TEXT PRIMARY KEY, status TEXT, total INTEGER, "
            "completed INTEGER DEFAULT 0, created_at TEXT, config_json TEXT, error TEXT)"
        ),
        "responses": (
            "CREATE TABLE responses (id TEXT PRIMARY KEY, run_id TEXT, question_id TEXT, "
            "config_id TEXT, provider TEXT, model TEXT, status TEXT, answer TEXT, "
            "raw_json TEXT, retrieved_json TEXT, input_tokens INTEGER, output_tokens INTEGER, "
            "latency_ms REAL, attempts INTEGER, error TEXT, created_at TEXT, "
            "UNIQUE(run_id, question_id, config_id))"
        ),
        "scores": (
            "CREATE TABLE scores (id TEXT PRIMARY KEY, response_id TEXT UNIQUE, method TEXT, "
            "earned REAL, maximum REAL, normalized REAL, details_json TEXT, created_at TEXT)"
        ),
    }
    expected = {
        "runs": ["id", "status", "total", "completed", "created_at", "config_json", "error"],
        "responses": RESPONSE_COLUMNS,
        "scores": SCORE_COLUMNS,
    }
    for table, statement in ddl.items():
        existing = _columns(c, table)
        if existing and existing != expected[table]:
            rows = c.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
            if rows:
                c.execute("ALTER TABLE %s RENAME TO %s_legacy_%s" % (table, table, int(time.time())))
            else:
                c.execute("DROP TABLE %s" % table)
            existing = []
        if not existing:
            c.execute(statement)
    c.commit()
    c.close()


init_db()


def insert(c, table, columns, values):
    c.execute(
        "INSERT OR REPLACE INTO %s (%s) VALUES (%s)"
        % (table, ",".join(columns), ",".join("?" * len(columns))),
        values,
    )


# --------------------------------------------------------------------------- input

def parse_questions(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    obj = json.loads(raw)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("questions", "items", "data", "records"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError("JSON must be an array or an object with a questions/items/data/records array")


QUESTION_SUFFIXES = (".json", ".txt", ".ndjson")


def discover_question_files():
    """Every file in data/ (then the app folder) that parses as a question set.

    Ordered by preference so the default pick is a real .json question set:
    data/ before the app folder, .json before .txt/.ndjson, then by name. The
    first file wins ties, and any later file with identical content is labelled
    a duplicate.

    Returns [{path, name, label, questions}] plus a list of files that looked
    like question sets but failed to parse, so bad files are reported rather
    than silently skipped.
    """
    candidates = []
    for rank, directory in enumerate((DATA_DIR, BASE_DIR)):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in QUESTION_SUFFIXES:
                candidates.append(((rank, path.suffix.lower() != ".json", path.name.lower()), path))
    candidates.sort(key=lambda item: item[0])

    found, broken, seen = [], [], set()
    for _, path in candidates:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except Exception as e:
            broken.append("%s: %s" % (path.name, e))
            continue
        if raw.lstrip()[:1] not in ("[", "{"):
            continue  # not JSON at all; not a question set
        try:
            questions = parse_questions(raw)
        except Exception as e:
            broken.append("%s: %s" % (path.name, e))
            continue
        if not questions:
            continue
        fingerprint = (len(questions), json.dumps(questions[0], sort_keys=True)[:400])
        duplicate = fingerprint in seen
        seen.add(fingerprint)
        relative = path.relative_to(BASE_DIR) if BASE_DIR in path.parents else path
        found.append({
            "path": path,
            "name": path.name,
            "questions": questions,
            "label": "%s (%d questions)%s" % (relative, len(questions),
                                              " - duplicate" if duplicate else ""),
        })
    return found, broken


def question_folder_signature():
    """Cheap fingerprint of the question files, so the cache drops when they change."""
    items = []
    for directory in (DATA_DIR, BASE_DIR):
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in QUESTION_SUFFIXES:
                stat = path.stat()
                items.append((str(path), stat.st_mtime, stat.st_size))
    return tuple(items)


@st.cache_data(show_spinner=False)
def cached_question_files(signature):
    """Parsing two 400KB+ files on every rerun made the UI feel frozen."""
    return discover_question_files()


def load_prompt(uploaded):
    if uploaded:
        return uploaded.getvalue().decode("utf-8")
    for path in (DATA_DIR / "nutrition_prompt.txt", DATA_DIR / "nutrition_prompt.md",
                 BASE_DIR / "nutrition_prompt.txt"):
        if path.exists():
            return path.read_text(encoding="utf-8")
    return "Answer accurately and directly. Follow the question's requested format."


FORMAT_RULES = {
    "multiple_choice": (
        "Answer with the single letter of the correct option. "
        "Put the letter alone on the final line, formatted exactly as: ANSWER: X"
    ),
    "calculation": (
        "Show your working, then put the final numeric value alone on the last line, "
        "formatted exactly as: ANSWER: <number>  (digits only, no units, no ranges)"
    ),
    "open_ended": "Answer concisely and completely, covering every part of the question.",
}


def build_user_content(q):
    """Render a question the way the model must see it (options included)."""
    parts = [q.get("question", "")]
    options = q.get("options")
    if isinstance(options, dict) and options:
        parts.append("\nOptions:")
        parts.extend("%s. %s" % (k, v) for k, v in sorted(options.items()))
    elif isinstance(options, list) and options:
        parts.append("\nOptions:")
        parts.extend("%s. %s" % (chr(65 + i), v) for i, v in enumerate(options))
    rule = FORMAT_RULES.get(q.get("type", ""))
    if rule:
        parts.append("\n" + rule)
    return "\n".join(parts)


# --------------------------------------------------------------------------- providers

def text_of(response):
    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", None) or getattr(response, "content", None) or []:
        if getattr(item, "type", "") == "text":
            chunks.append(getattr(item, "text", "") or "")
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", "") in ("output_text", "text"):
                chunks.append(getattr(block, "text", "") or "")
    return "\n".join(x for x in chunks if x)


def dump(response):
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {"repr": repr(response)}


def usage(response):
    u = getattr(response, "usage", None)
    if not u:
        return None, None
    inp = getattr(u, "input_tokens", None)
    out = getattr(u, "output_tokens", None)
    if inp is None:
        inp = getattr(u, "prompt_tokens", None)
    if out is None:
        out = getattr(u, "completion_tokens", None)
    return inp, out


def retrieved_of(raw):
    """Pull file_search results out of a Responses API payload, if any."""
    out = []
    for item in (raw or {}).get("output", []) or []:
        if item.get("type") == "file_search_call":
            for r in item.get("results") or []:
                out.append({
                    "file_id": r.get("file_id"),
                    "filename": r.get("filename"),
                    "score": r.get("score"),
                    "text": (r.get("text") or "")[:2000],
                })
    return out


def call_openai(key, model, prompt, question, vector_store=None, max_tokens=2048):
    if OpenAI is None:
        raise RuntimeError("Install openai: python -m pip install openai")
    if not key:
        raise RuntimeError("Missing OpenAI API key")
    client = OpenAI(api_key=key, timeout=180.0)
    kwargs = dict(model=model, instructions=prompt, input=question, max_output_tokens=max_tokens)
    if vector_store:
        kwargs["tools"] = [{"type": "file_search", "vector_store_ids": [vector_store]}]
    r = client.responses.create(**kwargs)
    raw = dump(r)
    answer = text_of(r)
    if not answer.strip():
        reason = (raw.get("incomplete_details") or {}).get("reason")
        if raw.get("status") == "incomplete":
            raise RuntimeError(
                "Model returned no text (status=incomplete, reason=%s). "
                "Raise 'Max output tokens' - reasoning models spend the budget on thinking." % reason
            )
        raise RuntimeError("Model returned an empty answer (status=%s)" % raw.get("status"))
    return answer, raw, retrieved_of(raw), usage(r)


def call_anthropic(key, model, prompt, question, max_tokens=2048, workspace_id=None):
    if anthropic is None:
        raise RuntimeError("Install anthropic: python -m pip install anthropic")
    if not key:
        raise RuntimeError("Missing Anthropic API key")
    # Identity-linked (OAuth/console) keys are rejected without a workspace id.
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(api_key=key, timeout=180.0, default_headers=headers)
    r = client.messages.create(
        model=model, system=prompt, max_tokens=max_tokens,
        messages=[{"role": "user", "content": question}],
    )
    answer = "\n".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    if not answer.strip():
        raise RuntimeError("Model returned an empty answer (stop_reason=%s)" % getattr(r, "stop_reason", None))
    u = getattr(r, "usage", None)
    return answer, dump(r), [], (getattr(u, "input_tokens", None), getattr(u, "output_tokens", None))


def run_config(cfg, keys, question_text, max_tokens):
    if cfg["provider"] == "openai_rag":
        return call_openai(keys.get("openai"), cfg["model"], cfg["prompt"], question_text,
                           cfg.get("vector_store"), max_tokens)
    if cfg["provider"] == "openai_direct":
        return call_openai(keys.get("openai"), cfg["model"], cfg["prompt"], question_text,
                           None, max_tokens)
    return call_anthropic(keys.get("anthropic"), cfg["model"], cfg["prompt"], question_text,
                          max_tokens, keys.get("anthropic_workspace"))


# --------------------------------------------------------------------------- scoring

ANSWER_TAG = re.compile(r"ANSWER\s*[:\-]\s*(.+)", re.IGNORECASE)


def tagged_answer(text):
    matches = ANSWER_TAG.findall(text or "")
    return matches[-1].strip() if matches else None


def letter(text, options=None):
    """Extract the chosen option letter, preferring an explicit ANSWER: tag."""
    valid = set(options.keys()) if isinstance(options, dict) and options else set("ABCDEFGH")
    valid = set(x.upper() for x in valid)
    tag = tagged_answer(text)
    for candidate in (tag, text):
        if not candidate:
            continue
        m = re.match(r"^\W*([A-H])\b", candidate.strip().upper())
        if m and m.group(1) in valid:
            return m.group(1)
    # last resort: a standalone letter on the final non-empty line
    for line in reversed((text or "").strip().splitlines()):
        m = re.search(r"\b([A-H])\b", line.upper())
        if m and m.group(1) in valid:
            return m.group(1)
    return None


def number(text):
    """Final numeric value: prefer the ANSWER: tag, else the last number in the text."""
    for candidate in (tagged_answer(text), text):
        if not candidate:
            continue
        cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", candidate)
        values = re.findall(r"[-+]?\d*\.?\d+", cleaned)
        if values:
            return float(values[-1])
    return None


JUDGE_SYSTEM = (
    "You are a strict registered-dietitian grader. You are given a question, a list of rubric "
    "points, and a candidate answer. For each rubric point decide whether the candidate answer "
    "clearly satisfies it. Be strict: partial hints do not count. "
    'Reply with JSON only: {"hits": [true/false, ...], "notes": "one short sentence"}. '
    "The hits array must have exactly one entry per rubric point, in order."
)


def parse_json_object(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in judge reply: %s" % text[:200])
    return json.loads(text[start:end + 1])


def judge_open_ended(q, answer, judge_cfg, keys, max_tokens=1024):
    points = q.get("rubric_points") or []
    if not points:
        return "no_rubric", 0.0, float(q.get("max_points") or 1), {"note": "question has no rubric_points"}
    payload = json.dumps({
        "question": q.get("question"),
        "rubric_points": points,
        "candidate_answer": answer,
    }, ensure_ascii=False)
    reply, _, _, _ = run_config(
        {"provider": judge_cfg["provider"], "model": judge_cfg["model"],
         "prompt": JUDGE_SYSTEM, "vector_store": ""},
        keys, payload, max_tokens,
    )
    parsed = parse_json_object(reply)
    hits = [bool(x) for x in parsed.get("hits", [])][:len(points)]
    hits += [False] * (len(points) - len(hits))
    maximum = float(q.get("max_points") or len(points))
    earned = round(maximum * (sum(hits) / float(len(points))), 4)
    return "llm_rubric", earned, maximum, {
        "hits": hits, "notes": parsed.get("notes", ""), "judge_model": judge_cfg["model"],
    }


def score(q, answer, judge_cfg=None, keys=None):
    maximum = float(q.get("max_points") or 1)
    typ = (q.get("type") or "").lower()
    rule = (q.get("scoring_rule") or "").lower()
    expected_raw = q.get("correct_answer", q.get("correctanswer"))

    if typ in ("multiple_choice", "multiplechoice") or rule in ("exact_match", "exactmatch"):
        expected = str(expected_raw or "").strip().upper()[:1]
        actual = letter(answer, q.get("options"))
        ok = actual is not None and actual == expected
        return "exact_match", maximum if ok else 0.0, maximum, {"expected": expected, "actual": actual}

    if typ == "calculation" or rule in ("numeric_tolerance", "numerictolerance"):
        actual = number(answer)
        try:
            expected = float(expected_raw)
        except (TypeError, ValueError):
            return "error", 0.0, maximum, {"error": "correct_answer is not numeric: %r" % (expected_raw,)}
        try:
            tolerance = float(q.get("numeric_tolerance") or q.get("numerictolerance") or 0)
        except (TypeError, ValueError):
            tolerance = 0.0
        ok = actual is not None and abs(actual - expected) <= tolerance
        return "numeric_tolerance", maximum if ok else 0.0, maximum, {
            "expected": expected, "actual": actual, "tolerance": tolerance}

    if judge_cfg:
        return judge_open_ended(q, answer, judge_cfg, keys)
    return "rubric_pending", 0.0, maximum, {"note": "open-ended: enable the LLM judge to score"}


def write_score(c, response_id, q, answer, judge_cfg, keys):
    try:
        method, earned, maximum, details = score(q, answer, judge_cfg, keys)
    except Exception as e:
        method, earned, maximum, details = "error", 0.0, float(q.get("max_points") or 1), {"error": str(e)}
    insert(c, "scores", SCORE_COLUMNS, (
        str(uuid.uuid4()), response_id, method, earned, maximum,
        (earned / maximum) if maximum else 0.0, json.dumps(details, ensure_ascii=False), now(),
    ))
    return method


# --------------------------------------------------------------------------- execution

def save_result(run_id, q, cfg, answer, raw, retrieved, in_tok, out_tok, latency,
                attempts, status, error, judge_cfg, keys):
    c = conn()
    try:
        rid = str(uuid.uuid4())
        existing = c.execute(
            "SELECT id FROM responses WHERE run_id=? AND question_id=? AND config_id=?",
            (run_id, q["id"], cfg["id"])).fetchone()
        if existing:
            rid = existing["id"]
            c.execute("DELETE FROM scores WHERE response_id=?", (rid,))
        insert(c, "responses", RESPONSE_COLUMNS, (
            rid, run_id, q["id"], cfg["id"], cfg["provider"], cfg["model"], status, answer,
            json.dumps(raw, ensure_ascii=False, default=str),
            json.dumps(retrieved, ensure_ascii=False, default=str),
            in_tok, out_tok, latency, attempts, error, now(),
        ))
        if status == "completed":
            write_score(c, rid, q, answer, judge_cfg, keys)
        c.execute(
            "UPDATE runs SET completed = (SELECT COUNT(*) FROM responses WHERE run_id=?) WHERE id=?",
            (run_id, run_id))
        c.commit()
    finally:
        c.close()


def execute(run_id, questions, configs, keys, max_tokens, retries, concurrency, judge_cfg):
    jobs = [(q, cfg) for q in questions for cfg in configs]

    def one(q, cfg):
        c = conn()
        try:
            row = c.execute(
                "SELECT status FROM responses WHERE run_id=? AND question_id=? AND config_id=?",
                (run_id, q["id"], cfg["id"])).fetchone()
        finally:
            c.close()
        if row and row["status"] == "completed":
            return
        content = build_user_content(q)
        err = ""
        for attempt in range(retries + 1):
            start = time.perf_counter()
            try:
                answer, raw, retrieved, tokens = run_config(cfg, keys, content, max_tokens)
                save_result(run_id, q, cfg, answer, raw, retrieved, tokens[0], tokens[1],
                            (time.perf_counter() - start) * 1000.0, attempt + 1,
                            "completed", "", judge_cfg, keys)
                return
            except Exception as e:
                err = "%s: %s" % (type(e).__name__, e)
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
        save_result(run_id, q, cfg, "", {}, [], None, None, None, retries + 1,
                    "failed", err, judge_cfg, keys)

    c = conn()
    c.execute("UPDATE runs SET status='running', error='' WHERE id=?", (run_id,))
    c.commit()
    c.close()
    fatal = ""
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, q, cfg) for q, cfg in jobs]
            for f in futures:
                f.result()
    except Exception as e:
        fatal = "%s: %s" % (type(e).__name__, e)
    c = conn()
    c.execute("UPDATE runs SET status=?, error=? WHERE id=?",
              ("failed" if fatal else "completed", fatal, run_id))
    c.commit()
    c.close()


def rescore_run(run_id, questions_by_id, judge_cfg, keys):
    c = conn()
    rows = c.execute("SELECT * FROM responses WHERE run_id=? AND status='completed'", (run_id,)).fetchall()
    c.close()
    for row in rows:
        q = questions_by_id.get(row["question_id"])
        if not q:
            continue
        c = conn()
        try:
            c.execute("DELETE FROM scores WHERE response_id=?", (row["id"],))
            write_score(c, row["id"], q, row["answer"], judge_cfg, keys)
            c.commit()
        finally:
            c.close()


# --------------------------------------------------------------------------- UI

def fetch_rows(run_id):
    c = conn()
    rows = c.execute(
        "SELECT r.question_id, r.config_id, r.provider, r.model, r.status, r.answer, "
        "r.input_tokens, r.output_tokens, r.latency_ms, r.attempts, r.error, r.id, "
        "s.method, s.earned, s.maximum, s.normalized, s.details_json "
        "FROM responses r LEFT JOIN scores s ON s.response_id=r.id "
        "WHERE r.run_id=? ORDER BY r.question_id, r.config_id", (run_id,)).fetchall()
    c.close()
    return [dict(x) for x in rows]


def leaderboard(df, questions_by_id):
    """Per-config summary. Every config appears, even if all of its calls failed."""
    df = df.copy()
    df["type"] = df["question_id"].map(lambda i: (questions_by_id.get(i) or {}).get("type"))
    df["domain"] = df["question_id"].map(lambda i: (questions_by_id.get(i) or {}).get("domain"))
    scorable = df["maximum"].notna() & (df["method"] != "rubric_pending") & (df["method"] != "error")

    records = []
    for config_id, group in df.groupby("config_id"):
        scored = group[scorable.reindex(group.index, fill_value=False)]
        max_points = scored["maximum"].sum()
        latency = group["latency_ms"].dropna()
        records.append({
            "config_id": config_id,
            "model": group["model"].iloc[0],
            "calls": int(len(group)),
            "failed": int((group["status"] == "failed").sum()),
            "scored": int(len(scored)),
            "points": round(float(scored["earned"].sum()), 2),
            "max_points": round(float(max_points), 2),
            "score_%": round(100.0 * scored["earned"].sum() / max_points, 1) if max_points else None,
            "avg_latency_s": round(float(latency.mean()) / 1000.0, 2) if len(latency) else None,
            "in_tokens": int(group["input_tokens"].fillna(0).sum()),
            "out_tokens": int(group["output_tokens"].fillna(0).sum()),
        })
    overall = pd.DataFrame(records).sort_values("score_%", ascending=False, na_position="last")

    scored_all = df[scorable]
    by_type = None
    if not scored_all.empty:
        by_type = (scored_all.pivot_table(index="type", columns="config_id",
                                          values="normalized", aggfunc="mean") * 100).round(1)
    return overall, by_type, df


def run_snapshot(run_id):
    c = conn()
    try:
        run = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        counts = {r["status"]: r["n"] for r in c.execute(
            "SELECT status, COUNT(*) n FROM responses WHERE run_id=? GROUP BY status",
            (run_id,)).fetchall()}
    finally:
        c.close()
    return run, counts


def render_progress(run_id):
    """Live progress.

    Refreshing used to be `time.sleep(2); st.rerun()`, which re-ran the whole
    script - re-parsing every question file - every 2s, so the page sat in a
    permanent 'running' state and the controls were unusable. A fragment
    refreshes only this panel, and stops once the run is no longer active.
    """
    run, counts = run_snapshot(run_id)
    if not run:
        return
    is_active = run["status"] in ("queued", "running")
    worker = st.session_state.get("worker")
    worker_dead = is_active and worker is not None and not worker.is_alive()

    def panel():
        current, current_counts = run_snapshot(run_id)
        if not current:
            return
        completed = sum(current_counts.values())
        total = current["total"] or 1
        st.progress(min(completed / float(total), 1.0),
                    text="Run %s - %s - %d/%d done (%d failed)"
                         % (run_id[:8], current["status"], completed, total,
                            current_counts.get("failed", 0)))
        if current["error"]:
            st.error(current["error"])
        if current["status"] == "completed":
            st.success("Run finished - open the Results tab.")
        elif current["status"] == "failed":
            st.error("Run failed. See the Results tab for per-call errors.")

    if worker_dead:
        panel()
        st.error("The worker thread stopped before finishing this run. "
                 "Use 'Retry failed / missing calls' in the Results tab - completed "
                 "answers are kept and only the missing calls are re-sent.")
        return
    st.fragment(run_every=2 if is_active else None)(panel)()


def main():
    st.set_page_config(page_title="Nutrition AI Benchmark", layout="wide")
    st.title("Nutrition AI Benchmark")

    with st.sidebar:
        st.header("Data")
        found, broken = cached_question_files(question_folder_signature())
        questions, source, source_error = [], None, None
        if found:
            labels = [f["label"] for f in found]
            picked = st.selectbox("Question set file", labels, index=0,
                                  help="Auto-loaded from %s and %s" % (DATA_DIR.name, BASE_DIR.name))
            chosen = found[labels.index(picked)]
            questions, source = chosen["questions"], chosen["name"]
        else:
            st.warning("No question files found in %s. Upload one below." % DATA_DIR)
        uploaded_q = st.file_uploader("...or upload a questions file", type=["json", "txt"])
        if uploaded_q:
            try:
                questions = parse_questions(uploaded_q.getvalue())
                source, source_error = uploaded_q.name, None
            except Exception as e:
                source_error = str(e)
        uploaded_p = st.file_uploader("Production prompt", type=["txt", "md"])
        prompt = load_prompt(uploaded_p)
        st.write("Questions loaded: **%d**" % len(questions))
        if source:
            st.caption("Source: %s" % source)
        if source_error:
            st.error(source_error)
        if broken:
            with st.expander("Files that failed to parse (%d)" % len(broken)):
                for line in broken:
                    st.caption(line)

        st.header("Credentials")
        openai_key = st.text_input("OpenAI API key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
        anthropic_key = st.text_input("Anthropic API key", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password")
        anthropic_workspace = st.text_input(
            "Anthropic workspace ID (optional)",
            value=os.environ.get("ANTHROPIC_WORKSPACE_ID", ""),
            help="Required if your key is identity-linked; the API then returns "
                 "'anthropic-workspace-id is required'. Find it in the Console workspace URL.")

        st.header("Models")
        vector_store = st.text_input("Vector store", DEFAULT_VECTOR_STORE)
        rag_model = st.text_input("RAG model (OpenAI)", "o3-mini")
        claude_model = st.text_input("Claude model", "claude-opus-5")
        sol_model = st.text_input("Second OpenAI model", "gpt-5.6-sol")
        max_tokens = int(st.number_input("Max output tokens", 256, 32768, 4096, 256,
                                         help="Reasoning models spend this budget on thinking; too low returns an empty answer."))
        retries = int(st.number_input("Retries", 0, 5, 2))
        concurrency = int(st.number_input("Concurrency", 1, 16, 4))

        st.header("Open-ended grading")
        use_judge = st.checkbox("Score open-ended with an LLM judge", value=True)
        judge_provider = st.selectbox("Judge provider", ["anthropic", "openai_direct"], disabled=not use_judge)
        judge_model = st.text_input("Judge model", claude_model if judge_provider == "anthropic" else sol_model,
                                    disabled=not use_judge)

    questions_by_id = {q["id"]: q for q in questions}
    keys = {"openai": openai_key, "anthropic": anthropic_key,
            "anthropic_workspace": anthropic_workspace.strip()}
    judge_cfg = {"provider": judge_provider, "model": judge_model} if use_judge else None
    configs = [
        {"id": "openai_rag", "provider": "openai_rag", "model": rag_model,
         "vector_store": vector_store, "prompt": prompt},
        {"id": "claude_direct", "provider": "anthropic", "model": claude_model, "vector_store": "",
         "prompt": "Answer accurately and directly. Follow the question and its requested format. Do not use external tools."},
        {"id": "openai_direct", "provider": "openai_direct", "model": sol_model, "vector_store": "",
         "prompt": "Answer accurately and directly. Follow the question and its requested format. Do not use external tools."},
    ]

    run_tab, results_tab, preview_tab = st.tabs(["Run benchmark", "Results", "Question preview"])

    with run_tab:
        col1, col2 = st.columns([2, 1])
        with col1:
            choice = st.selectbox("Question set", [
                "Pilot: first 20", "All questions", "Easy only", "Medium only", "Hard only",
                "Multiple choice only", "Calculation only", "Open ended only"])
        with col2:
            limit = int(st.number_input("Limit (0 = no limit)", 0, 10000, 0))
        if choice == "Pilot: first 20":
            selected = questions[:20]
        elif choice == "All questions":
            selected = questions
        elif choice.endswith("only") and choice.split()[0] in ("Easy", "Medium", "Hard"):
            selected = [q for q in questions if (q.get("difficulty") or "").lower() == choice.split()[0].lower()]
        else:
            wanted = {"Multiple choice only": "multiple_choice", "Calculation only": "calculation",
                      "Open ended only": "open_ended"}[choice]
            selected = [q for q in questions if q.get("type") == wanted]
        if limit:
            selected = selected[:limit]
        st.write("Selected: **%d** questions - **%d** model calls (%d configs)"
                 % (len(selected), len(selected) * len(configs), len(configs)))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Preflight check", disabled=not selected):
                probe = selected[0]
                for cfg in configs:
                    try:
                        answer, _, _, _ = run_config(cfg, keys, build_user_content(probe), max_tokens)
                        st.success("%s (%s): ok - %s" % (cfg["id"], cfg["model"], answer.strip()[:120]))
                    except Exception as e:
                        st.error("%s (%s): %s" % (cfg["id"], cfg["model"], e))
                if judge_cfg:
                    try:
                        judge_open_ended({"question": "2+2?", "rubric_points": ["says 4"], "max_points": 1},
                                         "The answer is 4.", judge_cfg, keys)
                        st.success("judge (%s): ok" % judge_cfg["model"])
                    except Exception as e:
                        st.error("judge (%s): %s" % (judge_cfg["model"], e))
        with c2:
            start = st.button("Start benchmark", type="primary", disabled=not selected)

        if start:
            if not openai_key or not anthropic_key:
                st.error("Enter both API keys.")
            else:
                run_id = str(uuid.uuid4())
                c = conn()
                insert(c, "runs", ["id", "status", "total", "completed", "created_at", "config_json", "error"],
                       (run_id, "queued", len(selected) * len(configs), 0, now(),
                        json.dumps([{k: v for k, v in cfg.items() if k != "prompt"} for cfg in configs]), ""))
                c.commit()
                c.close()
                worker = threading.Thread(
                    target=execute,
                    args=(run_id, selected, configs, keys, max_tokens, retries, concurrency, judge_cfg),
                    daemon=True)
                worker.start()
                st.session_state["active_run"] = run_id
                st.session_state["worker"] = worker
                st.success("Started run %s" % run_id[:8])

        active = st.session_state.get("active_run")
        if active:
            render_progress(active)

    with results_tab:
        c = conn()
        runs = c.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        c.close()
        if not runs:
            st.info("No runs yet.")
        else:
            labels = {"%s | %s | %s" % (r["id"][:8], r["created_at"][:19], r["status"]): r["id"] for r in runs}
            label = st.selectbox("Run", list(labels.keys()))
            rid = labels[label]
            rows = fetch_rows(rid)
            if not rows:
                st.info("No responses recorded for this run yet.")
            else:
                df = pd.DataFrame(rows)
                overall, by_type, df = leaderboard(df, questions_by_id)
                pending = int((df["method"] == "rubric_pending").sum())
                if pending:
                    st.warning("%d open-ended answers are unscored. Enable the LLM judge and press "
                               "'Re-score this run'." % pending)
                # st.table (static) rather than st.dataframe: dataframes inside
                # st.tabs render collapsed to ~0 width until the user interacts.
                st.subheader("Leaderboard")
                st.table(overall.set_index("config_id").style.format({
                    "calls": "{:.0f}", "failed": "{:.0f}", "scored": "{:.0f}",
                    "points": "{:.2f}", "max_points": "{:.2f}", "score_%": "{:.1f}",
                    "avg_latency_s": "{:.2f}", "in_tokens": "{:,.0f}", "out_tokens": "{:,.0f}",
                }, na_rep="-"))
                if by_type is not None:
                    st.subheader("Score % by question type")
                    st.table(by_type.style.format("{:.1f}", na_rep="-"))
                failures = df[df["status"] == "failed"]
                if not failures.empty:
                    with st.expander("Failed calls (%d)" % len(failures)):
                        st.dataframe(failures[["question_id", "config_id", "model", "attempts", "error"]],
                                     width="stretch", hide_index=True)
                st.subheader("Responses")
                st.dataframe(
                    df[["question_id", "config_id", "model", "status", "method", "earned", "maximum",
                        "latency_ms", "input_tokens", "output_tokens", "answer", "error"]],
                    width="stretch", hide_index=True)
                b1, b2 = st.columns(2)
                with b1:
                    st.download_button("Download results (JSON)",
                                       json.dumps(rows, ensure_ascii=False, indent=2),
                                       "benchmark-%s.json" % rid[:8], "application/json")
                with b2:
                    st.download_button("Download results (CSV)", df.to_csv(index=False),
                                       "benchmark-%s.csv" % rid[:8], "text/csv")
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("Re-score this run",
                                 help="Re-applies scoring (and the LLM judge) to saved answers - no API calls for the models"):
                        with st.spinner("Re-scoring..."):
                            rescore_run(rid, questions_by_id, judge_cfg, keys)
                        st.rerun()
                with r2:
                    run_row = next((r for r in runs if r["id"] == rid), None)
                    outstanding = int((df["status"] == "failed").sum())
                    expected = (run_row["total"] or 0) if run_row else 0
                    missing = max(expected - len(df), 0)
                    if st.button("Retry failed / missing calls (%d)" % (outstanding + missing),
                                 disabled=(outstanding + missing) == 0,
                                 help="Re-sends only the calls that failed, into the same run. "
                                      "Completed answers are kept."):
                        retry_ids = list(df[df["status"] == "failed"]["question_id"].unique())
                        retry_questions = [questions_by_id[i] for i in retry_ids if i in questions_by_id]
                        if not retry_questions:
                            st.warning("Those question ids are not in the loaded question set.")
                        else:
                            c = conn()
                            c.execute("UPDATE runs SET status='queued', error='' WHERE id=?", (rid,))
                            c.commit()
                            c.close()
                            worker = threading.Thread(
                                target=execute,
                                args=(rid, retry_questions, configs, keys, max_tokens,
                                      retries, concurrency, judge_cfg),
                                daemon=True)
                            worker.start()
                            st.session_state["active_run"] = rid
                            st.session_state["worker"] = worker
                            st.success("Retrying %d calls in run %s - watch the Run tab."
                                       % (len(retry_questions) * len(configs), rid[:8]))

    with preview_tab:
        if not questions:
            st.info("No questions loaded.")
        else:
            qid = st.selectbox("Question", [q["id"] for q in questions])
            q = questions_by_id[qid]
            st.json({k: v for k, v in q.items() if k != "options"})
            st.subheader("Exactly what the model receives")
            st.code(build_user_content(q))


if __name__ == "__main__":
    main()
