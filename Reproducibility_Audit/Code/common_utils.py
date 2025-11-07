# common_utils.py
import os, csv, json, hashlib, platform, subprocess, statistics as stats

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))

def prompt_hash(s: str) -> str:
    return sha256_bytes(s.replace("\r\n", "\n").encode("utf-8"))

def env_fingerprint() -> str:
    info = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}"
    }
    try:
        pip = subprocess.run(["pip", "freeze"], capture_output=True, text=True, timeout=5)
        info["pip_freeze"] = pip.stdout.splitlines()[:50]
    except Exception:
        info["pip_freeze"] = []
    return json.dumps(info, sort_keys=True)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_sentences_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r["sent_id"] = int(r["sent_id"])
            rows.append(r)
    return rows

def group_by_article(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["article_id"], []).append(r)
    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda x: x["sent_id"])
    return grouped

def write_csv_header(path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            "run_id","timestamp","condition","pipeline","article_id",
            "bias_label","bias_score","evidence_sent_ids_json",
            "rationale_hash","prompt_hash","model_name","model_version",
            "decoding_params_json","agentic_steps_json_hash",
            "tool_snapshot_id","code_git_sha","env_fingerprint"
        ])

def append_row(path, row):
    with open(path, "a", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            row.get("run_id"), row.get("timestamp"), row.get("condition"), row.get("pipeline"), row.get("article_id"),
            row.get("bias_label"), row.get("bias_score"),
            json.dumps(row.get("evidence_sent_ids", []), ensure_ascii=False),
            row.get("rationale_hash"), row.get("prompt_hash"),
            row.get("model_name"), row.get("model_version"),
            json.dumps(row.get("decoding_params_json"), sort_keys=True),
            row.get("agentic_steps_json_hash"), row.get("tool_snapshot_id"),
            row.get("code_git_sha"), row.get("env_fingerprint")
        ])

