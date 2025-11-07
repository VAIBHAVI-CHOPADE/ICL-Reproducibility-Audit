# run_p1.py
import json, time, datetime, argparse
from model_clients import LLMClient
from common_utils import *

SYSTEM_PROMPT = """
You are an impartial media-bias analyst.

Examples:
1. "The brave government defeated the corrupt opposition." → "Right"
2. "The administration’s actions hurt the poor and helpless." → "Left"
3. "The minister announced a new scheme." → "Neutral"

Follow the same pattern for the next article.
Output JSON schema: {...}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--model_name", default="mistral:latest")

    ap.add_argument("--decoding", default='{"temperature":0.0,"top_p":1.0,"max_tokens":512}')
    args = ap.parse_args()

    decoding = json.loads(args.decoding)
    client = LLMClient(args.model_name, decoding)
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    if not os.path.exists(args.out_csv): write_csv_header(args.out_csv)

    rows = read_sentences_csv(args.data_csv)
    grouped = group_by_article(rows)
    env_fp = env_fingerprint()

    for article_id, sents in grouped.items():
        user_prompt = json.dumps({
            "article_id": article_id,
            "sentences": [{"sent_id": s["sent_id"], "text": s["sentence"]} for s in sents]
        }, ensure_ascii=False)
        phash = prompt_hash(SYSTEM_PROMPT + user_prompt)
        raw = client.call(SYSTEM_PROMPT, user_prompt)
        try:
            obj = json.loads(raw)
        except:
            obj = {"bias_label":"Neutral","bias_score":0,"evidence_sent_ids":[],"rationale":"parse error"}

        append_row(args.out_csv, {
            "run_id": args.run_id,
            "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
            "condition": args.condition,
            "pipeline": "P1",
            "article_id": article_id,
            "bias_label": obj.get("bias_label","Neutral"),
            "bias_score": obj.get("bias_score",0),
            "evidence_sent_ids": obj.get("evidence_sent_ids",[]),
            "rationale_hash": sha256_bytes(obj.get("rationale","").encode()),
            "prompt_hash": phash,
            "model_name": args.model_name,
            "model_version": "local",
            "decoding_params_json": decoding,
            "agentic_steps_json_hash": "",
            "tool_snapshot_id": "",
            "code_git_sha": "",
            "env_fingerprint": env_fp
        })

if __name__ == "__main__":
    main()

