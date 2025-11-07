#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
#   C3: Prompt Perturbation Variants
# ------------------------------

MODELS=(
  "llama2:latest"
)

DATA="data/corpus_60_sample.csv"
CONF="config/c3_llm_decoding.json"
PROMPT_DIR="config/prompts"
COND="C3"

export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=5m
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p runs reports

# Pre-compress decoding JSON into one line
DECODING_JSON=$(tr -d '\n' < "$CONF" | tr -d '\r')

for MODEL in "${MODELS[@]}"; do
  SAFE_NAME=$(echo "$MODEL" | tr ':/' '__')
  for V in {1..5}; do
    PROMPT_FILE="${PROMPT_DIR}/c3_v${V}.json"
    PROMPT=$(jq -r '.system_prompt' "$PROMPT_FILE")

    for i in {1..5}; do
      RUN_ID="${COND}_V${V}_${SAFE_NAME}_r$(printf "%02d" $i)"
      OUT="runs/${RUN_ID}.csv"
      echo ">>> Running ${RUN_ID} ..."

            # Inline Python runner with explicit sys.argv injection
      python - <<EOF
import sys, os
sys.path.append(os.path.join(os.getcwd(), "code"))
import run_p1
sys.argv = [
    "run_p1.py",
    "--data_csv", "$DATA",
    "--out_csv", "$OUT",
    "--condition", "${COND}_V${V}",
    "--run_id", "$RUN_ID",
    "--model_name", "$MODEL",
    "--decoding", '$DECODING_JSON'
]
run_p1.SYSTEM_PROMPT = """$PROMPT"""
run_p1.main()
EOF


      sleep 2
    done
  done
done

echo "✅ All C3 variant runs complete. Outputs in ./runs"

