#!/usr/bin/env bash
set -euo pipefail

# ---- Models under test (Ollama names) ----
MODELS=(
  "mistral:latest"
  "devstral:latest"
  "llama2:latest"
)

DATA="data/corpus_60_sample.csv"     # or your subset file if you want faster cycles
CONF="config/c2_llm_decoding.json"  # decoding config from step 1
COND="C2"

# ---- Repro/stability env knobs ----
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=5m
# Optional: pin threads for stability (tweak to taste)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p runs reports

for MODEL in "${MODELS[@]}"; do
  SAFE_NAME=$(echo "$MODEL" | tr ':/' '__')
  for i in {1..5}; do
    RUN_ID="${COND}_P1_${SAFE_NAME}_r$(printf "%02d" $i)"
    OUT="runs/${RUN_ID}.csv"
    echo ">>> Running ${RUN_ID} ..."
    python code/run_p1.py \
      --data_csv "$DATA" \
      --out_csv "$OUT" \
      --condition "$COND" \
      --run_id "$RUN_ID" \
      --model_name "$MODEL" \
      --decoding "$(cat "$CONF")"
    # small gap to avoid resource jitter
    sleep 2
  done
done

echo "All runs complete. Outputs in ./runs"

