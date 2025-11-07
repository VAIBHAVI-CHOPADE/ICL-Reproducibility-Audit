#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports

for csv in runs/*.csv; do
  base=$(basename "$csv" .csv)
  awk -F, '
    NR==1 {
      for (i=1;i<=NF;i++) { if ($i=="article_id") Ai=i; if ($i=="prompt_hash") Ph=i; if ($i=="model_name") Mi=i }
      next
    }
    { k=$Mi "|" $Ai; h=$Ph; seen[k "|" h]++ ; pairs[k]++ }
    END{
      for (k in pairs) {
        # count unique prompt hashes per (model|article)
        cnt=0
        for (kh in seen) { split(kh, a, /\|/); if (a[1] "|" a[2] == k) cnt++ }
        print k, cnt
      }
    }
  ' "$csv" | sort > "reports/${base}_prompt_hash_counts.txt"
done

echo "Wrote per-run prompt-hash counts to ./reports/*_prompt_hash_counts.txt"

