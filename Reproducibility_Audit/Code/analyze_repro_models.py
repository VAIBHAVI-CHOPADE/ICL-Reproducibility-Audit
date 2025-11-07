# code/analyze_repro_models.py
import pandas as pd, numpy as np, glob, json, re
from collections import Counter, defaultdict

def load_runs(pattern="runs/*.csv"):
    files = glob.glob(pattern)
    if not files:
        raise SystemExit("No run CSVs found in ./runs")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__srcfile"] = f
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def normalize_scores(col):
    # robustly coerce bias_score to float, ignore non-numerics
    return pd.to_numeric(col, errors="coerce")

def per_model_metrics(df):
    out = []
    # model key: prefer the `model_name` column (keeps mistral/devstral/llama2 separate)
    for model, g_model in df.groupby("model_name"):
        # group by article, evaluate stability across multiple runs (run_id)
        article_stats = []
        flips = 0
        total_articles = 0
        stds = []

        for aid, sub in g_model.groupby("article_id"):
            # article-level labels across runs
            labels = sub["bias_label"].astype(str).tolist()
            # modal label for ERR
            mode_label = Counter(labels).most_common(1)[0][0]
            err = (np.array(labels) == mode_label).mean()  # “error” = agreement ratio w/ modal

            # score dispersion
            scores = normalize_scores(sub["bias_score"])
            std_val = np.nan
            if scores.notna().sum() > 1:
                std_val = scores.std()

            # flip detection (>= 2 distinct labels across runs)
            unique_labels = set(labels)
            if len(unique_labels) >= 2:
                flips += 1
            total_articles += 1

            article_stats.append(err)
            stds.append(std_val)

        mean_ERR = float(np.mean(article_stats)) if article_stats else np.nan
        median_std = float(np.nanmedian(stds)) if len(stds) else np.nan
        flip_rate = float(flips / total_articles) if total_articles else np.nan

        out.append({
            "model_name": model,
            "articles": total_articles,
            "mean_ERR": round(mean_ERR, 6),
            "median_std": round(median_std, 6) if not np.isnan(median_std) else np.nan,
            "flip_rate": round(flip_rate, 6)
        })
    return pd.DataFrame(out).sort_values(["flip_rate","median_std","model_name"])

def prompt_hash_sanity(df):
    # For same article & model, prompt_hash should be identical across runs
    bad = []
    for (model, aid), g in df.groupby(["model_name","article_id"]):
        if g["prompt_hash"].nunique() > 1:
            bad.append((model, aid, g["prompt_hash"].unique().tolist()))
    return bad

if __name__=="__main__":
    df = load_runs()
    # Only P1
    df = df[df["pipeline"]=="P1"].copy()

    # Print metrics
    m = per_model_metrics(df)
    print("\n=== Per-model stability metrics (P1, 5 runs each) ===")
    print(m.to_string(index=False))

    # Sanity check for prompt equality
    bad = prompt_hash_sanity(df)
    if bad:
        print("\n⚠️ Inconsistent prompt_hash for some (model, article). Check these:")
        for model, aid, hashes in bad:
            print(f"  model={model} article_id={aid} hashes={hashes}")
    else:
        print("\n✅ prompt_hash is consistent across runs for each (model, article).")

