# analyze_repro.py
import pandas as pd, numpy as np, glob

def load_runs():
    files = glob.glob("runs/*.csv")
    if not files:
        raise SystemExit("No run CSVs found in runs/*.csv")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def metrics(df):
    results = []
    # force consistent dtypes; coerce scores to numeric (strings -> NaN)
    df["bias_label"] = df["bias_label"].astype(str)
    df["bias_score"] = pd.to_numeric(df["bias_score"], errors="coerce")

    for (cond, pipe), g in df.groupby(["condition","pipeline"]):
        errs, stds = [], []
        for aid, sub in g.groupby("article_id"):
            labels = sub["bias_label"]
            # modal label for ERR (tolerates any strings)
            modal = labels.mode().iloc[0]
            err = (labels == modal).mean()

            # numeric-only std for scores
            sc = sub["bias_score"].dropna()
            if len(sc) >= 2:
                sstd = float(sc.std(ddof=0))
            else:
                sstd = 0.0

            errs.append(err)
            stds.append(sstd)

        results.append({
            "condition": cond,
            "pipeline": pipe,
            "mean_ERR": float(np.mean(errs)) if errs else np.nan,
            "median_std": float(np.median(stds)) if stds else np.nan,
            "articles": int(g["article_id"].nunique())
        })
    return pd.DataFrame(results)

if __name__=="__main__":
    df = load_runs()
    out = metrics(df)
    print(out.to_string(index=False))

