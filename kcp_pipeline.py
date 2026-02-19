import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

# Free local semantic embeddings
from sentence_transformers import SentenceTransformer


# -------------------------
# Helpers
# -------------------------

def normalize_keyword(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_google_keyword_export(path: str, skiprows: int) -> pd.DataFrame:
    # Google Keyword Planner exports: UTF-16, tab-separated, header rows at top
    return pd.read_csv(path, encoding="utf-16", sep="\t", skiprows=skiprows)


def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for item in inputs:
        if any(ch in item for ch in ["*", "?", "["]):
            paths.extend(glob.glob(item))
        else:
            paths.append(item)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def safe_minmax_norm(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    xmin = np.nanmin(x.values)
    xmax = np.nanmax(x.values)
    if np.isclose(xmax - xmin, 0) or np.isnan(xmin) or np.isnan(xmax):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - xmin) / (xmax - xmin)


def safe_log_norm(x: pd.Series) -> pd.Series:
    lx = np.log1p(x.astype(float))
    return safe_minmax_norm(lx)


# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    keep_cols: List[str]
    skiprows: int
    min_volume: int
    intents: List[Tuple[str, str]]  # (intent_name, regex)
    bands: Dict[str, List[int]]     # intent_name -> [low, high]
    bonus_weights: Dict[str, float] # cheapness, volume, competition
    seed: int
    k_topics: str                   # "auto" or number
    embedding_model: str            # sentence-transformers model name


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        keep_cols=raw["keep_cols"],
        skiprows=int(raw.get("skiprows", 2)),
        min_volume=int(raw.get("min_volume", 100)),
        intents=[(i["name"], i["regex"]) for i in raw["intents"]],
        bands=raw["bands"],
        bonus_weights=raw["bonus_weights"],
        seed=int(raw.get("seed", 42)),
        k_topics=str(raw.get("k_topics", "auto")),
        embedding_model=str(raw.get("embedding_model", "all-MiniLM-L6-v2")),
    )


# -------------------------
# Intent
# -------------------------

def assign_intent(keyword: str, intents: List[Tuple[str, str]]) -> str:
    if pd.isna(keyword):
        return "Other"
    s = str(keyword).lower()
    for name, pattern in intents:
        if re.search(pattern, s):
            return name
    return "Other"


# -------------------------
# Semantic topic clustering + labeling
# -------------------------

def pick_k(n_rows: int, k_topics: str) -> int:
    if k_topics != "auto":
        try:
            k = int(k_topics)
            return max(5, k)
        except ValueError:
            pass
    return max(20, min(120, int(n_rows ** 0.5)))


def embed_keywords(keywords: List[str], model_name: str) -> np.ndarray:
    """
    Returns an (n, d) numpy array of semantic embeddings.
    First run will download the model (one-time).
    """
    model = SentenceTransformer(model_name)
    emb = model.encode(
        keywords,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def cluster_topics_semantic(keywords: List[str], seed: int, k_topics: str, model_name: str) -> List[str]:
    """
    Semantic clustering using embeddings + KMeans.
    Topic label = representative keyword closest to cluster centroid.
    """
    n = len(keywords)
    k = pick_k(n, k_topics)

    X = embed_keywords(keywords, model_name=model_name)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    cluster_ids = km.fit_predict(X)

    centroids = km.cluster_centers_
    labels = {}
    for cid in range(k):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            labels[cid] = f"topic_{cid}"
            continue
        cluster_vectors = X[idx]
        center = centroids[cid]
        dists = np.linalg.norm(cluster_vectors - center, axis=1)
        best = idx[np.argmin(dists)]
        labels[cid] = keywords[best]

    topics = [labels[c] for c in cluster_ids]
    return topics


# -------------------------
# Scoring (3 bands)
# -------------------------

def score_rows(df: pd.DataFrame, bands: Dict[str, List[int]], bonus_weights: Dict[str, float]) -> pd.Series:
    # cheapness: lower bid => higher bonus
    low = pd.to_numeric(df["Top of page bid (low range)"], errors="coerce")
    high = pd.to_numeric(df["Top of page bid (high range)"], errors="coerce")
    mid = (low + high) / 2
    mid = mid.fillna(mid.median())
    bid_norm = safe_minmax_norm(mid)
    cheapness = 1 - bid_norm  # 0..1

    # volume: normalized log scale
    vol = pd.to_numeric(df["Avg. monthly searches"], errors="coerce").fillna(0)
    volume = safe_log_norm(vol)  # 0..1

    # competition: lower is better (indexed value)
    comp = pd.to_numeric(df["Competition (indexed value)"], errors="coerce")
    comp = comp.fillna(comp.median())
    comp_norm = safe_minmax_norm(comp)
    competition = 1 - comp_norm  # 0..1

    w_cheap = float(bonus_weights.get("cheapness", 0.45))
    w_vol = float(bonus_weights.get("volume", 0.35))
    w_comp = float(bonus_weights.get("competition", 0.20))
    w_sum = w_cheap + w_vol + w_comp
    if w_sum <= 0:
        w_cheap, w_vol, w_comp = 0.45, 0.35, 0.20
        w_sum = 1.0

    bonus = (w_cheap * cheapness + w_vol * volume + w_comp * competition) / w_sum  # 0..1

    scores = []
    for i, intent in enumerate(df["intent"].astype(str).tolist()):
        lo, hi = bands.get(intent, [10, 40])
        lo, hi = int(lo), int(hi)
        s = lo + bonus.iloc[i] * (hi - lo)
        scores.append(int(round(s)))
    return pd.Series(scores, index=df.index)


# -------------------------
# Main pipeline
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input files or glob patterns")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    in_files = expand_inputs(args.inputs)
    if not in_files:
        raise SystemExit("No input files matched. Check --in path/pattern.")

    # Merge
    parts = []
    for f in in_files:
        d = read_google_keyword_export(f, skiprows=cfg.skiprows)
        cols = [c for c in cfg.keep_cols if c in d.columns]
        d = d[cols].copy()
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)

    # Volume filter (drop invalid/missing volume)
    df["Avg. monthly searches"] = pd.to_numeric(df["Avg. monthly searches"], errors="coerce")
    df = df.dropna(subset=["Avg. monthly searches"])
    df = df[df["Avg. monthly searches"] >= cfg.min_volume].copy()

    # Normalized dedupe (keep highest volume)
    df["keyword_norm"] = df["Keyword"].apply(normalize_keyword)
    df = (
        df.sort_values("Avg. monthly searches", ascending=False)
          .drop_duplicates("keyword_norm")
          .drop(columns=["keyword_norm"])
          .reset_index(drop=True)
    )

    # Intent (priority order = negative guards)
    df["intent"] = df["Keyword"].apply(lambda x: assign_intent(x, cfg.intents))

    # Semantic topic clustering
    keywords = df["Keyword"].fillna("").astype(str).tolist()
    df["topic"] = cluster_topics_semantic(
        keywords,
        seed=cfg.seed,
        k_topics=cfg.k_topics,
        model_name=cfg.embedding_model,
    )

    # Cluster combo column
    df["cluster"] = df["intent"].astype(str) + " | " + df["topic"].astype(str)

    # Score (3-band)
    df["score"] = score_rows(df, bands=cfg.bands, bonus_weights=cfg.bonus_weights)

    # Export
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
