"""
Build the NearestNeighbors index + lookup files on top of an ALREADY-FITTED
TF-IDF vectorizer/matrix (e.g. from your TF-IDF.ipynb pipeline).

Use this instead of build_index.py when tfidf_vectorizer.pkl and
tfidf_matrix.npz already exist and you just need:
    - nn_model.pkl      (fitted NearestNeighbors index)
    - title_index.pkl   (title -> row-index lookup)
    - movies_meta.pkl   (title / poster_path / vote_average, for display)

This avoids re-running TfidfVectorizer.fit_transform() on 1.47M rows again.

Run (from the Notebooks/ or src/ folder, matching your existing paths):
    python build_nn_index.py \
        --data ../../Data/Final_Dataset_with_Equivalent.csv \
        --data-dir ../../Data
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_nn_index(data_path: str, data_dir: str = "../../Data") -> None:
    data_dir = Path(data_dir)

    tfidf_matrix_path = data_dir / "tfidf_matrix.npz"
    if not tfidf_matrix_path.exists():
        raise FileNotFoundError(
            f"{tfidf_matrix_path} not found. Run your TF-IDF.ipynb pipeline first "
            "(or use build_index.py to build everything from scratch)."
        )

    logger.info("Loading existing TF-IDF matrix from %s", tfidf_matrix_path)
    tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

    logger.info("Loading dataset from %s (for titles/metadata only)", data_path)
    df = pd.read_csv(data_path)
    if "title" not in df.columns:
        raise ValueError("Dataset must contain a 'title' column.")
    if len(df) != tfidf_matrix.shape[0]:
        raise ValueError(
            f"Row-count mismatch: CSV has {len(df)} rows but tfidf_matrix.npz has "
            f"{tfidf_matrix.shape[0]}. They must come from the exact same dataframe "
            "(same order) used to fit the vectorizer, or row indices won't line up."
        )

    # ---------------------------------------------------------------- #
    # Fit NearestNeighbors on the existing sparse matrix
    # ---------------------------------------------------------------- #
    logger.info("Fitting NearestNeighbors (metric=cosine, algorithm=brute)")
    t0 = time.time()
    nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    nn_model.fit(tfidf_matrix)
    logger.info("NearestNeighbors fit in %.2fs", time.time() - t0)

    # ---------------------------------------------------------------- #
    # Title lookup table
    # ---------------------------------------------------------------- #
    logger.info("Building title lookup index")
    title_index: dict[str, list[int]] = {}
    for idx, title in enumerate(df["title"].astype(str)):
        key = title.strip().lower()
        title_index.setdefault(key, []).append(idx)

    dupes = sum(1 for v in title_index.values() if len(v) > 1)
    if dupes:
        logger.warning("%d titles have duplicate entries; recommend() will use the first match.", dupes)

    # ---------------------------------------------------------------- #
    # Save
    # ---------------------------------------------------------------- #
    with open(data_dir / "nn_model.pkl", "wb") as f:
        pickle.dump(nn_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_dir / "title_index.pkl", "wb") as f:
        pickle.dump(title_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_cols = [c for c in ["title", "poster_path", "vote_average"] if c in df.columns]
    df[meta_cols].to_pickle(data_dir / "movies_meta.pkl")

    logger.info("Done. Wrote nn_model.pkl, title_index.pkl, movies_meta.pkl to %s", data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the NN index on top of an existing TF-IDF matrix.")
    parser.add_argument("--data", required=True, help="Path to the CSV used to fit the TF-IDF vectorizer")
    parser.add_argument("--data-dir", default="../../Data", help="Folder containing tfidf_matrix.npz")
    args = parser.parse_args()

    build_nn_index(args.data, data_dir=args.data_dir)