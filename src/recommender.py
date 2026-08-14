"""
Production inference module for the TF-IDF + Nearest-Neighbors movie recommender.

Loads pre-fitted artifacts (vectorizer, NN model, TF-IDF matrix, metadata)
ONCE and exposes a single recommend() call. Designed to be imported directly
by a Streamlit app: instantiate MovieRecommender once (e.g. behind
st.cache_resource) and call .recommend() on every user interaction.

Usage:
    from recommender import MovieRecommender

    rec = MovieRecommender(artifacts_dir="../../Data")
    results = rec.recommend("The Matrix", top_n=10)
"""

from __future__ import annotations

import difflib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401 (kept for unpickling)
from sklearn.neighbors import NearestNeighbors  # noqa: F401 (kept for unpickling)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    title: str
    similarity_score: float
    poster_path: Optional[str]
    vote_average: Optional[float]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "similarity_score": round(self.similarity_score, 4),
            "poster_path": _clean(self.poster_path),
            "vote_average": _clean(self.vote_average),
        }


def _clean(value):
    """Convert pandas/numpy scalars (incl. NaN) to plain Python types / None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):  # numpy scalar -> native Python scalar
        return value.item()
    return value


class MovieNotFoundError(Exception):
    """Raised when a title has no exact match in the index."""


class MovieRecommender:
    """
    Wraps a fitted TfidfVectorizer + NearestNeighbors index for fast,
    memory-efficient movie-to-movie recommendations.

    All heavy artifacts are loaded once at construction time and reused
    across calls — this matters in a Streamlit app where recommend() may
    run on every rerun/user interaction.
    """

    def __init__(self, artifacts_dir: str = "../../Data", preload: bool = True):
        self.artifacts_dir = Path(artifacts_dir)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.tfidf_matrix: Optional[sparse.csr_matrix] = None
        self.title_index: Optional[dict] = None
        self.meta: Optional[pd.DataFrame] = None

        if preload:
            self._load_artifacts()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_artifacts(self) -> None:
        logger.info("Loading recommender artifacts from %s", self.artifacts_dir)

        with open(self.artifacts_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(self.artifacts_dir / "nn_model.pkl", "rb") as f:
            self.nn_model = pickle.load(f)

        self.tfidf_matrix = sparse.load_npz(self.artifacts_dir / "tfidf_matrix.npz")

        with open(self.artifacts_dir / "title_index.pkl", "rb") as f:
            self.title_index = pickle.load(f)

        self.meta = pd.read_pickle(self.artifacts_dir / "movies_meta.pkl")

        logger.info(
            "Loaded: %d movies, %d TF-IDF features",
            self.tfidf_matrix.shape[0],
            self.tfidf_matrix.shape[1],
        )

    # ------------------------------------------------------------------ #
    # Lookup helpers
    # ------------------------------------------------------------------ #
    def _resolve_title(self, movie_title: str) -> int:
        """Return the row index for an exact (case/whitespace-insensitive) title match."""
        key = movie_title.strip().lower()
        matches = self.title_index.get(key)

        if not matches:
            close = difflib.get_close_matches(key, self.title_index.keys(), n=5, cutoff=0.6)
            suggestions = [self.meta.iloc[self.title_index[c][0]]["title"] for c in close]
            hint = f" Did you mean: {suggestions}?" if suggestions else ""
            raise MovieNotFoundError(f"'{movie_title}' not found.{hint}")

        if len(matches) > 1:
            logger.warning(
                "'%s' matches %d movies; using the first match (row %d). "
                "Disambiguate upstream (e.g. by year or ID) if this matters.",
                movie_title, len(matches), matches[0],
            )

        return matches[0]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recommend(self, movie_title: str, top_n: int = 10) -> list[dict]:
        """
        Return the top_n movies most similar to `movie_title`.

        1. Resolve the title to a row index.
        2. Slice that row's TF-IDF vector directly out of the already-fitted
           matrix (no re-vectorization, no reprocessing text).
        3. Query the fitted NearestNeighbors model for the closest vectors.
        4. Drop the query movie itself, convert cosine distance -> cosine
           similarity, attach display metadata, and return.
        """
        idx = self._resolve_title(movie_title)
        query_vector = self.tfidf_matrix[idx]

        # top_n + 1 because the movie itself will be its own nearest neighbor.
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=top_n + 1)

        results: list[Recommendation] = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            if neighbor_idx == idx:
                continue  # exclude the query movie itself

            row = self.meta.iloc[neighbor_idx]
            similarity = 1.0 - float(dist)  # cosine distance -> cosine similarity

            results.append(
                Recommendation(
                    title=row.get("title"),
                    similarity_score=similarity,
                    poster_path=row.get("poster_path"),
                    vote_average=row.get("vote_average"),
                )
            )
            if len(results) == top_n:
                break

        return [r.to_dict() for r in results]


# Module-level convenience singleton, handy for a quick Streamlit import:
#   from recommender import recommend
_default_recommender: Optional[MovieRecommender] = None


def recommend(movie_title: str, top_n: int = 10, artifacts_dir: str = "../../Data") -> list[dict]:
    global _default_recommender
    if _default_recommender is None:
        _default_recommender = MovieRecommender(artifacts_dir=artifacts_dir)
    return _default_recommender.recommend(movie_title, top_n=top_n)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python recommender.py '<movie title>' [top_n] [artifacts_dir]")
        sys.exit(1)

    title_arg = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    artifacts = sys.argv[3] if len(sys.argv) > 3 else "../../Data"

    rec = MovieRecommender(artifacts_dir=artifacts)
    for r in rec.recommend(title_arg, top_n=n):
        print(r)