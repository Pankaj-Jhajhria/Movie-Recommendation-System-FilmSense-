import argparse
import pickle
import difflib
import textwrap
from pathlib import Path
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "Data" / "New"


def load_artifacts():
    df = pd.read_csv(DATA_DIR / "Dataset_Final.csv")
    tfidf_matrix = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    with open(DATA_DIR / "nearest_neighbors.pkl", "rb") as f:
        nn_model = pickle.load(f)
    return df, tfidf_matrix, nn_model


def recommend(movie_title, df, tfidf_matrix, nn_model, top_n=10):
    title_to_index = pd.Series(df.index, index=df['title'])
    title_to_index = title_to_index[~title_to_index.index.duplicated(keep='first')]

    if movie_title not in title_to_index:
        close = difflib.get_close_matches(movie_title, title_to_index.index, n=5, cutoff=0.6)
        msg = f"'{movie_title}' not found in dataset."
        if close:
            msg += f" Did you mean: {close}?"
        print(msg)
        return None

    idx = title_to_index[movie_title]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx])
    distances, indices = distances.flatten(), indices.flatten()

    results = []
    for dist, i in zip(distances, indices):
        if i == idx:
            continue
        results.append({
            'title': df.iloc[i]['title'],
            'similarity_score': round(1 - dist, 4),
            'vote_average': df.iloc[i]['vote_average'],
            'release_date': df.iloc[i]['release_date'],
            'poster_path': df.iloc[i]['poster_path']
        })

    return (pd.DataFrame(results)
            .sort_values('similarity_score', ascending=False)
            .head(top_n)
            .reset_index(drop=True))


def _fmt_rating(val):
    try:
        val = float(val)
        return f"{val:.1f}/10"
    except (ValueError, TypeError):
        return "N/A"


def _fmt_year(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    return str(val).strip("() ")  # strips stray parentheses like "(2018)"


def _fmt_poster(val, width=70):
    if pd.isna(val) or str(val).strip() == "":
        return "N/A"
    val = str(val)
    return val if len(val) <= width else val[:width - 3] + "..."


def print_recommendations(movie_title, results):
    if results is None or results.empty:
        return

    header = f" Top {len(results)} recommendations for '{movie_title}' "
    print("\n" + header.center(64, "="))

    for rank, row in results.iterrows():
        print(f"\n{rank + 1}. {row['title']}")
        print(f"   {'Similarity':<10}: {row['similarity_score']:.4f}")
        print(f"   {'Rating':<10}: {_fmt_rating(row['vote_average'])}")
        print(f"   {'Released':<10}: {_fmt_year(row['release_date'])}")
        print(f"   {'Poster':<10}: {_fmt_poster(row['poster_path'])}")

    print("\n" + "=" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Movie recommendation CLI")
    parser.add_argument("title", type=str, help="Exact movie title to get recommendations for")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations (default 10)")
    args = parser.parse_args()

    df, tfidf_matrix, nn_model = load_artifacts()
    results = recommend(args.title, df, tfidf_matrix, nn_model, top_n=args.top_n)
    print_recommendations(args.title, results)


if __name__ == "__main__":
    main()