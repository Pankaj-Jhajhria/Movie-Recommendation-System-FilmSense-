import argparse
import pickle
import difflib
import re
from pathlib import Path
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MODEL_DIR = SCRIPT_DIR.parent / "models"

# ---- Dataset 2 artifacts ----
DATASET_FILE = "Final_Dataset_2.csv"
TFIDF_MATRIX_FILE = "tfidf_matrix2.npz"
NN_MODEL_FILE = "nearest_neighbors2.pkl"


def normalize_title(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def search_titles(query, df, max_results=10, fuzzy_cutoff=0.6):
    query_norm = normalize_title(query)
    if not query_norm:
        return pd.DataFrame()

    norm_titles = df['title'].apply(normalize_title)

    exact_mask = norm_titles == query_norm
    partial_mask = norm_titles.str.contains(re.escape(query_norm), regex=True) & ~exact_mask

    if exact_mask.any() or partial_mask.any():
        exact_df = df[exact_mask]
        partial_df = df[partial_mask]
        length_diff = (norm_titles[partial_mask].str.len() - len(query_norm)).abs()
        partial_df = partial_df.loc[length_diff.sort_values().index]
        return pd.concat([exact_df, partial_df]).head(max_results)

    unique_norms = norm_titles.unique().tolist()
    close = difflib.get_close_matches(query_norm, unique_norms, n=max_results, cutoff=fuzzy_cutoff)
    if close:
        return df[norm_titles.isin(close)].head(max_results)

    return pd.DataFrame()


def select_movie(query, df):
    matches = search_titles(query, df)

    if matches.empty:
        print(f"No movies found matching '{query}'.")
        return None

    if len(matches) == 1:
        row = matches.iloc[0]
        print(f"Found: {row['title']} ({_fmt_year(row['release_year'])})")
        return row.name

    print(f"\nMultiple matches found for '{query}':\n")
    for i, (_, row) in enumerate(matches.iterrows(), start=1):
        print(f"  {i}. {row['title']}  ({_fmt_year(row['release_year'])}, {_fmt_rating(row['vote_average'])})")

    choice = input(f"\nSelect a movie [1-{len(matches)}] (Enter to cancel): ").strip()
    if not choice:
        print("Cancelled.")
        return None

    if choice.isdigit() and 1 <= int(choice) <= len(matches):
        return matches.iloc[int(choice) - 1].name

    print("Invalid selection.")
    return None


def load_artifacts():
    df = pd.read_csv(DATA_DIR / DATASET_FILE)
    tfidf_matrix = sparse.load_npz(MODEL_DIR / TFIDF_MATRIX_FILE)

    nn_path = MODEL_DIR / NN_MODEL_FILE
    if not nn_path.exists():
        raise FileNotFoundError(
            f"'{NN_MODEL_FILE}' not found in {MODEL_DIR}. Run the training cell "
            "that fits NearestNeighbors on tfidf_matrix2 and saves it first."
        )
    with open(nn_path, "rb") as f:
        nn_model = pickle.load(f)

    assert df.shape[0] == tfidf_matrix.shape[0], (
        f"Row mismatch: {DATASET_FILE} has {df.shape[0]} rows, "
        f"{TFIDF_MATRIX_FILE} has {tfidf_matrix.shape[0]} rows."
    )
    return df, tfidf_matrix, nn_model


def recommend_by_index(idx, df, tfidf_matrix, nn_model, top_n=10):
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx])
    distances, indices = distances.flatten(), indices.flatten()

    rows = []
    result_index = []
    for dist, i in zip(distances, indices):
        if i == idx:
            continue
        result_index.append(df.index[i])   # preserve the movie's real df index
        rows.append({
            'title': df.iloc[i]['title'],
            'similarity_score': round(1 - dist, 4),
            'vote_average': df.iloc[i]['vote_average'],
            'imdb_rating': df.iloc[i]['IMDB_Rating'],
            'release_year': df.iloc[i]['release_year'],
            'poster_path': df.iloc[i]['poster_path']
        })

    result_df = pd.DataFrame(rows, index=result_index)   # index carried over, not reset
    return result_df.sort_values('similarity_score', ascending=False).head(top_n)


def _fmt_rating(val):
    try:
        val = float(val)
        return f"{val:.1f}/10"
    except (ValueError, TypeError):
        return "N/A"


def _fmt_year(val):
    if pd.isna(val):
        return "Unknown"
    try:
        return str(int(float(val)))  # handles 2021.0 -> "2021"
    except (ValueError, TypeError):
        return str(val).strip("() ")


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
        print(f"   {'Similarity':<12}: {row['similarity_score']:.4f}")
        print(f"   {'TMDB Rating':<12}: {_fmt_rating(row['vote_average'])}")
        print(f"   {'IMDB Rating':<12}: {_fmt_rating(row['imdb_rating'])}")
        print(f"   {'Released':<12}: {_fmt_year(row['release_year'])}")
        print(f"   {'Poster':<12}: {_fmt_poster(row['poster_path'])}")

    print("\n" + "=" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Movie recommendation CLI")
    parser.add_argument("title", type=str, help="Movie title to search (partial/fuzzy matching supported)")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations (default 10)")
    args = parser.parse_args()

    df, tfidf_matrix, nn_model = load_artifacts()

    idx = select_movie(args.title, df)
    if idx is None:
        return

    results = recommend_by_index(idx, df, tfidf_matrix, nn_model, top_n=args.top_n)
    print_recommendations(df.loc[idx, 'title'], results)


if __name__ == "__main__":
    main()