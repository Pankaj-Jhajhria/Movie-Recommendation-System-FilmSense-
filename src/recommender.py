import argparse
import pickle
import difflib
import textwrap
from pathlib import Path
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "Data" / "New"

import re

def normalize_title(text):
    """Lowercase and strip everything except letters/digits, so 'Spider-Man'
    and 'spiderman' become identical, punctuation/spacing stops mattering."""
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def search_titles(query, df, max_results=10, fuzzy_cutoff=0.6):
    """
    Case-insensitive, partial, and fuzzy title search.
    Returns candidate rows ranked: exact match > partial match > fuzzy match.
    """
    query_norm = normalize_title(query)
    if not query_norm:
        return pd.DataFrame()

    norm_titles = df['title'].apply(normalize_title)

    # 1. Exact match (after normalization) — handles "spiderman" -> "Spider-Man"
    exact_mask = norm_titles == query_norm
    if exact_mask.any():
        return df[exact_mask].head(max_results)

    # 2. Partial / substring match — handles "inter" -> "Interstellar",
    #    "dark knight" -> "The Dark Knight"
    partial_mask = norm_titles.str.contains(re.escape(query_norm), regex=True)
    if partial_mask.any():
        matches = df[partial_mask]
        # rank tighter matches (title length closer to query length) first
        length_diff = (norm_titles[partial_mask].str.len() - len(query_norm)).abs()
        return matches.loc[length_diff.sort_values().index].head(max_results)

    # 3. Fuzzy match — catches typos, e.g. "intersteller", "spidrman"
    unique_norms = norm_titles.unique().tolist()
    close = difflib.get_close_matches(query_norm, unique_norms, n=max_results, cutoff=fuzzy_cutoff)
    if close:
        fuzzy_mask = norm_titles.isin(close)
        return df[fuzzy_mask].head(max_results)

    return pd.DataFrame()


def select_movie(query, df):
    """
    Runs search_titles and, if multiple candidates are found, lets the
    user pick interactively. Returns the chosen row's df index, or None.
    """
    matches = search_titles(query, df)

    if matches.empty:
        print(f"No movies found matching '{query}'.")
        return None

    if len(matches) == 1:
        row = matches.iloc[0]
        print(f"Found: {row['title']} ({_fmt_year(row['release_date'])})")
        return row.name  # original df index

    print(f"\nMultiple matches found for '{query}':\n")
    for i, (_, row) in enumerate(matches.iterrows(), start=1):
        print(f"  {i}. {row['title']}  ({_fmt_year(row['release_date'])}, {_fmt_rating(row['vote_average'])})")

    choice = input(f"\nSelect a movie [1-{len(matches)}] (Enter to cancel): ").strip()
    if not choice:
        print("Cancelled.")
        return None

    if choice.isdigit() and 1 <= int(choice) <= len(matches):
        return matches.iloc[int(choice) - 1].name

    print("Invalid selection.")
    return None

def load_artifacts():
    df = pd.read_csv(DATA_DIR / "Dataset_Final.csv")
    tfidf_matrix = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    with open(DATA_DIR / "nearest_neighbors.pkl", "rb") as f:
        nn_model = pickle.load(f)
    return df, tfidf_matrix, nn_model


def recommend_by_index(idx, df, tfidf_matrix, nn_model, top_n=10):
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