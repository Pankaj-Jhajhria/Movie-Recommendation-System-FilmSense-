import sys
import base64
from pathlib import Path

import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
ASSETS_DIR = BASE_DIR / "Assets"
LOGO_PATH = ASSETS_DIR / "Logo.png"
sys.path.append(str(SRC_DIR))

from recommender import (
    load_artifacts,
    search_titles,
    recommend_by_index,
    _fmt_rating,
    _fmt_year,
)

MATCH_COLUMNS = 6
REC_COLUMNS = 5
REC_TOP_N = 10
MAX_MATCH_RESULTS = 24

st.set_page_config(page_title="FilmSense", page_icon=str(LOGO_PATH), layout="wide")


@st.cache_resource
def get_data():
    return load_artifacts()


@st.cache_data
def logo_base64():
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


df, tfidf_matrix, nn_model = get_data()


def poster_url(path):
    if pd.isna(path) or not str(path).strip():
        return "https://via.placeholder.com/300x450?text=No+Poster"
    path = str(path)
    return path if path.startswith("http") else f"https://image.tmdb.org/t/p/w500{path}"


def render_selectable_grid(movies_df, columns_per_row, key_prefix):
    """Plain poster image + a minimal, borderless title/year button beneath it."""
    rows = [movies_df.iloc[i:i + columns_per_row] for i in range(0, len(movies_df), columns_per_row)]
    for row_df in rows:
        cols = st.columns(columns_per_row)
        for col, (idx, movie) in zip(cols, row_df.iterrows()):
            with col:
                st.image(poster_url(movie["poster_path"]), width='stretch')
                label = f"{movie['title']} ({_fmt_year(movie['release_year'])})"
                if st.button(label, key=f"{key_prefix}_{idx}", width='stretch'):
                    st.session_state.selected_idx = idx


def render_static_grid(movies_df, columns_per_row):
    rows = [movies_df.iloc[i:i + columns_per_row] for i in range(0, len(movies_df), columns_per_row)]
    for row_df in rows:
        cols = st.columns(columns_per_row)
        for col, (_, movie) in zip(cols, row_df.iterrows()):
            with col:
                st.image(poster_url(movie["poster_path"]), width='stretch')
                st.caption(f"{movie['title']} ({_fmt_year(movie['release_year'])})")


defaults = {"last_query": "", "selected_idx": None}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---- Styling ----
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    [data-testid="stHeader"] { background: transparent; }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Reset/title button: strip all button chrome, make it look like a heading */
    .reset-title-btn button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        padding: 0 !important;
        text-align: left !important;
    }
    .reset-title-btn button:hover {
        color: #e63946 !important;
    }

    /* Matches Found grid: plain text-link style buttons, no box */
    .plain-grid button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 4px 0 !important;
        font-size: 0.85rem !important;
    }
    .plain-grid button:hover {
        color: #e63946 !important;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header: always pinned at top, centered ----
spacer_l, logo_col, title_col, spacer_r = st.columns([3, 1, 2, 3])
with logo_col:
    st.image(str(LOGO_PATH), width=80)
with title_col:
    st.markdown('<div class="reset-title-btn">', unsafe_allow_html=True)
    if st.button("Film Sense", key="app_title_btn"):
        st.session_state.search_query = ""
        st.session_state.last_query = ""
        st.session_state.selected_idx = None
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# ---- Search bar: centered vertically when empty, sits right below header once content exists ----
show_content = bool(st.session_state.get("search_query", "").strip()) or st.session_state.selected_idx is not None
if not show_content:
    st.markdown("<div style='height:18vh;'></div>", unsafe_allow_html=True)

current_query = st.text_input("Search Movie", key="search_query")
query_stripped = current_query.strip()

if query_stripped != st.session_state.last_query:
    st.session_state.last_query = query_stripped
    st.session_state.selected_idx = None

# ---- Results ----
if st.session_state.selected_idx is not None:
    movie = df.loc[st.session_state.selected_idx]

    st.markdown("---")
    st.subheader("Selected Movie")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(poster_url(movie["poster_path"]), width=200)
    with col2:
        st.markdown(f"### {movie['title']}")
        st.write(f"**Year:** {_fmt_year(movie['release_year'])}")
        st.write(f"**Rating:** {_fmt_rating(movie['vote_average'])}")

    st.markdown("---")
    st.subheader("Recommended Movies")
    results = recommend_by_index(st.session_state.selected_idx, df, tfidf_matrix, nn_model, top_n=REC_TOP_N)
    render_static_grid(results, columns_per_row=REC_COLUMNS)

elif query_stripped:
    st.markdown("---")
    st.subheader("Matches Found")

    matches = search_titles(query_stripped, df, max_results=MAX_MATCH_RESULTS)
    if matches.empty:
        st.warning(f"No movies found matching '{query_stripped}'.")
    else:
        st.markdown('<div class="plain-grid">', unsafe_allow_html=True)
        render_selectable_grid(matches, columns_per_row=MATCH_COLUMNS, key_prefix="match")
        st.markdown('</div>', unsafe_allow_html=True)