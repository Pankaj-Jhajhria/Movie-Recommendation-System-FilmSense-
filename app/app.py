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

REC_TOP_N = 10
GRID_COLUMNS = 5
TRENDING_COUNT = 10
EXPLORE_COUNT = 15
TOP_RATED_COUNT = 15

st.set_page_config(page_title="FilmSense", page_icon=str(LOGO_PATH), layout="wide")


@st.cache_resource
def get_data():
    return load_artifacts()


df, tfidf_matrix, nn_model = get_data()


def poster_url(path):
    if pd.isna(path) or not str(path).strip():
        return "https://via.placeholder.com/300x450?text=No+Poster"
    path = str(path)
    return path if path.startswith("http") else f"https://image.tmdb.org/t/p/w500{path}"


def backdrop_url(path):
    if pd.isna(path) or not str(path).strip():
        return None
    path = str(path)
    return path if path.startswith("http") else f"https://image.tmdb.org/t/p/original{path}"


def fmt_runtime(minutes):
    try:
        minutes = int(float(minutes))
        h, m = divmod(minutes, 60)
        return f"{h}h {m}m" if h else f"{m}m"
    except (ValueError, TypeError):
        return "N/A"


def render_selectable_grid(movies_df, columns_per_row, key_prefix, jump_home=False):
    """Plain poster + a minimal clickable title/year link beneath it."""
    rows = [movies_df.iloc[i:i + columns_per_row] for i in range(0, len(movies_df), columns_per_row)]
    for row_df in rows:
        cols = st.columns(columns_per_row)
        for col, (idx, movie) in zip(cols, row_df.iterrows()):
            with col:
                st.markdown('<div class="poster-card">', unsafe_allow_html=True)
                st.image(poster_url(movie["poster_path"]), width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                label = f"{movie['title']} ({_fmt_year(movie['release_year'])})"
                st.markdown('<div class="poster-btn">', unsafe_allow_html=True)
                clicked = st.button(label, key=f"{key_prefix}_{idx}", width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                if clicked:
                    st.session_state.selected_idx = idx
                    if jump_home:
                        st.session_state.current_page = "Home"
                    st.rerun()



defaults = {
    "last_query": "",
    "selected_idx": None,
    "current_page": "Home",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------------- global styling ----------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --sage: #7e9f9b;
        --sage-light: #a9c1bd;
        --sage-pale: #eef3f2;
        --sage-dark: #5f7d79;
        --ink: #16201e;
        --white: #ffffff;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    * { transition: background-color 0.25s ease, color 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease; }

    html, body { background-color: var(--sage); }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background: transparent !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    [data-testid="stSidebar"] {
        background: var(--sage-pale);
        border-right: 1px solid rgba(22, 32, 30, 0.1);
    }
    .sidebar-logo { display: flex; align-items: center; gap: 10px; }
    .sidebar-logo img { width: 42px; height: 42px; border-radius: 10px; }
    .sidebar-logo .brand { font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.3rem; color: var(--ink); }
    .sidebar-logo .brand span { color: var(--sage-dark); }
    .sidebar-tagline { color: #5c6b68; font-size: 0.72rem; margin: 2px 0 16px 52px; }

    .nav-active {
        background: var(--sage);
        color: var(--white) !important;
        font-weight: 600;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--ink) !important;
        text-align: left !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(126, 159, 155, 0.18) !important;
        color: var(--ink) !important;
    }

    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .hero-title .accent { color: var(--sage-dark); }
    .hero-sub { color: rgba(22,32,30,0.75); font-size: 1.05rem; margin-bottom: 22px; }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--ink);
        border-left: 4px solid var(--sage-dark);
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    .grid-btn button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--ink) !important;
        padding: 4px 0 !important;
        font-size: 0.82rem !important;
        text-align: left !important;
    }
    .grid-btn button:hover { color: var(--sage-dark) !important; }

    .poster-card { padding: 4px; }
    .poster-card [data-testid="stImage"] img {
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.18);
    }
    .poster-card [data-testid="stImage"] img:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 24px rgba(0,0,0,0.28);
    }

    .poster-btn button {
        background: var(--white) !important;
        border: 1px solid rgba(22,32,30,0.12) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        color: var(--ink) !important;
        padding: 10px 12px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-align: left !important;
    }
    .poster-btn button:hover {
        border-color: var(--sage-dark) !important;
        color: var(--sage-dark) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }

    .rating-pill {
        display: inline-block;
        background: var(--sage-pale);
        color: var(--sage-dark);
        border: 1px solid rgba(95,125,121,0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0 8px 8px 0;
        font-weight: 600;
    }

    div[data-testid="stTextInput"] input {
        background: var(--white) !important;
        border: 1px solid rgba(22,32,30,0.15) !important;
        color: var(--ink) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--sage-dark) !important;
        box-shadow: 0 0 0 3px rgba(126,159,155,0.25) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


NAV_ITEMS = [
    ("Home", "🏠"),
    ("Explore", "🧭"),
    ("Top Rated", "⭐"),
    ("Genres", "🎭"),
    ("Watchlist", "🔖"),
    ("About", "ℹ️"),
]

with st.sidebar:
    logo_b64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
    st.markdown(
        f"""
        <div class="sidebar-logo">
            <img src="data:image/png;base64,{logo_b64}">
            <div class="brand">Film<span>Sense</span></div>
        </div>
        <div class="sidebar-tagline">Find Films. Feel More.</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="grid-btn">', unsafe_allow_html=True)
    if st.button("⟲  New Search", key="reset_btn", width='stretch'):
        st.session_state.search_query = ""
        st.session_state.last_query = ""
        st.session_state.selected_idx = None
        st.session_state.current_page = "Home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    for label, icon in NAV_ITEMS:
        if st.session_state.current_page == label:
            st.markdown(f"<div class='nav-active'>{icon}&nbsp;&nbsp;{label}</div>", unsafe_allow_html=True)
        else:
            if st.button(f"{icon}   {label}", key=f"nav_{label}", width='stretch'):
                st.session_state.current_page = label
                st.rerun()


def render_matches(query_stripped):
    st.markdown('<div class="section-title">Matches Found</div>', unsafe_allow_html=True)
    matches = search_titles(query_stripped, df, max_results=24)
    if matches.empty:
        st.warning(f"No movies found matching '{query_stripped}'.")
    else:
        render_selectable_grid(matches, columns_per_row=6, key_prefix="match")


def render_selected_and_recs():
    movie = df.loc[st.session_state.selected_idx]
    bdrop = backdrop_url(movie.get("backdrop_path"))

    st.markdown('<div class="section-title">Selected Movie</div>', unsafe_allow_html=True)
    if bdrop:
        st.image(bdrop, width='stretch')

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(poster_url(movie["poster_path"]), width='stretch')
    with col2:
        st.markdown(
            f"<div style='font-family:Poppins,sans-serif; font-size:2rem; font-weight:800; color:#16201e;'>{movie['title']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <span class="rating-pill">⭐ TMDB {_fmt_rating(movie['vote_average'])}</span>
            <span class="rating-pill">IMDB {_fmt_rating(movie['IMDB_Rating'])}</span>
            <span class="rating-pill">{_fmt_year(movie['release_year'])}</span>
            <span class="rating-pill">{fmt_runtime(movie['runtime'])}</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if st.button("🔖  Add to Watchlist", key="watchlist_btn"):
            st.toast("Watchlist is coming soon!")

    st.markdown('<div class="section-title">Recommended For You</div>', unsafe_allow_html=True)
    results = recommend_by_index(st.session_state.selected_idx, df, tfidf_matrix, nn_model, top_n=REC_TOP_N)
    render_selectable_grid(results, columns_per_row=GRID_COLUMNS, key_prefix="rec")


def render_home():
    show_content = bool(st.session_state.get("search_query", "").strip()) or st.session_state.selected_idx is not None

    if not show_content:
        st.markdown(
            """
            <div class="hero-title">Find Your <span class="accent">Next Favorite</span><br>Movie.</div>
            <div class="hero-sub">Smart recommendations. Just for you.</div>
            """,
            unsafe_allow_html=True,
        )

    current_query = st.text_input(
        "Search Movie", key="search_query",
        placeholder="Search for a movie...", label_visibility="collapsed",
    )
    query_stripped = current_query.strip()

    if query_stripped != st.session_state.last_query:
        st.session_state.last_query = query_stripped
        st.session_state.selected_idx = None

    if st.session_state.selected_idx is None and not query_stripped:
        st.markdown('<div class="section-title">Trending Now</div>', unsafe_allow_html=True)
        if "trending_sample" not in st.session_state:
            top_pool = df.sort_values("vote_count", ascending=False).head(30)
            st.session_state.trending_sample = top_pool.sample(min(TRENDING_COUNT, len(top_pool))).index
        render_selectable_grid(df.loc[st.session_state.trending_sample], columns_per_row=GRID_COLUMNS, key_prefix="trend")

    if st.session_state.selected_idx is not None:
        render_selected_and_recs()
    elif query_stripped:
        render_matches(query_stripped)


def render_explore():
    st.markdown('<div class="section-title">Explore</div>', unsafe_allow_html=True)
    if "explore_sample" not in st.session_state:
        st.session_state.explore_sample = df.sample(min(EXPLORE_COUNT, len(df))).index
    if st.button("🔄 Shuffle"):
        st.session_state.explore_sample = df.sample(min(EXPLORE_COUNT, len(df))).index
        st.rerun()
    render_selectable_grid(df.loc[st.session_state.explore_sample], columns_per_row=GRID_COLUMNS, key_prefix="explore", jump_home=True)


def render_top_rated():
    st.markdown('<div class="section-title">Top Rated</div>', unsafe_allow_html=True)
    top_df = df.sort_values("vote_average", ascending=False).head(TOP_RATED_COUNT)
    render_selectable_grid(top_df, columns_per_row=GRID_COLUMNS, key_prefix="toprated", jump_home=True)


def render_genres():
    st.markdown('<div class="section-title">Genres</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#16201e; font-size:1.4rem; font-weight:600; margin-top:20px;'>Coming Soon......</div>",
        unsafe_allow_html=True,
    )


def render_watchlist():
    st.markdown('<div class="section-title">Watchlist</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#16201e; font-size:1.4rem; font-weight:600; margin-top:20px;'>Coming Soon......</div>",
        unsafe_allow_html=True,
    )


def render_about():
    st.markdown('<div class="section-title">About FilmSense</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style='color:#3d4a48; line-height:1.8; max-width:700px; font-size:1rem;'>
            <p style='color:#16201e; font-size:1.15rem; font-weight:600; margin-bottom:18px;'>
                Find your next movie without endlessly searching.
            </p>
            <p>
                FilmSense is a movie discovery and recommendation platform created to make finding
                something worth watching simpler.
            </p>
            <p>
                Sometimes you know you enjoyed a particular movie, but you don't know what to watch
                next. FilmSense is built for exactly that.
            </p>
            <p>
                Search for a movie you like, and FilmSense finds other movies that share similar
                themes, genres, stories, people, and characteristics.
            </p>
            <p>
                FilmSense was created by <span style='color:#5f7d79; font-weight:600;'>Pankaj</span>,
                an engineering student interested in Machine Learning, Artificial Intelligence,
                Python, and building useful technology through real-world projects.
            </p>
            <p>
                This project started as an attempt to take what I was learning about Machine
                Learning and turn it into something people could actually use.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


page = st.session_state.current_page
if page == "Home":
    render_home()
elif page == "Explore":
    render_explore()
elif page == "Top Rated":
    render_top_rated()
elif page == "Genres":
    render_genres()
elif page == "Watchlist":
    render_watchlist()
elif page == "About":
    render_about()