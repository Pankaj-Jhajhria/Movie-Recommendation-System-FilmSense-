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
NUM_BG_IMAGES = 3

st.set_page_config(page_title="FilmSense", page_icon=str(LOGO_PATH), layout="wide")


@st.cache_resource
def get_data():
    return load_artifacts()


df, tfidf_matrix, nn_model = get_data()


# ---------------- helpers ----------------

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


def pick_background_images(n=NUM_BG_IMAGES):
    pool = df["backdrop_path"].dropna()
    if pool.empty:
        return []
    sample = pool.sample(min(n, len(pool)))
    return [backdrop_url(p) for p in sample]


def render_selectable_grid(movies_df, columns_per_row, key_prefix, jump_home=False):
    """Plain poster + a minimal clickable title/year link beneath it."""
    rows = [movies_df.iloc[i:i + columns_per_row] for i in range(0, len(movies_df), columns_per_row)]
    for row_df in rows:
        cols = st.columns(columns_per_row)
        for col, (idx, movie) in zip(cols, row_df.iterrows()):
            with col:
                st.image(poster_url(movie["poster_path"]), width='stretch')
                label = f"{movie['title']} ({_fmt_year(movie['release_year'])})"
                st.markdown('<div class="grid-btn">', unsafe_allow_html=True)
                clicked = st.button(label, key=f"{key_prefix}_{idx}", width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                if clicked:
                    st.session_state.selected_idx = idx
                    if jump_home:
                        st.session_state.current_page = "Home"
                    st.rerun()


# ---------------- session defaults ----------------

defaults = {
    "last_query": "",
    "selected_idx": None,
    "current_page": "Home",
    "bg_images": None,
    "bg_tracker": "__init__",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Background only regenerates when the selected movie actually changes (incl. first load)
if st.session_state.bg_tracker != st.session_state.selected_idx or st.session_state.bg_images is None:
    st.session_state.bg_images = pick_background_images()
    st.session_state.bg_tracker = st.session_state.selected_idx

bg_imgs_html = "".join(f'<img src="{u}">' for u in st.session_state.bg_images if u)

# ---------------- global styling ----------------

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    #bg-collage {{
        position: fixed;
        inset: 0;
        z-index: -1;
        display: flex;
        filter: blur(14px) brightness(0.35) saturate(1.15);
        transform: scale(1.08);
    }}
    #bg-collage img {{
        flex: 1;
        object-fit: cover;
        height: 100vh;
        width: 100%;
    }}

    .stApp {{ background: #0a0a0f; }}
    [data-testid="stHeader"] {{ background: transparent; }}

    [data-testid="stSidebar"] {{
        background: rgba(10, 8, 14, 0.92);
        border-right: 1px solid rgba(230, 57, 70, 0.15);
    }}
    .sidebar-logo {{ display: flex; align-items: center; gap: 10px; }}
    .sidebar-logo img {{ width: 42px; height: 42px; border-radius: 10px; }}
    .sidebar-logo .brand {{ font-family: 'Poppins', sans-serif; font-weight: 800; font-size: 1.3rem; color: #ffffff; }}
    .sidebar-logo .brand span {{ color: #e63946; }}
    .sidebar-tagline {{ color: #8b8b96; font-size: 0.72rem; margin: 2px 0 16px 52px; }}

    .nav-active {{
        background: #e63946;
        color: white !important;
        font-weight: 600;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 4px;
    }}
    [data-testid="stSidebar"] button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #c9c9d1 !important;
        text-align: left !important;
        font-weight: 500 !important;
    }}
    [data-testid="stSidebar"] button:hover {{ color: #ffffff !important; }}

    .hero-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        text-shadow: 0 2px 20px rgba(0,0,0,0.6);
        margin-bottom: 6px;
    }}
    .hero-title .accent {{ color: #e63946; }}
    .hero-sub {{ color: #c9c9d1; font-size: 1.05rem; margin-bottom: 22px; }}

    .section-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 4px solid #e63946;
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }}

    .grid-btn button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #d8d8de !important;
        padding: 4px 0 !important;
        font-size: 0.82rem !important;
        text-align: left !important;
    }}
    .grid-btn button:hover {{ color: #e63946 !important; }}

    .rating-pill {{
        display: inline-block;
        background: rgba(230,57,70,0.15);
        color: #ff8a94;
        border: 1px solid rgba(230,57,70,0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0 8px 8px 0;
    }}

    div[data-testid="stTextInput"] input {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }}
    </style>

    <div id="bg-collage">{bg_imgs_html}</div>
    """,
    unsafe_allow_html=True,
)

# ---------------- sidebar ----------------

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


# ---------------- page renderers ----------------

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
            f"<div style='font-family:Poppins,sans-serif; font-size:2rem; font-weight:800; color:#fff;'>{movie['title']}</div>",
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
        "<div style='color:#ffffff; font-size:1.4rem; font-weight:600; margin-top:20px;'>Coming Soon......</div>",
        unsafe_allow_html=True,
    )


def render_watchlist():
    st.markdown('<div class="section-title">Watchlist</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#ffffff; font-size:1.4rem; font-weight:600; margin-top:20px;'>Coming Soon......</div>",
        unsafe_allow_html=True,
    )


def render_about():
    st.markdown('<div class="section-title">About FilmSense</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style='color:#c9c9d1; line-height:1.8; max-width:700px; font-size:1rem;'>
            <p style='color:#ffffff; font-size:1.15rem; font-weight:600; margin-bottom:18px;'>
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
                FilmSense was created by <span style='color:#e63946; font-weight:600;'>Pankaj</span>,
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