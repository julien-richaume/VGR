import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

# Allow override via environment variable for portability
PATH_TO_DATA = os.environ.get("STEAM_DATA_PATH", "./data")


@st.cache_data
def load_data(path_to_data):
    steam200k = pd.read_csv(f"{path_to_data}/steam200k.csv")
    steam200k.columns = ["user_id", "title", "status", "playtime", "osef"]
    steam200k = steam200k[steam200k["status"] == "play"].drop(columns=["status", "osef"])

    games = pd.read_csv(f"{path_to_data}/games.csv").sort_values("app_id")
    games_metadata = pd.read_json(f"{path_to_data}/games_metadata.json", lines=True)
    games = pd.merge(games, games_metadata, on="app_id")

    merged = pd.merge(games, steam200k, on="title")
    success_ratios = (
        merged[["title", "positive_ratio"]]
        .drop_duplicates("title")
        .set_index("title")["positive_ratio"]
    )
    # Build title -> app_id lookup for Steam CDN cover images
    app_id_map = (
        merged[["title", "app_id"]]
        .drop_duplicates("title")
        .set_index("title")["app_id"]
    )
    return merged, success_ratios, app_id_map


@st.cache_data
def build_hybrid_engine(_df):
    """Build tag matrix, user-item matrix, and SVD predictions."""
    # Build tags_dummies with game title as index.
    # "tags" may be a JSON list, a comma-separated string, or mixed —
    # normalise everything to a flat list before exploding.
    tags_base = _df[["title", "tags"]].drop_duplicates("title").copy()

    def _normalise_tags(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            # Handle stringified Python lists: "['Action', 'RPG', 'Indie']"
            if val.startswith("["):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return [str(t).strip() for t in parsed if str(t).strip()]
                except (ValueError, SyntaxError):
                    pass
            # Plain comma-separated: "Action, RPG, Indie"
            return [t.strip() for t in val.split(",") if t.strip()]
        return []

    tags_base["tags"] = tags_base["tags"].apply(_normalise_tags)
    tags_exploded = tags_base.explode("tags").dropna(subset=["tags"])
    tags_exploded = tags_exploded[tags_exploded["tags"] != ""]
    tags_dummies = (
        pd.get_dummies(tags_exploded.set_index("title")["tags"])
        .groupby(level=0)
        .sum()
    )

    matrix = _df.pivot_table(index="user_id", columns="title", values="playtime").fillna(0)

    k = min(50, min(matrix.shape) - 1)
    U, sigma, Vt = svds(matrix.values.astype(float), k=k)
    preds_df = pd.DataFrame(
        np.dot(np.dot(U, np.diag(sigma)), Vt),
        columns=matrix.columns,
        index=matrix.index,
    )
    return matrix, preds_df, tags_dummies


# ─────────────────────────────────────────────────────────────
# RECOMMENDATION LOGIC
# ─────────────────────────────────────────────────────────────

def get_hybrid_recommendations(
    user_id, matrix, preds_df, tags_dummies, success_ratios, alpha=0.5, beta=0.2, top_n=5
):
    # Guard: user must exist in preds_df
    if user_id not in preds_df.index:
        st.error(f"User ID {user_id} not found in the SVD predictions matrix.")
        return pd.Series(dtype=float)

    common_games = (
        matrix.columns
        .intersection(tags_dummies.index)
        .intersection(success_ratios.index)
    )
    tags_aligned = tags_dummies.loc[common_games]
    preds_aligned = preds_df[common_games]
    matrix_aligned = matrix[common_games]
    ratios_aligned = success_ratios.loc[common_games]

    collab_scores = preds_aligned.loc[user_id]
    user_playtime = matrix_aligned.loc[user_id]
    user_profile = user_playtime.values @ tags_aligned.values

    if user_profile.sum() == 0:
        final_scores = collab_scores
    else:
        content_scores = cosine_similarity(
            user_profile.reshape(1, -1), tags_aligned.values
        ).flatten()
        content_scores = pd.Series(content_scores, index=common_games)
        final_scores = alpha * collab_scores + (1 - alpha) * content_scores

    final_scores = final_scores * (1 + beta * ratios_aligned)

    played = matrix_aligned.columns[matrix_aligned.loc[user_id] > 0]
    final_scores = final_scores.drop(played, errors="ignore")
    return final_scores.sort_values(ascending=False).head(top_n)


def get_content_recommendations(selected_games, tags_dummies, top_n=5):
    valid = [g for g in selected_games if g in tags_dummies.index]
    if not valid:
        return pd.Series(dtype=float)

    user_profile = tags_dummies.loc[valid].mean(axis=0)

    # Drop tag columns that are all-zero across the whole matrix to avoid
    # the "0 features" error when the profile and matrix are fully sparse.
    active_cols = tags_dummies.columns[tags_dummies.sum(axis=0) > 0]
    if active_cols.empty:
        st.warning("Tag matrix has no active features — cannot compute recommendations.")
        return pd.Series(dtype=float)

    profile_vec = user_profile[active_cols].values.reshape(1, -1)
    matrix_mat = tags_dummies[active_cols].values

    # Guard: if profile is still a zero vector, cosine similarity is undefined
    if profile_vec.sum() == 0:
        st.warning(
            "None of the selected games have tag data. "
            "Try different titles or check that tags loaded correctly."
        )
        return pd.Series(dtype=float)

    scores = cosine_similarity(profile_vec, matrix_mat).flatten()
    recos = pd.Series(scores, index=tags_dummies.index).sort_values(ascending=False)
    return recos.drop(valid, errors="ignore").head(top_n)


# ─────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────

def _steam_cover_url(app_id):
    """Return the Steam CDN portrait cover URL for a given app_id."""
    return f"https://cdn.akamai.steamstatic.com/steam/apps/{int(app_id)}/library_600x900.jpg"


def render_recommendation_card(game, score, tags_dummies, app_id_map):
    # Fix: ensure score is a clean float, default 0 if NaN
    score = float(score) if not np.isnan(float(score)) else 0.0
    with st.container(border=True):
        col_img, col_info = st.columns([1, 4])
        with col_img:
            if game in app_id_map.index:
                img_url = _steam_cover_url(app_id_map[game])
            else:
                img_url = "https://via.placeholder.com/120x160?text=No+image+found+on+Steam+API"
            st.image(img_url, use_container_width=True)
        with col_info:
            st.markdown(f"### {game}")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(f"Relevance score: **{score:.2f}**")
            if game in tags_dummies.index:
                active_tags = tags_dummies.loc[game]
                top_tags = active_tags[active_tags > 0].index[:5].tolist()
                st.write(" · ".join(top_tags))


def show_recommendations(recos, tags_dummies, app_id_map):
    if recos.empty:
        st.info("No recommendations available. Try selecting different games.")
        return
    recos = recos.fillna(0)
    for game, score in recos.items():
        render_recommendation_card(game, score, tags_dummies, app_id_map)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — global controls
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
    st.divider()
    st.caption(f"Data path: `{PATH_TO_DATA}`")


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Steam Recommender", page_icon="🎮", layout="wide")

# Load data once
df, success_ratios, app_id_map = load_data(PATH_TO_DATA)
matrix, preds, tags = build_hybrid_engine(df)

# Header
st.title("🎮 Steam Game Recommender")
st.caption("Hybrid collaborative + content-based recommendations")

# Tabs
tab_new, tab_existing, tab_browse = st.tabs(
    ["🆕 New User", "🔑 Existing User", "👥 Browse Users"]
)

# ───────────── TAB 1: New User ─────────────
with tab_new:
    st.subheader("Tell us what you like")
    st.write(
        "Select games you've enjoyed and we'll find similar ones based on their tags. "
        "No account needed — purely based on what you pick."
    )

    all_games = sorted(tags.index.tolist())
    selected = st.multiselect(
        "Select games you like",
        options=all_games,
        placeholder="Start typing a game title...",
        help="Pick 3–10 games for best results",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Get Recommendations", type="primary", disabled=len(selected) == 0):
            # Clear stale results before computing new ones
            st.session_state.pop("new_user_recos", None)
            with st.spinner("Calculating..."):
                recos = get_content_recommendations(selected, tags, top_n=top_n)
            st.session_state["new_user_recos"] = recos

    if "new_user_recos" in st.session_state:
        st.divider()
        st.subheader("Recommended for you")
        show_recommendations(st.session_state["new_user_recos"], tags, app_id_map)


# ───────────── TAB 2: Existing User ─────────────
with tab_existing:
    st.subheader("Recommendations for returning users")
    st.write(
        "Recommendations here combine **your play history** (collaborative filtering) "
        "with **game tags** (content-based), weighted by review scores."
    )

    with st.expander("⚙️ Tune the hybrid model"):
        alpha = st.slider(
            "Collaborative vs Content weight (α)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="1.0 = pure collaborative filtering · 0.0 = pure content-based",
        )
        beta = st.slider(
            "Popularity boost (β)",
            min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="Higher values favour games with better review ratios",
        )

    # ── Game-title lookup ──────────────────────────────────────
    all_game_titles = sorted(matrix.columns.tolist())

    lookup_games = st.multiselect(
        "🔍 Find your user ID by game titles you've played",
        options=all_game_titles,
        placeholder="Type a game title...",
        help="Select one or more games; we'll find users who played all of them",
    )

    if lookup_games:
        # Keep only columns that exist in the matrix
        valid_lookup = [g for g in lookup_games if g in matrix.columns]
        if valid_lookup:
            mask = (matrix[valid_lookup] > 0).all(axis=1)
            matching_users = matrix.index[mask].tolist()
        else:
            matching_users = []

        if matching_users:
            st.success(f"Found **{len(matching_users)}** user(s) who played all selected games.")
            user_id = st.selectbox(
                "Select your User ID",
                options=matching_users,
                key="existing_uid_from_games",
            )
        else:
            st.warning("No user found who played all those games — try fewer titles.")
            user_id = None
    else:
        user_id = st.selectbox(
            "Or select a User ID directly",
            options=matrix.index.tolist(),
            help="Numeric Steam user ID from the dataset",
            key="existing_uid_direct",
        )

    # ── Play history ──────────────────────────────────────────
    if user_id is not None:
        with st.expander("View play history for this user"):
            user_games = matrix.loc[user_id]
            played = user_games[user_games > 0].sort_values(ascending=False)
            if played.empty:
                st.write("No recorded playtime.")
            else:
                # Fix: rename correctly — after reset_index the columns are 'title' and 0
                played_df = (
                    played
                    .reset_index()
                    .rename(columns={"title": "Game", 0: "Playtime (hrs)"})
                )
                st.dataframe(played_df, use_container_width=True, hide_index=True)

        if st.button("Generate Recommendations", type="primary", key="existing_btn"):
            # Clear stale results before computing new ones
            st.session_state.pop("existing_recos", None)
            with st.spinner("Running hybrid model..."):
                recos = get_hybrid_recommendations(
                    user_id, matrix, preds, tags, success_ratios,
                    alpha=alpha, beta=beta, top_n=top_n,
                )
            st.session_state["existing_recos"] = recos

    if "existing_recos" in st.session_state:
        st.divider()
        st.subheader("Your personalized picks")
        show_recommendations(st.session_state["existing_recos"], tags, app_id_map)


# ───────────── TAB 3: Browse Users ─────────────
with tab_browse:
    st.subheader("Explore users in the dataset")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Users", f"{len(matrix):,}")
    col_b.metric("Total Games", f"{len(tags):,}")
    col_c.metric("Interactions", f"{(matrix > 0).sum().sum():,}")

    st.divider()

    st.write("**User activity overview** (sorted by total playtime)")

    user_stats = (
        matrix.sum(axis=1)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"user_id": "User ID", 0: "Total Playtime (hrs)"})
    )
    user_stats["Games Played"] = (matrix > 0).sum(axis=1).values

    page_size = 25
    total_pages = max(1, (len(user_stats) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * page_size
    end = start + page_size

    st.dataframe(
        user_stats.iloc[start:end],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing {start + 1}–{min(end, len(user_stats))} of {len(user_stats)} users")