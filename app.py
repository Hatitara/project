import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import kagglehub

PLACEHOLDER_IMAGE = "https://via.placeholder.com/225x320?text=No+Image"
JIKAN_PLACEHOLDER_HINTS = [
    "apple-touch-icon",
    "questionmark",
    "no_image",
    "placeholder",
]


def create_search_label(row: pd.Series) -> str:

    name = row.get("Name", row.get("name", ""))
    if pd.isna(name) or str(name).strip() == "":
        name = "Unknown"
    parts = [str(name)]

    eng = row.get("English name", "Unknown")
    if pd.notna(eng):
        eng_str = str(eng).strip()
        if eng_str and eng_str != "Unknown":
            parts.append(eng_str)

    jap = row.get("Japanese name", "Unknown")
    if pd.notna(jap):
        jap_str = str(jap).strip()
        if jap_str and jap_str != "Unknown":
            parts.append(jap_str)

    return " | ".join(parts)


@st.cache_data(show_spinner=False)
def load_anime_csv() -> pd.DataFrame:
    """Load anime.csv for full metadata including English and Japanese names."""
    if not os.path.exists("/kaggle/input/anime-recommendation-database-2020/anime.csv"):
        os.environ['KAGGLE_USERNAME'] = "hatitara"
        os.environ['KAGGLE_KEY'] = "8076631b38e08331efbdfe56a0264de0"
        path = kagglehub.dataset_download("hernan4444/anime-recommendation-database-2020")

    df = pd.read_csv(os.path.join(path, 'anime.csv'))

    if "MAL_ID" in df.columns:
        df["anime_id"] = df["MAL_ID"].astype(str)

    for col in ["Name", "English name", "Japanese name"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model and metadata from disk once per session.

    Merges pickle data with anime.csv to get full metadata.
    """
    artifact_path = Path(__file__).parent / "anime_lightfm_final.pkl"
    with artifact_path.open("rb") as f:
        data = pickle.load(f)

    model = data["model"]
    dataset = data["dataset"]
    item_features = data["item_features"]
    anime_df: pd.DataFrame = data["anime_df"]

    csv_df = load_anime_csv()

    if "anime_id" in anime_df.columns:
        anime_df["anime_id"] = anime_df["anime_id"].astype(str)

    if not csv_df.empty and "anime_id" in csv_df.columns:

        merge_cols = ["anime_id"]
        csv_cols_to_merge = [
            "Name",
            "English name",
            "Japanese name",
            "Rating",
            "Members",
        ]
        csv_cols_to_merge = [col for col in csv_cols_to_merge if col in csv_df.columns]

        csv_subset = csv_df[merge_cols + csv_cols_to_merge].copy()

        anime_df = anime_df.merge(
            csv_subset, on="anime_id", how="left", suffixes=("", "_csv")
        )

        for col in csv_cols_to_merge:
            csv_col = f"{col}_csv"
            if csv_col in anime_df.columns:

                if col in anime_df.columns:

                    anime_df[col] = anime_df[csv_col].fillna(anime_df[col])
                else:
                    anime_df[col] = anime_df[csv_col]

                anime_df = anime_df.drop(columns=[csv_col])

    if "Members" in anime_df.columns:
        anime_df["Members"] = (
            anime_df["Members"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", "0")
            .astype(float)
        )

    for col in ["Name", "English name", "Japanese name"]:
        if col in anime_df.columns:
            anime_df[col] = anime_df[col].fillna("Unknown")

    english_col = "English name" if "English name" in anime_df.columns else None
    base_name_col = "Name" if "Name" in anime_df.columns else "name"
    if english_col and base_name_col in anime_df.columns:
        anime_df["Display_Name"] = np.where(
            (anime_df[english_col].fillna("").str.strip() != "") & (anime_df[english_col] != "Unknown"),
            anime_df[english_col],
            anime_df[base_name_col],
        )
    else:
        anime_df["Display_Name"] = anime_df.get(base_name_col, anime_df.get("anime_id", ""))

    anime_df["search_label"] = anime_df.apply(create_search_label, axis=1)

    name_col = "Name" if "Name" in anime_df.columns else "name"
    if name_col in anime_df.columns:
        anime_df["_original_name"] = anime_df[name_col]
    elif "name" in anime_df.columns:
        anime_df["_original_name"] = anime_df["name"]
    else:
        anime_df["_original_name"] = anime_df["anime_id"]

    return model, dataset, item_features, anime_df


@st.cache_resource(show_spinner=False)
def prepare_item_data():
    """Prepare item embeddings and mapping dictionaries."""
    model, dataset, item_features, anime_df = load_artifacts()
    _, item_embeddings = model.get_item_representations(features=item_features)

    item_id_map: Dict[str, int] = dataset.mapping()[2]
    reverse_item_map: Dict[int, str] = {v: k for k, v in item_id_map.items()}
    return item_embeddings, item_id_map, reverse_item_map, anime_df


def _looks_like_placeholder(url: Optional[str]) -> bool:
    if not url:
        return True
    lower = url.lower()
    if url == PLACEHOLDER_IMAGE:
        return True
    return any(hint in lower for hint in JIKAN_PLACEHOLDER_HINTS)


@st.cache_data(show_spinner=False)
def fetch_image_url(anime_id: str) -> Optional[str]:
    """Fetch cover image URL for an anime using the Jikan API."""
    url = f"https://api.jikan.moe/v4/anime/{anime_id}"
    try:
        response = requests.get(url, timeout=8)
        if response.status_code != 200:
            return None
        data = response.json()
        image_url = (
            data.get("data", {}).get("images", {}).get("jpg", {}).get("image_url")
        )
        if _looks_like_placeholder(image_url):
            return None
        return image_url
    except Exception:
        return None


def resolve_item_index(anime_id: str, item_id_map: Dict[str, int]) -> Optional[int]:
    """Return the internal LightFM index for an anime id."""
    if anime_id in item_id_map:
        return item_id_map[anime_id]
    str_id = str(anime_id)
    if str_id in item_id_map:
        return item_id_map[str_id]
    try:
        int_id = int(float(anime_id))
        if int_id in item_id_map:
            return item_id_map[int_id]
    except Exception:
        pass
    return None


def filter_anime_df(
    anime_df: pd.DataFrame, user_age: int, hide_adult: bool = False
) -> pd.DataFrame:
    """Filter anime by Rating according to age and optional adult toggle."""
    if anime_df is None or anime_df.empty:
        return anime_df

    allowed_df = anime_df.copy()
    if "Rating" not in allowed_df.columns:
        return allowed_df

    adult_patterns = ["rx", "hentai", "r+", "r - 17+"]
    ratings = allowed_df["Rating"].fillna("").str.lower()

    if user_age < 18 or hide_adult:
        mask = ~ratings.str.contains("|".join(adult_patterns), regex=True)
        allowed_df = allowed_df[mask]

    return allowed_df


def compute_recommendations(
    selected_names: List[str],
    anime_df: pd.DataFrame,
    item_vectors: np.ndarray,
    item_id_map: Dict[str, int],
    reverse_item_map: Dict[int, str],
    top_k: int = 12,
) -> Tuple[pd.DataFrame, List[str], List[str], bool]:
    """Generate top-k recommendations based on selected titles using filtered data.

    Returns:
        rec_df: DataFrame of recommendations (or popularity fallback).
        missing_names: selections not found in mapping.
        missing_ids: ids not found in model mapping.
        used_popularity: True if fallback was used due to no selections.
    """
    if anime_df is None or anime_df.empty:
        return pd.DataFrame(), [], [], False

    if not selected_names:
        buffer_size = min(len(anime_df), top_k + 10)
        if "Members" in anime_df.columns:
            candidates = (
                anime_df.sort_values("Members", ascending=False)
                .head(buffer_size)
                .reset_index(drop=True)
            )
        else:
            candidates = anime_df.head(buffer_size).reset_index(drop=True)

        kept_rows = []
        for _, row in candidates.iterrows():
            anime_id = str(row.get("anime_id", ""))
            image_url = fetch_image_url(anime_id) if anime_id else None
            if image_url:
                row = row.copy()
                row["image_url"] = image_url
                kept_rows.append(row)
            if len(kept_rows) >= top_k:
                break

        popular_df = pd.DataFrame(kept_rows)
        return popular_df, [], [], True

    name_to_id = (
        anime_df[["name", "anime_id"]]
        .dropna(subset=["name", "anime_id"])
        .set_index("name")["anime_id"]
        .to_dict()
    )
    selected_ids = []
    missing_names = []
    for name in selected_names:
        anime_id = name_to_id.get(name)
        if anime_id is None:
            missing_names.append(name)
        else:
            selected_ids.append(anime_id)

    selected_indices = [
        resolve_item_index(anime_id, item_id_map) for anime_id in selected_ids
    ]
    missing_ids = []
    resolved_indices = []
    for anime_id, idx in zip(selected_ids, selected_indices):
        if idx is None:
            missing_ids.append(anime_id)
        else:
            resolved_indices.append(idx)

    if not resolved_indices:
        return pd.DataFrame(), missing_names, missing_ids, False

    allowed_indices = [
        resolve_item_index(aid, item_id_map)
        for aid in anime_df["anime_id"].dropna().astype(str).tolist()
    ]
    allowed_indices = [idx for idx in allowed_indices if idx is not None]
    allowed_mask = np.zeros(item_vectors.shape[0], dtype=bool)
    allowed_mask[allowed_indices] = True

    user_vector = item_vectors[resolved_indices].mean(axis=0)
    item_norms = np.linalg.norm(item_vectors, axis=1)
    user_norm = np.linalg.norm(user_vector)

    scores = np.zeros(item_vectors.shape[0], dtype=float)
    denom = item_norms * user_norm
    valid_mask = (denom > 0) & allowed_mask
    scores[valid_mask] = (item_vectors[valid_mask] @ user_vector) / denom[valid_mask]
    scores[~allowed_mask] = -np.inf

    for idx in resolved_indices:
        scores[idx] = -np.inf

    sorted_indices = np.argsort(-scores)
    sorted_indices = [idx for idx in sorted_indices if np.isfinite(scores[idx])]
    top_indices = sorted_indices[:top_k]

    buffer_size = min(len(sorted_indices), top_k + 10)
    buffered_indices = sorted_indices[:buffer_size]

    kept_rows = []
    anime_indexed = anime_df.set_index("anime_id")
    for idx in buffered_indices:
        anime_id = reverse_item_map.get(idx)
        if anime_id is None:
            continue
        image_url = fetch_image_url(anime_id)
        if not image_url:
            continue

        row = anime_indexed.loc[[str(anime_id)]].reset_index()
        if row.empty:
            continue
        row = row.iloc[0].copy()
        row["similarity"] = scores[idx]
        row["image_url"] = image_url
        kept_rows.append(row)
        if len(kept_rows) >= top_k:
            break

    rec_df = pd.DataFrame(kept_rows)
    return rec_df.dropna(subset=["anime_id"]), missing_names, missing_ids, False


def display_grid(df: pd.DataFrame, title: str, columns: int = 4):
    """Render a grid of anime cards."""
    st.subheader(title)
    if df.empty:
        st.info("No items to display yet.")
        return

    rows = list(df.to_dict(orient="records"))
    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, row in zip(cols, rows[start : start + columns]):
            with col:
                anime_id = str(row.get("anime_id", ""))
                image_url = row.get("image_url")
                if not image_url and anime_id:
                    image_url = fetch_image_url(anime_id)
                if not image_url:
                    continue
                col.image(image_url, use_container_width=True)
                display_title = row.get("Display_Name") or row.get("name") or row.get("Name") or "Unknown"
                col.markdown(f"**{display_title}**")

                genres = row.get("Genres") or row.get("genres")
                if pd.notna(genres):
                    col.caption(str(genres))

                rating = (
                    row.get("rating")
                    or row.get("Rating")
                    or row.get("score")
                    or row.get("Score")
                )
                if pd.notna(rating):
                    col.write(f"⭐ {rating}")


def main():
    st.set_page_config(page_title="Anime Recommender", layout="wide")
    st.session_state.setdefault("age_verified", False)
    st.session_state.setdefault("user_age", None)

    if not st.session_state["age_verified"]:
        st.title("Content Warning")
        st.warning(
            "This app contains mature themes. Please verify your age to continue."
        )
        age_input = st.text_input("Please enter your age", value="")
        if st.button("Enter App"):

            if not age_input or not age_input.strip():
                st.error("Please enter a valid number.")
            else:
                try:
                    age = int(age_input.strip())

                    if 1 <= age <= 120:
                        st.session_state["user_age"] = age
                        st.session_state["age_verified"] = True
                        st.rerun()
                    else:
                        st.error("Age must be between 1 and 120.")
                except ValueError:
                    st.error("Please enter a valid number.")
        return

    user_age = st.session_state.get("user_age") or 0
    hide_adult = False

    if user_age >= 18:
        hide_adult = st.sidebar.checkbox("Hide adult content", value=False)

    st.title("Anime Recommendation System")
    st.write(
        "Select a few anime you have watched. We'll build a profile from your picks "
        "and recommend similar titles using item embeddings."
    )

    item_vectors, item_id_map, reverse_item_map, anime_df_full = prepare_item_data()
    anime_df = filter_anime_df(anime_df_full, user_age=user_age, hide_adult=hide_adult)

    if "_original_name" in anime_df.columns and "name" not in anime_df.columns:
        anime_df["name"] = anime_df["_original_name"]
    elif "Name" in anime_df.columns and "name" not in anime_df.columns:
        anime_df["name"] = anime_df["Name"]

    options = sorted(anime_df["search_label"].dropna().unique().tolist())

    label_to_name = (
        anime_df[["search_label", "name"]]
        .dropna(subset=["search_label", "name"])
        .set_index("search_label")["name"]
        .to_dict()
    )

    selected_labels = st.multiselect(
        "Search and select anime you have watched",
        options=options,
        default=[],
    )

    selected_names = [label_to_name.get(label) for label in selected_labels]
    selected_names = [name for name in selected_names if name is not None]

    selected_df = anime_df[anime_df["name"].isin(selected_names)]
    with st.container():
        display_grid(selected_df, title="Your List", columns=5)

    top_k = st.slider("Number of recommendations", min_value=4, max_value=30, value=12)

    if st.button("Generate Recommendations"):
        rec_df, missing_names, missing_ids, used_popularity = compute_recommendations(
            selected_names,
            anime_df=anime_df,
            item_vectors=item_vectors,
            item_id_map=item_id_map,
            reverse_item_map=reverse_item_map,
            top_k=top_k,
        )
        if missing_names:
            st.warning(
                "These selections could not be matched to the model ids: "
                + ", ".join(missing_names)
            )
        if missing_ids:
            st.warning(
                "Some anime IDs were not found in the model mapping: "
                + ", ".join(map(str, missing_ids))
            )
        st.session_state["recommendations"] = {
            "df": rec_df,
            "used_popularity": used_popularity,
        }

    rec_state = st.session_state.get(
        "recommendations", {"df": pd.DataFrame(), "used_popularity": False}
    )
    if isinstance(rec_state, pd.DataFrame):
        rec_df = rec_state
        used_popularity = False
    else:
        rec_df = rec_state.get("df", pd.DataFrame())
        used_popularity = rec_state.get("used_popularity", False)

    if not isinstance(rec_df, pd.DataFrame):
        rec_df = pd.DataFrame(rec_df)

    if not rec_df.empty:
        title = (
            "Top Trending Anime (No selection made)"
            if used_popularity
            else "Recommendations"
        )
        display_grid(rec_df, title=title, columns=5)


if __name__ == "__main__":
    main()
