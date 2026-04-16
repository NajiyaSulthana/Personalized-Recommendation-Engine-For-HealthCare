import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Healthcare Recommender", layout="wide")

# =========================
# LOAD DATA (CACHE)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    df["diseases"] = df["diseases"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["diseases"]).reset_index(drop=True)
    return df

# =========================
# BUILD MODEL (CACHE)
# =========================
@st.cache_resource
def build_model(df):
    X = df.drop(columns=["diseases"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(X)

    return model, X

# =========================
# LOAD DATA + MODEL
# =========================
df = load_data()
model, X = build_model(df)

# =========================
# TITLE
# =========================
st.title("🏥 Healthcare Recommendation System")
st.write("Select a disease and get similar recommendations")

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(disease_name, top_n=5):

    idx_list = df[df["diseases"] == disease_name].index

    if len(idx_list) == 0:
        return []

    idx = idx_list[0]

    distances, indices = model.kneighbors(
        X.iloc[[idx]],
        n_neighbors=top_n + 10
    )

    results = df.iloc[indices[0]]["diseases"].values

    # remove same disease
    results = [r for r in results if r != disease_name]

    # remove duplicates
    seen = set()
    final = []

    for r in results:
        if r not in seen:
            final.append(r)
            seen.add(r)

    return final[:top_n]

# =========================
# DROPDOWN LIST
# =========================
disease_list = sorted(df["diseases"].unique())

selected_disease = st.selectbox(
    "Select Disease",
    disease_list
)

# =========================
# BUTTON
# =========================
if st.button("Get Recommendations"):

    results = recommend(selected_disease)

    if len(results) == 0:
        st.error("No recommendations found")

    else:
        st.success(f"Recommendations for: {selected_disease}")

        for i, r in enumerate(results, 1):
            st.write(f"{i}. {r}")
