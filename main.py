"""
Streamlit app for the ITI House Price Prediction project.

This app expects two artifacts produced at the end of the notebook
(ITI_Project.ipynb) to be present in the same folder as this script:

    - house_prices_cleaned.csv   (final cleaned dataset, from df.to_csv(...))
    - best_model.pkl             (best pipeline, from joblib.dump(best_model, ...))
    - XGBoost_Model.pkl          (optional, the XGBoost pipeline)

Run with:
    streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_PATH = "house_prices_cleaned.csv"
BEST_MODEL_PATH = "best_model.pkl"
XGB_MODEL_PATH = "best_Model2.pkl"

FEATURE_COLUMNS = [
    "bhk", "bathroom", "balcony",                       # numeric_features
    "area_sqft", "current_floor", "total_floors",        # skewed_numeric_features
    "Furnishing", "Transaction", "Ownership", "facing",   # categorical_features
    "location",
]
TARGET_COLUMN = "price_clean"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None, None
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption("Trained pipeline from the ITI house-price notebook (cleaning + feature engineering + regression).")

df = load_data(DATA_PATH)

if df is None:
    st.error(
        f"Couldn't find `{DATA_PATH}` next to this app.\n\n"
        "Run the notebook to the end (the `df.to_csv('house_prices_cleaned.csv', ...)` cell) "
        "and place the resulting file in this folder."
    )
    st.stop()

missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing_cols:
    st.error(f"The dataset is missing expected columns: {missing_cols}")
    st.stop()

# Model choice
available_models = {}
if os.path.exists(BEST_MODEL_PATH):
    available_models["Best model (best_model.pkl)"] = BEST_MODEL_PATH
if os.path.exists(XGB_MODEL_PATH):
    available_models["XGBoost (XGBoost_Model.pkl)"] = XGB_MODEL_PATH

if not available_models:
    st.error(
        "No trained model found. Expected `best_model.pkl` and/or `XGBoost_Model.pkl` "
        "next to this app — these are saved by the last cells of the notebook via `joblib.dump(...)`."
    )
    st.stop()

st.sidebar.header("⚙️ Settings")
model_label = st.sidebar.radio("Model to use", list(available_models.keys()))
model, load_error = load_model(available_models[model_label])

if load_error is not None:
    st.error(
        f"**`{available_models[model_label]}` failed to load — it's likely corrupted, not just outdated.**\n\n"
        f"Error: `{load_error}`\n\n"
        "This is a file-integrity issue, not a code issue. In Colab, run:\n\n"
        "```python\n"
        "import os, hashlib\n"
        f"print(os.path.getsize('{available_models[model_label]}'))\n"
        f"print(hashlib.md5(open('{available_models[model_label]}','rb').read()).hexdigest())\n"
        "```\n"
        "Then run the same two lines locally (adjust the path) and compare both numbers. "
        "If they differ, the file changed in transit — re-download it directly from Colab's file "
        "browser (right-click → Download) rather than copy/paste, git, or a sync tool, and try again."
    )
    other_models = {k: v for k, v in available_models.items() if k != model_label}
    if other_models:
        st.info(f"Meanwhile you can switch to **{list(other_models.keys())[0]}** in the sidebar.")
    st.stop()

st.sidebar.header("🏡 Property details")

# --- Build input widgets from the actual dataset's ranges / categories ---
locations = sorted(df["location"].dropna().unique().tolist())
furnishing_opts = sorted(df["Furnishing"].dropna().unique().tolist())
transaction_opts = sorted(df["Transaction"].dropna().unique().tolist())
ownership_opts = sorted(df["Ownership"].dropna().unique().tolist())
facing_opts = sorted(df["facing"].dropna().unique().tolist())

col1, col2 = st.columns(2)

with col1:
    bhk = st.number_input(
        "BHK", min_value=1, max_value=int(df["bhk"].max()), value=2, step=1
    )
    bathroom = st.number_input(
        "Bathrooms", min_value=1, max_value=int(df["bathroom"].max()), value=2, step=1
    )
    balcony = st.number_input(
        "Balconies", min_value=0, max_value=int(df["balcony"].max()), value=1, step=1
    )
    area_sqft = st.number_input(
        "Area (sqft)",
        min_value=50.0,
        max_value=20000.0,
        value=float(round(df["area_sqft"].median(), 1)),
        step=10.0,
    )

with col2:
    total_floors = st.number_input(
        "Total floors in building",
        min_value=1.0,
        max_value=float(max(100, df["total_floors"].max())),
        value=float(round(df["total_floors"].median(), 1)),
        step=1.0,
    )
    current_floor = st.number_input(
        "Current floor",
        min_value=0.0,
        max_value=total_floors,
        value=min(float(round(df["current_floor"].median(), 1)), total_floors),
        step=1.0,
    )
    location = st.selectbox("Location", locations)

furnishing = st.selectbox("Furnishing", furnishing_opts)
transaction = st.selectbox("Transaction type", transaction_opts)
ownership = st.selectbox("Ownership", ownership_opts)
facing = st.selectbox("Facing", facing_opts)

st.divider()

if st.button("Predict price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "bhk": bhk,
        "bathroom": bathroom,
        "balcony": balcony,
        "area_sqft": area_sqft,
        "current_floor": current_floor,
        "total_floors": total_floors,
        "Furnishing": furnishing,
        "Transaction": transaction,
        "Ownership": ownership,
        "facing": facing,
        "location": location,
    }])[FEATURE_COLUMNS]

    try:
        pred = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success(f"Estimated price: ₹ {pred:,.0f}")

    if pred >= 1e7:
        st.metric("Approx. value", f"{pred / 1e7:.2f} Cr")
    else:
        st.metric("Approx. value", f"{pred / 1e5:.2f} Lac")

    price_per_sqft = pred / area_sqft
    st.caption(f"≈ ₹{price_per_sqft:,.0f} per sqft")

st.divider()

with st.expander("📊 Dataset preview"):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

with st.expander("📤 Batch prediction (upload a CSV)"):
    st.write(f"Upload a CSV with columns: `{', '.join(FEATURE_COLUMNS)}`")
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing:
            st.error(f"Uploaded file is missing columns: {missing}")
        else:
            batch_df["predicted_price"] = model.predict(batch_df[FEATURE_COLUMNS])
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download predictions",
                batch_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )