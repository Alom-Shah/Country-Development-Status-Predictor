import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Country Development Predictor", page_icon="🌍", layout="wide"
)

LOGO_URL = "https://i.postimg.cc/B6bxTPHx/Whats-App-Image-2026-11-08-at-7-57-57-PM.jpg"


@st.cache_resource
def load_artifacts():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts = joblib.load(os.path.join(script_dir, "model.pkl"))
    # accept either spelling of the dataset file name
    csv_path = next(
        (
            os.path.join(script_dir, n)
            for n in ("Life Expectancy Data.csv", "Life_Expectancy_Data.csv")
            if os.path.exists(os.path.join(script_dir, n))
        ),
        None,
    )
    if csv_path is None:
        raise FileNotFoundError(
            "Put 'Life Expectancy Data.csv' in the same folder as App.py"
        )
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip()
    return artifacts, df_raw


artifacts, df_raw = load_artifacts()
imputer, scaler, model, features = (
    artifacts["imputer"],
    artifacts["scaler"],
    artifacts["model"],
    artifacts["features"],
)
threshold = artifacts.get("threshold", 0.5)  # decision threshold chosen by cross-validation

# Sidebar - Brand, Developer Credit & Selection
with st.sidebar:
    # 1) Logo
    st.image(LOGO_URL, width="stretch")

    # 2) Developer credit (directly below the logo) - pure Streamlit, no HTML
    with st.container(border=True):
        st.caption("DEVELOPED BY")
        st.markdown("**Alom Shah**")
        st.caption("Junior Student, Class of 2028")
        st.markdown(":blue[**Parami University**]")

    st.divider()

    # 3) Settings
    st.title("Settings")
    countries = ["Custom Inputs"] + sorted(df_raw["Country"].unique().tolist())
    selected_country = st.selectbox("Auto-fill from Country:", countries)

# Main Header
st.title("🌍 Country Development Status Predictor")
st.caption(
    "Classify national economic status using WHO health metrics & socioeconomic indicators."
)

# Pre-fill defaults based on selected country
if selected_country != "Custom Inputs":
    country_df = df_raw[df_raw["Country"] == selected_country]
    defaults = {
        col: float(country_df[col].median())
        if col in country_df and pd.notnull(country_df[col].median())
        else 0.0
        for col in features
    }
else:
    defaults = {col: 0.0 for col in features}

# Feature Input Grid (2 Columns)
st.subheader(f"Input Indicators ({selected_country})")
input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        # the country is part of the key, so switching country refreshes the values
        input_data[feature] = st.number_input(
            label=feature,
            value=defaults[feature],
            key=f"in_{selected_country}_{feature}",
        )

st.divider()

# Prediction Output Section
if st.button("Run Classification", type="primary", width="stretch"):
    input_df = pd.DataFrame([input_data])
    scaled_data = scaler.transform(imputer.transform(input_df))

    prob_dev = model.predict_proba(scaled_data)[0][1]
    pred = int(prob_dev >= threshold)
    label = "Developed" if pred == 1 else "Developing"

    col_a, col_b = st.columns(2)
    with col_a:
        if pred == 1:
            st.success(f"### Result: **{label}**")
        else:
            st.warning(f"### Result: **{label}**")

    with col_b:
        st.metric(label="Developed Probability", value=f"{prob_dev:.1%}")
        st.progress(float(prob_dev))
        st.caption(f"Decision threshold: {threshold:.2f}")
