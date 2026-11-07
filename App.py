import os
import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Country Development Predictor",
    page_icon="🌍",
    layout="wide"
)

LOGO_URL = "https://i.postimg.cc/s2NnhybN/Whats-App-Image-2025-11-20-at-11-46-01-57c0431d.jpg"

@st.cache_resource
def load_artifacts():
    script_dir = os.path.dirname(__file__)
    artifacts = joblib.load(os.path.join(script_dir, 'model.pkl'))
    df_raw = pd.read_csv(os.path.join(script_dir, 'Life Expectancy Data.csv'))
    df_raw.columns = df_raw.columns.str.strip()
    return artifacts, df_raw

artifacts, df_raw = load_artifacts()
imputer, scaler, model, features = (
    artifacts['imputer'], 
    artifacts['scaler'], 
    artifacts['model'], 
    artifacts['features']
)

# Sidebar - Brand & Selection
with st.sidebar:
    st.image(LOGO_URL, use_container_width=True)
    st.title("Settings")
    countries = ["Custom Inputs"] + sorted(df_raw['Country'].unique().tolist())
    selected_country = st.selectbox("Auto-fill from Country:", countries)

# Main Header
st.title("🌍 Country Development Status Predictor")
st.caption("Classify national economic status using WHO health metrics & socioeconomic indicators.")

# Pre-fill defaults based on selected country
defaults = {}
if selected_country != "Custom Inputs":
    country_df = df_raw[df_raw['Country'] == selected_country]
    defaults = {col: float(country_df[col].median()) if col in country_df and pd.notnull(country_df[col].median()) else 0.0 for col in features}
else:
    defaults = {col: 0.0 for col in features}

# Feature Input Grid (2 Columns)
st.subheader(f"Input Indicators ({selected_country})")
input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        input_data[feature] = st.number_input(
            label=feature, 
            value=defaults[feature], 
            key=f"in_{feature}"
        )

st.divider()

# Prediction Output Section
if st.button("Run Classification", type="primary", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    scaled_data = scaler.transform(imputer.transform(input_df))
    
    pred = model.predict(scaled_data)[0]
    prob_dev = model.predict_proba(scaled_data)[0][1]
    
    label = "Developed" if pred == 1 else "Developing"
    
    col_a, col_b = st.columns(2)
    with col_a:
        if pred == 1:
            st.success(f"### Result: **{label}**")
        else:
            st.warning(f"### Result: **{label}**")
            
    with col_b:
        st.metric(label="Developed Probability", value=f"{prob_dev:.1%}")
        st.progress(prob_dev)