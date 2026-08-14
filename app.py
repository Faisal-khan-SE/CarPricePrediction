import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

# ---------------------------------------------------------------------------
# LOAD MODEL + METADATA (dumped by pricePrediction.py)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("car_price_model.joblib")
    dropdown_options = joblib.load("dropdown_options.joblib")
    medians = joblib.load("medians.joblib")
    numeric_ranges = joblib.load("numeric_ranges.joblib")
    return model, dropdown_options, medians, numeric_ranges

try:
    model, dropdown_options, medians, numeric_ranges = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python pricePrediction.py` first "
        "so it can create car_price_model.joblib and the other .joblib files "
        "in this same folder."
    )
    st.stop()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("🚗 Used Car Price Predictor")
st.write(
    "Fill in the details of the car below and get an estimated resale price. "
    "The model is an XGBoost regressor tuned with Optuna (test R² ≈ 0.81)."
)

st.divider()

# ---------------------------------------------------------------------------
# FORM
# ---------------------------------------------------------------------------
with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", dropdown_options["brand"])
        model_name = st.selectbox("Model", dropdown_options["model"])
        model_year = st.number_input(
            "Model Year",
            min_value=numeric_ranges["model_year"][0],
            max_value=numeric_ranges["model_year"][1],
            value=numeric_ranges["model_year"][1],
            step=1,
        )
        milage = st.number_input(
            "Mileage (mi.)",
            min_value=0.0,
            max_value=float(numeric_ranges["milage"][1]),
            value=30000.0,
            step=1000.0,
        )
        fuel_type = st.selectbox("Fuel Type", dropdown_options["fuel_type"])
        transmission = st.selectbox("Transmission", dropdown_options["transmission"])

    with col2:
        ext_col = st.selectbox("Exterior Color", dropdown_options["ext_col"])
        int_col = st.selectbox("Interior Color", dropdown_options["int_col"])
        accident = st.selectbox("Accident History", dropdown_options["accident"])
        clean_title = st.selectbox("Clean Title", dropdown_options["clean_title"])

    st.markdown("**Engine specs** (leave defaults if unknown)")
    col3, col4, col5 = st.columns(3)
    with col3:
        horsepower = st.number_input(
            "Horsepower",
            min_value=0.0,
            max_value=float(numeric_ranges["horsepower"][1]),
            value=float(medians["horsepower"]),
        )
    with col4:
        engine_liters = st.number_input(
            "Engine Size (L)",
            min_value=0.0,
            max_value=float(numeric_ranges["engine_liters"][1]),
            value=float(medians["engine_liters"]),
            step=0.1,
        )
    with col5:
        cylinders = st.number_input(
            "Cylinders",
            min_value=0.0,
            max_value=float(numeric_ranges["cylinders"][1]),
            value=float(medians["cylinders"]),
            step=1.0,
        )

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def predict_price(model, brand, model_name, model_year, milage, fuel_type,
                   transmission, ext_col, int_col, accident, clean_title,
                   horsepower, engine_liters, cylinders):
    row = pd.DataFrame([{
        "brand": brand,
        "model": model_name,
        "model_year": model_year,
        "milage": milage,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "clean_title": clean_title,
        "horsepower": horsepower,
        "engine_liters": engine_liters,
        "cylinders": cylinders,
    }])
    return float(model.predict(row)[0])

if submitted:
    predicted_price = predict_price(
        model, brand, model_name, model_year, milage, fuel_type,
        transmission, ext_col, int_col, accident, clean_title,
        horsepower, engine_liters, cylinders
    )
    st.divider()
    st.metric("Estimated Price", f"${predicted_price:,.2f}")
    st.caption(
        "This is a model estimate based on historical listings, not an "
        "appraisal. Actual sale price can vary."
    )