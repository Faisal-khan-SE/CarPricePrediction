import re
import numpy as np
import pandas as pd
import joblib
# import sweetviz as st
# import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
cars_data = pd.read_csv('used_cars.csv')

# report = st.analyze(cars_data)
# report.show_html('DataInfo.html')

# ---------------------------------------------------------------------------
# 2. CLEAN NUMERIC COLUMNS THAT ARE STORED AS TEXT
#    price   -> "$10,300"   -> 10300.0
#    milage  -> "51,000 mi." -> 51000.0
# ---------------------------------------------------------------------------
cars_data["price"] = pd.to_numeric(
    cars_data["price"].astype(str).str.replace(r"[\$,]", "", regex=True),
    errors="coerce"
)
cars_data["milage"] = pd.to_numeric(
    cars_data["milage"].astype(str).str.replace(r"[, ]|mi\.", "", regex=True),
    errors="coerce"
)

# drop rows where the target itself is missing/unparseable
cars_data = cars_data.dropna(subset=["price"])

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING FROM THE "engine" FREE-TEXT COLUMN
#    e.g. "300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability"
#    -> horsepower, engine size in liters, cylinder count
# ---------------------------------------------------------------------------
def extract_hp(text):
    match = re.search(r"([\d.]+)\s*HP", str(text))
    return float(match.group(1)) if match else np.nan

def extract_liters(text):
    match = re.search(r"([\d.]+)\s*L\b", str(text))
    return float(match.group(1)) if match else np.nan

def extract_cylinders(text):
    match = re.search(r"(\d+)\s*Cylinder", str(text))
    return float(match.group(1)) if match else np.nan

cars_data["horsepower"] = cars_data["engine"].apply(extract_hp)
cars_data["engine_liters"] = cars_data["engine"].apply(extract_liters)
cars_data["cylinders"] = cars_data["engine"].apply(extract_cylinders)

# ---------------------------------------------------------------------------
# 4. FILL MISSING VALUES
#    - categorical / text columns with missing values -> "Unknown"
#    - engineered numeric engine features -> median (keeps distribution sane)
# ---------------------------------------------------------------------------
for col in ["fuel_type", "accident", "clean_title"]:
    cars_data[col] = cars_data[col].fillna("Unknown")

# a couple of datasets use "–" or "not supported" as a disguised missing value
cars_data["fuel_type"] = cars_data["fuel_type"].replace(
    {"–": "Unknown", "not supported": "Unknown"}
)

# NOTE: these medians are computed on the FULL cleaned dataset and are also
# dumped to disk (medians.joblib) so the Streamlit app can fill missing/
# unparsed engine specs on user input exactly the way training did.
engine_medians = {}
for col in ["horsepower", "engine_liters", "cylinders"]:
    engine_medians[col] = cars_data[col].median()
    cars_data[col] = cars_data[col].fillna(engine_medians[col])

# ---------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X = cars_data.drop(columns=["price", "engine"])
Y = cars_data["price"]

cat_cols = [
    "brand", "model", "fuel_type", "transmission",
    "ext_col", "int_col", "accident", "clean_title"
]
num_cols = ["model_year", "milage", "horsepower", "engine_liters", "cylinders"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------------------------
# 6. TRANSFORMERS (same structure as original: one without scaling for
#    tree-based models, one with scaling for distance/linear based models)
# ---------------------------------------------------------------------------
def transformation():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder="passthrough"
    )

def scalling_transformation():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

# ---------------------------------------------------------------------------
# 7. OPTUNA SEARCH -- ALREADY RUN, WINNER LOCKED IN BELOW.
#    Kept here (commented) for reference / re-tuning later.
# ---------------------------------------------------------------------------
# def objective(trail):
#     algorithms = trail.suggest_categorical(
#         "algorithm",
#         ["SVR", "XGBRegressor", "KNeighborsRegressor",
#          "GradientBoostingRegressor", "RandomForestRegressor",
#          "Ridge", "LinearRegression"]
#     )
#
#     if algorithms == "SVR":
#         preprocessor = scalling_transformation()
#         model = SVR(
#             C=trail.suggest_float("svr_c", 0.01, 0.5),
#             epsilon=trail.suggest_float("svr_epsilon", 0.01, 1.0),
#             kernel=trail.suggest_categorical("svr_kernel", ["linear", "rbf", "poly"]),
#             gamma=trail.suggest_categorical("svr_gamma", ["scale", "auto"])
#         )
#     elif algorithms == "XGBRegressor":
#         preprocessor = transformation()
#         model = XGBRegressor(
#             n_estimators=trail.suggest_int("xgb_n_estimators", 50, 1000),
#             max_depth=trail.suggest_int("xgb_max_depth", 3, 12),
#             learning_rate=trail.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
#             subsample=trail.suggest_float("xgb_subsample", 0.6, 1.0),
#             colsample_bytree=trail.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
#             min_child_weight=trail.suggest_int("xgb_min_child_weight", 1, 10),
#             reg_alpha=trail.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
#             reg_lambda=trail.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),
#             random_state=42,
#             n_jobs=-1
#         )
#
#     elif algorithms == "KNeighborsRegressor":
#         preprocessor = scalling_transformation()
#         model = KNeighborsRegressor(
#             n_neighbors=trail.suggest_int("knn_n_neighbors", 2, 30),
#             weights=trail.suggest_categorical("knn_weights", ["uniform", "distance"]),
#             p=trail.suggest_int("knn_p", 1, 2)
#         )
#
#     elif algorithms == "GradientBoostingRegressor":
#         preprocessor = transformation()
#         model = GradientBoostingRegressor(
#             n_estimators=trail.suggest_int("gbr_n_estimators", 100, 1000),
#             learning_rate=trail.suggest_float("gbr_learning_rate", 0.01, 0.3, log=True),
#             max_depth=trail.suggest_int("gbr_max_depth", 2, 16),
#             min_samples_split=trail.suggest_int("gbr_min_samples_split", 2, 20),
#             min_samples_leaf=trail.suggest_int("gbr_min_samples_leaf", 1, 20),
#             subsample=trail.suggest_float("gbr_subsample", 0.6, 1.0),
#             loss=trail.suggest_categorical(
#                 "gbr_loss",
#                 ["squared_error", "huber", "absolute_error"]
#             ),
#             random_state=42
#         )
#
#     elif algorithms == "RandomForestRegressor":
#         preprocessor = transformation()
#         model = RandomForestRegressor(
#             n_estimators=trail.suggest_int("rf_n_estimators", 100, 1000),
#             max_depth=trail.suggest_int("rf_max_depth", 3, 30),
#             min_samples_split=trail.suggest_int("rf_min_samples_split", 2, 20),
#             min_samples_leaf=trail.suggest_int("rf_min_samples_leaf", 1, 10),
#             max_features=trail.suggest_categorical(
#                 "rf_max_features", ["sqrt", "log2", 1.0]
#             ),
#             bootstrap=trail.suggest_categorical(
#                 "rf_bootstrap", [True, False]
#             ),
#             random_state=42,
#             n_jobs=-1
#         )
#
#     elif algorithms == "Ridge":
#         preprocessor = scalling_transformation()
#         model = Ridge(
#             alpha=trail.suggest_float("ridge_alpha", 1e-4, 100.0, log=True),
#             solver=trail.suggest_categorical(
#                 "ridge_solver",
#                 ["auto", "lsqr"]
#             )
#         )
#
#     else:
#         preprocessor = scalling_transformation()
#         model = LinearRegression(
#             fit_intercept=trail.suggest_categorical(
#                 "lr_fit_intercept", [True, False]
#             )
#         )
#
#     pipe = Pipeline(
#         [
#             ("preprocessor", preprocessor),
#             ("model", model)
#         ]
#     )
#
#     return cross_val_score(
#         pipe,
#         X_train,
#         y_train,
#         cv=5,
#         scoring="r2"
#     ).mean()
#
# if __name__ == "__main__":
#     study = optuna.create_study(direction="maximize")
#     optuna.logging.set_verbosity(optuna.logging.WARNING)
#     study.optimize(objective, n_trials=50)
#     print(study.best_value)
#     print(study.best_params)

# ---------------------------------------------------------------------------
# 8. FINAL MODEL -- best Optuna trial (r2 = 0.8081) was XGBRegressor with:
#    {'xgb_n_estimators': 917, 'xgb_max_depth': 11,
#     'xgb_learning_rate': 0.010130310517439501,
#     'xgb_subsample': 0.9507425599231973,
#     'xgb_colsample_bytree': 0.7585066179622171,
#     'xgb_min_child_weight': 1,
#     'xgb_reg_alpha': 2.8282931989055627e-07,
#     'xgb_reg_lambda': 4.727147966568546e-08}
#    XGBRegressor uses the non-scaling transformer (transformation()).
# ---------------------------------------------------------------------------
best_params = {
    "n_estimators": 917,
    "max_depth": 11,
    "learning_rate": 0.010130310517439501,
    "subsample": 0.9507425599231973,
    "colsample_bytree": 0.7585066179622171,
    "min_child_weight": 1,
    "reg_alpha": 2.8282931989055627e-07,
    "reg_lambda": 4.727147966568546e-08,
    "random_state": 42,
    "n_jobs": -1
}

final_model = Pipeline(
    [
        ("preprocessor", transformation()),
        ("model", XGBRegressor(**best_params))
    ]
)

final_model.fit(X_train, y_train)

train_r2 = r2_score(y_train, final_model.predict(X_train))
test_pred = final_model.predict(X_test)
test_r2 = r2_score(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"Train R2: {train_r2:.4f}")
print(f"Test  R2: {test_r2:.4f}")
print(f"Test  MAE: {test_mae:,.2f}")

# ---------------------------------------------------------------------------
# 9. PREDICTION HELPER
#    The pipeline already bundles preprocessing + model, so "predicting" is
#    just pipeline.predict(df). This wrapper exists mainly so Streamlit can
#    build a one-row DataFrame from form inputs and get a single float back.
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

# ---------------------------------------------------------------------------
# 10. DUMP MODEL + METADATA FOR STREAMLIT
#     - car_price_model.joblib : the full fitted pipeline (preprocess+model)
#     - dropdown_options.joblib: unique values per categorical column, so the
#       Streamlit UI only offers choices the model was actually trained on
#     - medians.joblib         : medians used to fill engine specs, so the
#       app can default/impute the same way training did
#     - X_train_columns.joblib : exact column order/names X was trained on
# ---------------------------------------------------------------------------
joblib.dump(final_model, "car_price_model.joblib")

dropdown_options = {col: sorted(cars_data[col].dropna().unique().tolist())
                     for col in cat_cols}
joblib.dump(dropdown_options, "dropdown_options.joblib")

joblib.dump(engine_medians, "medians.joblib")

joblib.dump(list(X.columns), "X_train_columns.joblib")

numeric_ranges = {
    "model_year": (int(cars_data["model_year"].min()), int(cars_data["model_year"].max())),
    "milage": (float(cars_data["milage"].min()), float(cars_data["milage"].max())),
    "horsepower": (float(cars_data["horsepower"].min()), float(cars_data["horsepower"].max())),
    "engine_liters": (float(cars_data["engine_liters"].min()), float(cars_data["engine_liters"].max())),
    "cylinders": (float(cars_data["cylinders"].min()), float(cars_data["cylinders"].max())),
}
joblib.dump(numeric_ranges, "numeric_ranges.joblib")

print("Saved: car_price_model.joblib, dropdown_options.joblib, "
      "medians.joblib, X_train_columns.joblib, numeric_ranges.joblib")