import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---- Load CSV relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data.csv")
data = pd.read_csv(csv_path)

# ---- Configuration
target_col = "Calories_Burned"
categorical_cols = ["Gender", "Workout_Type", "Experience_Level"]  # categorical columns
feature_cols = [c for c in data.columns if c != target_col]

# ---- Train/test split
X = data[feature_cols]
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Preprocessor for categorical and numeric features
numeric_cols = [c for c in feature_cols if c not in categorical_cols]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ---- Define models
models = {
    "Linear Regression": Pipeline([
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ]),
    "Polynomial Degree 2": Pipeline([
        ("preprocess", preprocessor),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("regressor", LinearRegression())
    ]),
    "Polynomial Degree 3": Pipeline([
        ("preprocess", preprocessor),
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("regressor", LinearRegression())
    ])
}

# ---- Train and evaluate models
results = []
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = sqrt(mean_squared_error(y_test, preds))
    
    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })
    
    print(f"\n{name}")
    print(f"R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    if best_model is None or r2 > best_model["R2"]:
        best_model = {"name": name, "model": model, "R2": r2, "MAE": mae, "RMSE": rmse}

# ---- Print best model
print("\n----------------------------------")
print("Best Model:", best_model["name"])
print(f"R² = {best_model['R2']:.4f}, MAE = {best_model['MAE']:.4f}, RMSE = {best_model['RMSE']:.4f}")
print("----------------------------------")

# ---- save all trained models
script_dir = os.path.dirname(os.path.abspath(__file__))
for name, model in models.items():
    # create a safe filename
    filename = name.lower().replace(" ", "_") + ".pkl"  # e.g., linear_regression.pkl
    path = os.path.join(script_dir, filename)
    joblib.dump(model, path)
    print(f"Saved {name} to: {path}")