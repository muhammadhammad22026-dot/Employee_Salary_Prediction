"""
Employee Salary Prediction - Model Training and Honest Evaluation

This script:
1. Loads the cleaned employee dataset.
2. Builds a preprocessing + regression pipeline.
3. Compares several algorithms using 5-fold cross-validation.
4. Compares the best real model with a simple average-salary baseline.
5. Saves a model only if it beats the baseline on cross-validation.

Important:
A machine-learning model cannot learn reliable salary patterns if the
input features do not contain useful information about Salary.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Cleaned_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "employee_salary_model.joblib"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place Cleaned_data.csv inside data/processed/."
    )


# ---------------------------------------------------------
# 2. LOAD AND CHECK DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Remove accidental spaces around column names.
df.columns = df.columns.str.strip()

required_columns = [
    "Age",
    "Status",
    "Join_Date",
    "Salary",
    "Performance_Score",
    "Remote_Work",
    "Department",
    "Region",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Join_Date"] = pd.to_datetime(df["Join_Date"], errors="coerce")

# Salary is the target, so rows without a known salary cannot train the model.
df = df.dropna(subset=["Salary", "Join_Date"]).copy()

print("\n========== DATASET CHECK ==========")
print(f"Rows available: {len(df)}")
print(f"Missing Age values: {df['Age'].isna().sum()}")
print(f"Average salary: {df['Salary'].mean():,.2f}")
print(f"Salary standard deviation: {df['Salary'].std():,.2f}")

# Tenure changes with time. This calculation uses the date the script runs.
reference_date = pd.Timestamp.today().normalize()
df["Tenure_year"] = (
    (reference_date - df["Join_Date"]).dt.days / 365.25
).clip(lower=0)

# Show simple correlations as a diagnostic, not as proof of causation.
print("\n========== NUMERIC CORRELATIONS WITH SALARY ==========")
print(
    df[["Age", "Tenure_year", "Salary"]]
    .corr(numeric_only=True)["Salary"]
    .sort_values(ascending=False)
)


# ---------------------------------------------------------
# 3. FEATURES AND TARGET
# ---------------------------------------------------------

numeric_features = ["Age", "Tenure_year"]

categorical_features = [
    "Performance_Score",
    "Status",
    "Department",
    "Region",
    "Remote_Work",
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = df["Salary"].copy()


# ---------------------------------------------------------
# 4. PREPROCESSING
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# ---------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ---------------------------------------------------------
# 6. COMPARE MODELS USING CROSS-VALIDATION
# ---------------------------------------------------------

models = {
    "Dummy Baseline": DummyRegressor(strategy="mean"),
    "Ridge Regression": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=5,
        loss="huber",
        random_state=42,
    ),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

for model_name, regressor in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    mean_mae = -scores.mean()
    cv_results[model_name] = {
        "mae": mean_mae,
        "pipeline": pipeline,
    }

    print(f"{model_name}: CV MAE = {mean_mae:,.2f}")


# ---------------------------------------------------------
# 7. COMPARE BEST REAL MODEL WITH BASELINE
# ---------------------------------------------------------

baseline_mae = cv_results["Dummy Baseline"]["mae"]

real_model_results = {
    name: result
    for name, result in cv_results.items()
    if name != "Dummy Baseline"
}

best_name = min(
    real_model_results,
    key=lambda name: real_model_results[name]["mae"],
)
best_result = real_model_results[best_name]
best_cv_mae = best_result["mae"]

print("\n========== MODEL SELECTION ==========")
print(f"Baseline CV MAE: {baseline_mae:,.2f}")
print(f"Best real model: {best_name}")
print(f"Best real model CV MAE: {best_cv_mae:,.2f}")


# ---------------------------------------------------------
# 8. FINAL TEST-SET EVALUATION
# ---------------------------------------------------------

best_pipeline = best_result["pipeline"]
best_pipeline.fit(X_train, y_train)

test_predictions = best_pipeline.predict(X_test)

test_r2 = r2_score(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5

baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", DummyRegressor(strategy="mean")),
    ]
)
baseline_pipeline.fit(X_train, y_train)
baseline_predictions = baseline_pipeline.predict(X_test)
baseline_test_mae = mean_absolute_error(y_test, baseline_predictions)

print("\n========== FINAL TEST PERFORMANCE ==========")
print(f"Selected model: {best_name}")
print(f"R2 Score: {test_r2:.4f}")
print(f"Mean Absolute Error: {test_mae:,.2f}")
print(f"RMSE: {test_rmse:,.2f}")
print(f"Baseline Test MAE: {baseline_test_mae:,.2f}")


# ---------------------------------------------------------
# 9. SAVE ONLY IF THE MODEL BEATS THE BASELINE
# ---------------------------------------------------------

# Require at least a 2% CV MAE improvement to avoid saving a model
# that only beats the baseline by a negligible amount.
required_improvement = 0.02
beats_baseline = best_cv_mae < baseline_mae * (1 - required_improvement)

if beats_baseline:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nModel saved successfully: {MODEL_PATH}")
else:
    print("\nMODEL NOT SAVED.")
    print(
        "The available features did not beat the average-salary baseline "
        "by at least 2% in cross-validation."
    )
    print(
        "Do not use this model for real salary decisions. Review the dataset "
        "and collect meaningful salary-related features first."
    )


# ---------------------------------------------------------
# 10. EXAMPLE PREDICTION (ONLY IF A MODEL WAS SAVED)
# ---------------------------------------------------------

if beats_baseline:
    new_employee = pd.DataFrame(
        {
            "Age": [30],
            "Tenure_year": [2.5],
            "Performance_Score": ["Excellent"],
            "Status": ["Active"],
            "Department": ["DevOps"],
            "Region": ["California"],
            "Remote_Work": [True],
        }
    )

    predicted_salary = best_pipeline.predict(new_employee)[0]
    print(f"\nExample predicted salary: {predicted_salary:,.2f}")
