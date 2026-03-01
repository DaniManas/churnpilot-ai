"""
ml/train.py
-----------
Train the churn prediction model.

Data source priority:
  1. Real IBM Telco Customer Churn dataset  →  place at data/Telco-Customer-Churn.csv
     Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
  2. Synthetic fallback  →  generated automatically if real data not found
     Mirrors the real dataset's column names, distributions, and known data quality issues.

Run once before starting the API:
    python ml/train.py

Produces:
    ml/artifacts/model.pkl            — sklearn Pipeline (preprocessing + RandomForest)
    ml/artifacts/model_metadata.json  — version, metrics, baseline feature distributions
    data/sample_batch.csv             — 20-row sample for testing /batch
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# ── Real dataset path ─────────────────────────────────────────────────────────
REAL_DATA_PATH = "data/Telco-Customer-Churn.csv"

# ── Feature groups (must match app/services/ml_service.py) ───────────────────
NUMERIC_FEATURES     = ["tenure", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
BINARY_FEATURES      = ["senior_citizen", "partner", "dependents",
                         "phone_service", "paperless_billing"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Valid categories — used by OrdinalEncoder and Pydantic schema validation
# These match the IBM Telco dataset exactly
CONTRACT_CATEGORIES         = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICE_CATEGORIES = ["No", "DSL", "Fiber optic"]
PAYMENT_METHOD_CATEGORIES   = ["Mailed check", "Bank transfer", "Credit card", "Electronic check"]


# ── 1a. Load REAL IBM Telco dataset ───────────────────────────────────────────

def load_real_data(path: str) -> pd.DataFrame:
    """
    Loads the IBM Telco Customer Churn dataset and cleans it.

    Known data quality issues in the real dataset:
      - TotalCharges: stored as string, contains empty strings for new customers
        (tenure=0 means they haven't been billed yet)
      - Churn: stored as "Yes"/"No" strings, not 0/1
      - Column names: mixed casing, need normalisation for the API
    """
    print(f"  Loading real data from {path}...")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Fix TotalCharges — empty string → NaN → fill with 0 (no charges yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_fixed = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if n_fixed > 0:
        print(f"  Fixed {n_fixed} empty TotalCharges values (new customers, filled with 0)")

    # Normalise column names to snake_case (matches API schema)
    df = df.rename(columns={
        "tenure":          "tenure",
        "MonthlyCharges":  "monthly_charges",
        "TotalCharges":    "total_charges",
        "Contract":        "contract",
        "InternetService": "internet_service",
        "PaymentMethod":   "payment_method",
        "SeniorCitizen":   "senior_citizen",
        "Partner":         "partner",
        "Dependents":      "dependents",
        "PhoneService":    "phone_service",
        "PaperlessBilling":"paperless_billing",
        "Churn":           "churn",
    })

    # Encode Yes/No binary columns → 1/0
    yes_no_cols = ["partner", "dependents", "phone_service", "paperless_billing"]
    for col in yes_no_cols:
        df[col] = (df[col] == "Yes").astype(int)

    # Encode target
    df["churn"] = (df["churn"] == "Yes").astype(int)

    # Standardise PaymentMethod labels (real dataset has " (automatic)" suffix)
    df["payment_method"] = df["payment_method"].str.replace(
        r" \(automatic\)", "", regex=True
    )

    # Drop unused columns (customerID, gender, and service add-ons)
    drop_cols = [c for c in df.columns if c not in ALL_FEATURES + ["churn"]]
    df = df.drop(columns=drop_cols, errors="ignore")

    print(f"  Clean shape: {df.shape}  |  Churn rate: {df['churn'].mean():.1%}")
    return df


# ── 1b. Synthetic fallback — mirrors IBM Telco dataset ───────────────────────

def generate_synthetic_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic data that mirrors the IBM Telco dataset:
    - Same column names and categories
    - Same approximate distributions and churn rate (~26.5%)
    - Same known data quality issue: new customers (tenure=0) have total_charges=0

    This lets the project run without needing to download the real dataset.
    To use real data: place Telco-Customer-Churn.csv in data/ and re-run.
    """
    rng = np.random.default_rng(random_state)

    # Contract type
    contract = rng.choice(CONTRACT_CATEGORIES, n_samples, p=[0.55, 0.21, 0.24])

    # Tenure — longer-contract customers stay longer
    tenure = np.where(
        contract == "Two year",   rng.integers(12, 73, n_samples),
        np.where(
        contract == "One year",   rng.integers(6,  61, n_samples),
                                  rng.integers(0,  37, n_samples)
        )
    )

    # Internet service
    internet_service = rng.choice(
        INTERNET_SERVICE_CATEGORIES, n_samples, p=[0.22, 0.34, 0.44]
    )

    # Monthly charges (Fiber optic is most expensive)
    monthly_charges = np.where(
        internet_service == "Fiber optic", rng.uniform(70, 120, n_samples),
        np.where(
        internet_service == "DSL",         rng.uniform(25,  70, n_samples),
                                           rng.uniform(18,  30, n_samples)
        )
    ).round(2)

    # Total charges — mirrors real dataset: tenure=0 customers have 0 charges
    total_charges = np.where(
        tenure == 0,
        0.0,
        (tenure * monthly_charges + rng.uniform(-30, 30, n_samples)).clip(0)
    ).round(2)

    # Payment method
    payment_method = rng.choice(
        PAYMENT_METHOD_CATEGORIES, n_samples, p=[0.23, 0.22, 0.21, 0.34]
    )

    # Binary features
    senior_citizen    = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner           = rng.choice([0, 1], n_samples, p=[0.52, 0.48])
    dependents        = rng.choice([0, 1], n_samples, p=[0.70, 0.30])
    phone_service     = rng.choice([0, 1], n_samples, p=[0.10, 0.90])
    paperless_billing = rng.choice([0, 1], n_samples, p=[0.41, 0.59])

    # Churn probability (mirrors known drivers in the real dataset)
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(contract == "Month-to-month",         0.35, 0.0)
    churn_prob += np.where(contract == "One year",               0.08, 0.0)
    churn_prob += np.where(internet_service == "Fiber optic",    0.10, 0.0)
    churn_prob += np.where(monthly_charges > 80,                 0.08, 0.0)
    churn_prob += np.where(tenure < 6,                           0.15, 0.0)
    churn_prob += np.where(tenure > 48,                         -0.12, 0.0)
    churn_prob += np.where(senior_citizen == 1,                  0.05, 0.0)
    churn_prob += np.where(payment_method == "Electronic check", 0.07, 0.0)
    churn_prob += np.where(paperless_billing == 1,               0.03, 0.0)
    churn_prob  = churn_prob.clip(0.02, 0.95)

    churn = (rng.uniform(0, 1, n_samples) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure":            tenure,
        "monthly_charges":   monthly_charges,
        "total_charges":     total_charges,
        "contract":          contract,
        "internet_service":  internet_service,
        "payment_method":    payment_method,
        "senior_citizen":    senior_citizen,
        "partner":           partner,
        "dependents":        dependents,
        "phone_service":     phone_service,
        "paperless_billing": paperless_billing,
        "churn":             churn,
    })


# ── 2. Build sklearn Pipeline ─────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Single Pipeline: preprocessing → RandomForest.

    Why a Pipeline?
    Packing preprocessing + model into one object prevents train/serve skew.
    ml_service.py calls pipeline.predict_proba(df) — no risk of forgetting
    to scale or encode before prediction.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(
                    categories=[
                        CONTRACT_CATEGORIES,
                        INTERNET_SERVICE_CATEGORIES,
                        PAYMENT_METHOD_CATEGORIES,
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ── 3. Baseline distributions for PSI drift detection ─────────────────────────

def compute_baseline_distributions(df: pd.DataFrame) -> dict:
    """
    Saves histogram bin counts per numeric feature from training data.
    /monitor uses these to compute PSI — detects when live data drifts
    away from the distribution the model was trained on.
    """
    distributions = {}
    for col in NUMERIC_FEATURES:
        counts, bin_edges = np.histogram(df[col], bins=10)
        distributions[col] = {
            "bin_edges": bin_edges.tolist(),
            "counts":    counts.tolist(),
            "mean":      float(df[col].mean()),
            "std":       float(df[col].std()),
            "min":       float(df[col].min()),
            "max":       float(df[col].max()),
        }
    return distributions


# ── 4. Train, evaluate, save ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ML AI Platform — Model Training")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    if os.path.exists(REAL_DATA_PATH):
        df = load_real_data(REAL_DATA_PATH)
        data_source = "IBM Telco Customer Churn (real)"
    else:
        print(f"  Real data not found at {REAL_DATA_PATH}.")
        print("  Using synthetic data that mirrors the IBM Telco dataset.")
        print("  To use real data: download from kaggle.com/datasets/blastchar/telco-customer-churn")
        print("  and place at data/Telco-Customer-Churn.csv, then re-run this script.\n")
        df = generate_synthetic_data(n_samples=7043)
        data_source = "Synthetic (mirrors IBM Telco structure)"

    print(f"\n  Rows: {len(df):,}  |  Churn rate: {df['churn'].mean():.1%}")

    # Split
    X = df[ALL_FEATURES]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Train
    print("\n[2/4] Training RandomForest pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\n[3/4] Evaluating...")
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    print(f"\n  ROC-AUC: {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Save
    print("[4/4] Saving artifacts...")
    os.makedirs("ml/artifacts", exist_ok=True)

    with open("ml/artifacts/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    metadata = {
        "model_version":        "1.0.0",
        "trained_at":           datetime.utcnow().isoformat(),
        "data_source":          data_source,
        "algorithm":            "RandomForestClassifier",
        "n_estimators":         200,
        "features":             ALL_FEATURES,
        "numeric_features":     NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features":      BINARY_FEATURES,
        "target":               "churn",
        "roc_auc_score":        round(auc, 4),
        "train_samples":        len(X_train),
        "test_samples":         len(X_test),
        "churn_rate_train":     round(float(y_train.mean()), 4),
        "baseline_distributions": compute_baseline_distributions(X_train),
    }

    with open("ml/artifacts/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Sample batch CSV for testing /batch endpoint
    os.makedirs("data", exist_ok=True)
    X_test.head(20).to_csv("data/sample_batch.csv", index=False)

    print("\n  ✓  ml/artifacts/model.pkl")
    print("  ✓  ml/artifacts/model_metadata.json")
    print(f"     data_source:  {data_source}")
    print(f"     roc_auc:      {auc:.4f}")
    print("  ✓  data/sample_batch.csv")
    print("\n  Training complete.\n")


if __name__ == "__main__":
    main()
