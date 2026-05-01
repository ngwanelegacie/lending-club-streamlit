"""
Setup script for the Lending Club Streamlit app.
Reads the cleaned CSV, trains models, and exports:
  - model/model.pkl       (trained Gradient Boosting classifier)
  - model/scaler.pkl      (StandardScaler fitted on training data)
  - model/metadata.pkl    (feature order, thresholds, dataset info)
  - data/*.csv            (pre-computed EDA summaries for the dashboard)

Usage:
    python setup_model.py --data path/to/lending_club_cleaned.csv

If --data is omitted, it looks for the file in the default project location.
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def find_data(cli_path):
    """Resolve the cleaned CSV path."""
    if cli_path and os.path.exists(cli_path):
        return cli_path
    # Default locations (relative to this script or the project root)
    candidates = [
        os.path.join(SCRIPT_DIR, "[C] lending_club_cleaned.csv"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "Lending & Credit Analysis",
                     "[C] lending_club_cleaned.csv"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "Lending & Credit Analysis",
                     "lending-credit-analysis", "data", "processed",
                     "lending_club_cleaned.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_and_prepare(path):
    """Load the cleaned CSV and prepare features / target."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # Target
    if "default" not in df.columns:
        raise ValueError("Column 'default' not found in dataset.")

    # ── Feature engineering ──
    df["income_to_loan"] = (df["annual_inc"] / df["loan_amnt"].clip(lower=1)).clip(upper=100)
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"].clip(lower=1)).clip(upper=5)
    df["high_risk_credit"] = (df["fico_range_low"] < 670).astype(int)
    df["high_dti"] = (df["dti"] > 30).astype(int)
    df["installment_burden"] = (df["installment"] * 12 / df["annual_inc"].clip(lower=1)).clip(upper=1)

    # ── Select numeric features ──
    # Encode sub_grade as ordinal (A1=1 … G5=35)
    if "sub_grade" in df.columns:
        grade_order = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
        grade_map = {sg: i + 1 for i, sg in enumerate(grade_order)}
        df["sub_grade_num"] = df["sub_grade"].map(grade_map)
    else:
        df["sub_grade_num"] = 0

    # Encode home_ownership
    ownership_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 0, "NONE": 0, "ANY": 0}
    if "home_ownership" in df.columns:
        df["home_ownership_num"] = df["home_ownership"].map(ownership_map).fillna(0).astype(int)
    else:
        df["home_ownership_num"] = 0

    feature_cols = [
        "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
        "emp_length", "fico_range_low", "revol_util", "open_acc", "total_acc",
        "sub_grade_num", "home_ownership_num", "delinq_2yrs", "inq_last_6mths",
        "pub_rec", "revol_bal", "total_rev_hi_lim", "mort_acc",
        "credit_history_months", "issue_year",
        "income_to_loan", "loan_to_income", "high_risk_credit", "high_dti",
        "installment_burden",
    ]

    # Only keep features that exist in the data
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    y = df["default"].values

    return df, X, y, feature_cols


def train_models(X_train, X_test, y_train, y_test):
    """Train three classifiers and return results + best model."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=50, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=50, subsample=0.8, random_state=42
        ),
    }

    results = []
    trained = {}

    for name, clf in models.items():
        print(f"  Training {name}...")
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({"model": name, "acc": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4)})
        trained[name] = clf
        print(f"    AUC: {auc:.4f}  |  Acc: {acc:.4f}  |  F1: {f1:.4f}")

    return scaler, trained, results


def export_eda(df):
    """Compute and save EDA summary CSVs."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Default rate by grade
    if "grade" in df.columns:
        g = df.groupby("grade")["default"].mean().reset_index()
        g.columns = ["grade", "default_rate"]
        g["default_rate"] = (g["default_rate"] * 100).round(1)
        g.to_csv(os.path.join(DATA_DIR, "grade_defaults.csv"), index=False)
        print("  Saved grade_defaults.csv")

    # 2. Default rate by purpose
    if "purpose" in df.columns:
        p = df.groupby("purpose")["default"].mean().reset_index()
        p.columns = ["purpose", "default_rate"]
        p["default_rate"] = (p["default_rate"] * 100).round(1)
        p.to_csv(os.path.join(DATA_DIR, "purpose_defaults.csv"), index=False)
        print("  Saved purpose_defaults.csv")

    # 3. Default rate by term
    if "term" in df.columns:
        t = df.groupby("term")["default"].mean().reset_index()
        t.columns = ["term", "default_rate"]
        t["term"] = t["term"].astype(int).astype(str) + " months"
        t["default_rate"] = (t["default_rate"] * 100).round(1)
        t.to_csv(os.path.join(DATA_DIR, "term_defaults.csv"), index=False)
        print("  Saved term_defaults.csv")

    # 4. Default rate by home ownership
    if "home_ownership" in df.columns:
        h = df[df["home_ownership"].isin(["RENT", "OWN", "MORTGAGE"])]
        h = h.groupby("home_ownership")["default"].mean().reset_index()
        h.columns = ["home_ownership", "default_rate"]
        h["default_rate"] = (h["default_rate"] * 100).round(1)
        h.to_csv(os.path.join(DATA_DIR, "ownership_defaults.csv"), index=False)
        print("  Saved ownership_defaults.csv")

    # 5. Default rate by year
    if "issue_year" in df.columns:
        y = df.groupby("issue_year")["default"].mean().reset_index()
        y.columns = ["year", "default_rate"]
        y["default_rate"] = (y["default_rate"] * 100).round(1)
        y.to_csv(os.path.join(DATA_DIR, "yearly_defaults.csv"), index=False)
        print("  Saved yearly_defaults.csv")


def export_model(scaler, trained, results, feature_cols):
    """Save model artifacts and results."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_model = trained["Gradient Boosting"]

    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "feature_order": feature_cols,
        "best_model": "Gradient Boosting",
        "n_features": len(feature_cols),
    }
    with open(os.path.join(MODEL_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"  Saved model.pkl, scaler.pkl, metadata.pkl to {MODEL_DIR}/")

    # Model comparison CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(DATA_DIR, "model_results.csv"), index=False)
    print("  Saved model_results.csv")

    # Feature importance
    fi = best_model.feature_importances_
    df_fi = pd.DataFrame({"feature": feature_cols, "importance": fi})
    df_fi = df_fi.sort_values("importance", ascending=False)
    df_fi.to_csv(os.path.join(DATA_DIR, "feature_importance.csv"), index=False)
    print("  Saved feature_importance.csv")


def main():
    parser = argparse.ArgumentParser(description="Export Lending Club model + EDA data for Streamlit app")
    parser.add_argument("--data", type=str, default=None, help="Path to lending_club_cleaned.csv")
    args = parser.parse_args()

    data_path = find_data(args.data)
    if not data_path:
        print("ERROR: Could not find lending_club_cleaned.csv")
        print("Pass the path explicitly:  python setup_model.py --data /path/to/file.csv")
        return

    # Load
    print("\n1. Loading and preparing data...")
    df, X, y, feature_cols = load_and_prepare(data_path)

    # EDA exports
    print("\n2. Exporting EDA summaries...")
    export_eda(df)

    # Train
    print("\n3. Training models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler, trained, results = train_models(X_train, X_test, y_train, y_test)

    # Export
    print("\n4. Exporting model artifacts...")
    os.makedirs(DATA_DIR, exist_ok=True)
    export_model(scaler, trained, results, feature_cols)

    print("\n✓ Setup complete! You can now run the Streamlit app:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
