"""P1 steel quality regression pipeline (see README for the step list).

Data files (data/raw/) are already min-max normalised by the course and come as a
separate train and test CSV. Leakage rules: every fitted step (imputer, encoder,
scaler, outlier fences) is fitted on the training split only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

SEED = 42
TARGET = "output"


def data_report(df: pd.DataFrame) -> dict:
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1]),
            "duplicates": int(df.duplicated().sum()),
            "missing_total": int(df.isna().sum().sum()),
            "non_numeric_columns": df.select_dtypes(exclude="number").columns.tolist()}


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def iqr_bounds(train: pd.DataFrame, columns, k: float = 3.0) -> dict:
    """k*IQR fences computed on training data only."""
    bounds = {}
    for c in columns:
        q1, q3 = train[c].quantile([0.25, 0.75])
        bounds[c] = (q1 - k * (q3 - q1), q3 + k * (q3 - q1))
    return bounds


def clip_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        out[c] = out[c].clip(lo, hi)
    return out


def split_train_val(df: pd.DataFrame, target: str = TARGET, val_size: float = 0.2, seed: int = SEED):
    X, y = df.drop(columns=[target]), df[target]
    return train_test_split(X, y, test_size=val_size, random_state=seed)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def build_models(X: pd.DataFrame) -> dict:
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                              random_state=SEED, n_jobs=-1),
        "SVR": SVR(C=1.0, epsilon=0.02),
    }
    return {n: Pipeline([("prep", build_preprocessor(X)), ("model", m)]) for n, m in models.items()}


def regression_metrics(y_true, y_pred) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def cross_validate_models(models: dict, X, y, folds: int = 5, seed: int = SEED) -> pd.DataFrame:
    cv = KFold(folds, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X, y, cv=cv, scoring={
            "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"})
        rows.append({"model": name,
                     "cv_RMSE_mean": -s["test_RMSE"].mean(), "cv_RMSE_std": s["test_RMSE"].std(),
                     "cv_MAE_mean": -s["test_MAE"].mean(), "cv_MAE_std": s["test_MAE"].std(),
                     "cv_R2_mean": s["test_R2"].mean(), "cv_R2_std": s["test_R2"].std()})
    return pd.DataFrame(rows)


def run(train_csv, test_csv, out_dir="results") -> pd.DataFrame:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    raw_train, raw_test = pd.read_csv(train_csv), pd.read_csv(test_csv)
    report = {"train_raw": data_report(raw_train), "test_raw": data_report(raw_test)}
    train, test = remove_duplicates(raw_train), remove_duplicates(raw_test)
    train, test = train.dropna(subset=[TARGET]), test.dropna(subset=[TARGET])
    (out / "tables" / "data_report.json").write_text(json.dumps(report, indent=2))

    # Step: histograms and input/output correlations (training data only)
    train.hist(bins=30, figsize=(15, 11))
    plt.tight_layout(); plt.savefig(out / "figures" / "histograms.png", dpi=120); plt.close()
    corr = train.corr(numeric_only=True)
    corr.to_csv(out / "tables" / "correlations.csv")
    corr[TARGET].drop(TARGET).sort_values().plot.barh(figsize=(6, 7), title="Correlation with output")
    plt.tight_layout(); plt.savefig(out / "figures" / "correlation_with_output.png", dpi=120); plt.close()

    X_tr, X_val, y_tr, y_val = split_train_val(train)
    X_te, y_te = test.drop(columns=[TARGET]), test[TARGET]
    bounds = iqr_bounds(X_tr, X_tr.select_dtypes(include="number").columns)
    X_tr, X_val, X_te = (clip_outliers(x, bounds) for x in (X_tr, X_val, X_te))

    models = build_models(X_tr)
    table = cross_validate_models(models, X_tr, y_tr)
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        rows.append({"model": name,
                     **{f"val_{k}": v for k, v in regression_metrics(y_val, m.predict(X_val)).items()},
                     **{f"test_{k}": v for k, v in regression_metrics(y_te, m.predict(X_te)).items()}})
    table = table.merge(pd.DataFrame(rows), on="model")
    table.to_csv(out / "tables" / "model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m in zip(axes, ["RMSE", "MAE", "R2"]):
        ax.bar(table["model"], table[f"cv_{m}_mean"], yerr=table[f"cv_{m}_std"], capsize=4)
        ax.set_title(f"{m}: 5-fold CV mean ± std")
    plt.tight_layout(); plt.savefig(out / "figures" / "model_comparison.png", dpi=120); plt.close()
    return table
