from __future__ import annotations
from pathlib import Path
import json
import yaml
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state
import joblib

# XGBoost opcional
try:
    from xgboost import XGBClassifier  # type: ignore

    XGB_OK = True
except Exception:
    XGB_OK = False
    warnings.filterwarnings("ignore", "xgboost")

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(
    (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
)
PROC_DIR = ROOT / PARAMS["data"]["paths"]["processed_dir"]
MODELS_DIR = ROOT / "models"
PIPE_DIR = MODELS_DIR / "model_pipelines"
TRAINED_DIR = MODELS_DIR / "trained_models"
PIPE_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ["match_id", "date", "season", "matchweek", "home_team", "away_team", "split"]
TARGET = "y"


def load_split(name: str) -> pd.DataFrame:
    p = PROC_DIR / f"features_{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Falta {p}. Corre make_features primero.")
    return pd.read_parquet(p)


def select_columns(df: pd.DataFrame):
    y = df[TARGET].astype("category")
    leak_cols = {"y", "ftr", "home_goals", "away_goals", "home_points", "away_points"}
    id_cols = set(ID_COLS)
    feat = [c for c in df.columns if c not in leak_cols | id_cols]
    num_cols = [c for c in feat if (df[c].dtype.kind in "fcbi") and c != "bookmaker"]
    cat_cols = [c for c in feat if c == "bookmaker" and c in df.columns]
    return df[feat], y, num_cols, cat_cols


def make_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    )
    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
        )
    return ColumnTransformer([("num", num_pipe, num_cols)])


class LabelEncodedEstimator(BaseEstimator, ClassifierMixin):
    """Codifica y en enteros para estimadores como XGB que los requieren."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.le_ = None
        self.classes_ = None
        self.est_ = None

    def fit(self, X, y):
        y = pd.Series(y).astype(str)
        self.le_ = LabelEncoder().fit(y)
        y_enc = self.le_.transform(y)
        self.est_ = clone(self.base_estimator)
        self.est_.fit(X, y_enc)
        self.classes_ = self.le_.classes_.tolist()
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def predict(self, X):
        pred_enc = self.est_.predict(X)
        return self.le_.inverse_transform(pred_enc)


def metrics(
    y_true: pd.Series, proba: np.ndarray, class_order: list[str]
) -> Dict[str, Any]:
    y_true = y_true.astype(str)
    ll = log_loss(y_true, proba, labels=class_order)
    briers = [
        brier_score_loss((y_true == cls).astype(int), proba[:, i])
        for i, cls in enumerate(class_order)
    ]
    pred = np.array([class_order[i] for i in proba.argmax(axis=1)])
    acc = accuracy_score(y_true, pred)
    return {
        "log_loss": float(ll),
        "brier": float(np.mean(briers)),
        "accuracy": float(acc),
    }


def calibrate(est, method="isotonic", cv=3):
    return CalibratedClassifierCV(est, method=method, cv=cv)


def tune_logreg(C_grid=(0.5, 1.0, 2.0), calibration="isotonic", random_state=7):
    print("\n=== Tuning LogisticRegression ===")
    df_train, df_valid, df_test = (
        load_split("train"),
        load_split("valid"),
        load_split("test"),
    )
    Xtr, ytr, num_cols, cat_cols = select_columns(df_train)
    Xva, yva, _, _ = select_columns(df_valid)
    Xte, yte, _, _ = select_columns(df_test)
    pre = make_preprocess(num_cols, cat_cols)
    results = []

    for C in C_grid:
        base = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", C=C, max_iter=300
        )
        pipe = Pipeline(
            [("pre", pre), ("clf", calibrate(base, method=calibration, cv=3))]
        )
        pipe.fit(Xtr, ytr)
        class_order = pipe.named_steps["clf"].classes_.tolist()
        pv, pt = pipe.predict_proba(Xva), pipe.predict_proba(Xte)
        mval, mtest = metrics(yva, pv, class_order), metrics(yte, pt, class_order)
        results.append(
            {
                "C": C,
                "valid": mval,
                "test": mtest,
                "class_order": class_order,
                "pipe": pipe,
            }
        )

        print(
            f"C={C} | valid ll={mval['log_loss']:.4f} brier={mval['brier']:.3f} | "
            f"test ll={mtest['log_loss']:.4f} brier={mtest['brier']:.3f}"
        )

    best = min(results, key=lambda r: r["valid"]["log_loss"])
    # guarda
    joblib.dump(best["pipe"], PIPE_DIR / f"logreg_{calibration}_C{best['C']}.joblib")
    with open(
        TRAINED_DIR / f"logreg_{calibration}_C{best['C']}_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model": "logreg",
                "C": best["C"],
                "calibration": calibration,
                "valid": best["valid"],
                "test": best["test"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"⭐ Mejor C={best['C']} (valid ll={best['valid']['log_loss']:.4f}). Guardado pipeline y reporte."
    )
    return best


def tune_xgb(calibration="isotonic", random_state=7):
    if not XGB_OK:
        raise RuntimeError("xgboost no está instalado.")
    print("\n=== Tuning XGBoost ===")
    df_train, df_valid, df_test = (
        load_split("train"),
        load_split("valid"),
        load_split("test"),
    )
    Xtr, ytr, num_cols, cat_cols = select_columns(df_train)
    Xva, yva, _, _ = select_columns(df_valid)
    Xte, yte, _, _ = select_columns(df_test)
    pre = make_preprocess(num_cols, cat_cols)

    # grid “ligero”
    grid = [
        {
            "max_depth": 4,
            "learning_rate": 0.03,
            "n_estimators": 1200,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
        },
        {
            "max_depth": 5,
            "learning_rate": 0.03,
            "n_estimators": 1500,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
        },
        {
            "max_depth": 5,
            "learning_rate": 0.06,
            "n_estimators": 800,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
        },
        {
            "max_depth": 6,
            "learning_rate": 0.03,
            "n_estimators": 1200,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
        },
    ]

    results = []
    for i, params in enumerate(grid, 1):
        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=check_random_state(random_state),
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        est = LabelEncodedEstimator(xgb)
        pipe = Pipeline(
            [("pre", pre), ("clf", calibrate(est, method=calibration, cv=3))]
        )
        pipe.fit(Xtr, ytr)
        class_order = pipe.named_steps["clf"].classes_.tolist()
        pv, pt = pipe.predict_proba(Xva), pipe.predict_proba(Xte)
        mval, mtest = metrics(yva, pv, class_order), metrics(yte, pt, class_order)
        results.append(
            {
                "params": params,
                "valid": mval,
                "test": mtest,
                "class_order": class_order,
                "pipe": pipe,
            }
        )
        print(
            f"{i}) {params} | valid ll={mval['log_loss']:.4f} brier={mval['brier']:.3f} | "
            f"test ll={mtest['log_loss']:.4f} brier={mtest['brier']:.3f}"
        )

    best = min(results, key=lambda r: r["valid"]["log_loss"])
    # guarda
    tag = "_".join([f"{k}{v}" for k, v in best["params"].items()])
    joblib.dump(best["pipe"], PIPE_DIR / f"xgb_{calibration}_{tag}.joblib")
    with open(
        TRAINED_DIR / f"xgb_{calibration}_{tag}_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "model": "xgb",
                "params": best["params"],
                "calibration": calibration,
                "valid": best["valid"],
                "test": best["test"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"⭐ Mejor XGB: {best['params']} (valid ll={best['valid']['log_loss']:.4f}). Guardado."
    )
    return best


if __name__ == "__main__":
    # corre ambos tunings
    best_lr = tune_logreg(C_grid=(0.5, 1.0, 2.0), calibration="isotonic")
    if XGB_OK:
        best_xgb = tune_xgb(calibration="isotonic")
