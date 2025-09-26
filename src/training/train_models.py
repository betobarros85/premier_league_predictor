# src/training/train_models.py
from __future__ import annotations

from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
import joblib

# XGBoost es opcional
try:
    from xgboost import XGBClassifier  # type: ignore

    XGB_OK = True
except Exception:
    XGB_OK = False
    warnings.filterwarnings("ignore", "xgboost")

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
CATS = ["bookmaker"]  # si existe


def load_split(name: str) -> pd.DataFrame:
    p = PROC_DIR / f"features_{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Falta {p}. Corre make_features primero.")
    return pd.read_parquet(p)


def select_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    y = df["y"].astype("category")

    # columnas que NUNCA deben entrar al modelo
    leak_cols = {
        "y",
        "ftr",
        "home_goals",
        "away_goals",
        "home_points",
        "away_points",  # puntos del partido actual
    }
    id_cols = {
        "match_id",
        "date",
        "season",
        "matchweek",
        "home_team",
        "away_team",
        "split",
    }

    feat = [c for c in df.columns if c not in leak_cols | id_cols]

    # num/cat
    num_cols = [
        c for c in feat if (df[c].dtype.kind in "fcbi") and c not in ["bookmaker"]
    ]
    cat_cols = [c for c in feat if c in ["bookmaker"] and c in df.columns]

    return df[feat], y, num_cols, cat_cols


def make_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        pre = ColumnTransformer(
            [
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ]
        )
    else:
        pre = ColumnTransformer([("num", num_pipe, num_cols)])
    return pre


def build_estimator(kind: str, random_state: int = 7):
    rs = check_random_state(random_state)
    if kind == "logreg":
        return LogisticRegression(
            multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=200
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=10,
            random_state=rs,
            n_jobs=-1,
        )
    if kind == "xgb":
        if not XGB_OK:
            raise RuntimeError("xgboost no está disponible en el entorno.")
        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=6,
            learning_rate=0.06,
            n_estimators=800,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=rs,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
        )
        return LabelEncodedEstimator(xgb)

    raise ValueError(kind)


class LabelEncodedEstimator(BaseEstimator, ClassifierMixin):
    """Envuelve un clasificador para codificar y a enteros internamente (p.ej. XGB)."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.le_ = None
        self.classes_ = None

    def fit(self, X, y):
        y = pd.Series(y).astype(str)
        self.le_ = LabelEncoder().fit(y)
        y_enc = self.le_.transform(y)
        self.est_ = clone(self.base_estimator)
        self.est_.fit(X, y_enc)
        # exposicion de clases como strings (p.ej. ['A','D','H'])
        self.classes_ = self.le_.classes_.tolist()
        return self

    def predict_proba(self, X):
        proba = self.est_.predict_proba(X)
        # columnas ya están en el orden 0..K-1 que mapea a self.classes_
        return proba

    def predict(self, X):
        pred_enc = self.est_.predict(X)
        return self.le_.inverse_transform(pred_enc)


def calibrate(est, method: str, cv: int = 3):
    return CalibratedClassifierCV(est, method=method, cv=cv)


# ===== métricas usando el ORDEN REAL de clases del modelo =====
def metrics(
    y_true: pd.Series, proba: np.ndarray, class_order: list[str]
) -> Dict[str, Any]:
    y_true = y_true.astype(str)
    ll = log_loss(y_true, proba, labels=class_order)

    briers = []
    for i, cls in enumerate(class_order):
        briers.append(brier_score_loss((y_true == cls).astype(int), proba[:, i]))
    brier_mc = float(np.mean(briers))

    pred = np.array([class_order[i] for i in proba.argmax(axis=1)])
    acc = accuracy_score(y_true, pred)

    return {"log_loss": float(ll), "brier": brier_mc, "accuracy": float(acc)}


def train_and_eval(
    model_name: str, calib_method: str, random_state: int = 7
) -> Dict[str, Any]:
    print(f"\n=== Modelo: {model_name} | calibración: {calib_method} ===")

    df_train = load_split("train")
    df_valid = load_split("valid")
    df_test = load_split("test")
    df_prod = load_split("prod")

    Xtr_all, ytr, num_cols, cat_cols = select_columns(df_train)
    Xva_all, yva, _, _ = select_columns(df_valid)
    Xte_all, yte, _, _ = select_columns(df_test)
    Xpr_all, _, _, _ = select_columns(df_prod)

    pre = make_preprocess(num_cols, cat_cols)
    base_est = build_estimator(model_name, random_state=random_state)
    cal = calibrate(base_est, method=calib_method, cv=3)

    pipe = Pipeline([("pre", pre), ("clf", cal)])
    pipe.fit(Xtr_all, ytr)

    # ¡Clave! Orden real de clases del clasificador calibrado
    class_order = pipe.named_steps["clf"].classes_.tolist()

    # VALID / TEST
    proba_valid = pipe.predict_proba(Xva_all)
    proba_test = pipe.predict_proba(Xte_all)
    m_valid = metrics(yva, proba_valid, class_order)
    m_test = metrics(yte, proba_test, class_order)

    out = {"model": model_name, "calibration": calib_method, "classes": class_order}
    out["valid"] = m_valid
    out["test"] = m_test

    # Guardar pipeline y reporte
    pipe_path = PIPE_DIR / f"{model_name}_{calib_method}.joblib"
    joblib.dump(pipe, pipe_path)
    report_path = TRAINED_DIR / f"{model_name}_{calib_method}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("→ valid:", out["valid"])
    print("→ test :", out["test"])
    print(f"💾 Guardado pipeline en {pipe_path.name} y reporte en {report_path.name}")

    # Predicciones PROD
    prod_proba = pipe.predict_proba(Xpr_all)
    prod_df = df_prod[ID_COLS].copy()
    for i, cls in enumerate(class_order):
        prod_df[f"p_{cls}"] = prod_proba[:, i]
    pred_path = PROC_DIR / f"predictions_prod_{model_name}_{calib_method}.csv"
    prod_df.to_csv(pred_path, index=False)
    print(f"📄 Predicciones prod → {pred_path.name} ({len(prod_df)} filas)")
    return out


def main(args):
    models = args.models or ["logreg", "rf"] + (["xgb"] if XGB_OK else [])
    method = args.calibration
    results = []
    for m in models:
        if m == "xgb" and not XGB_OK:
            print("⚠️ xgboost no instalado: omito 'xgb'.")
            continue
        res = train_and_eval(m, method, random_state=args.random_state)
        results.append(res)

    if results:
        ranking = sorted(results, key=lambda r: r["valid"]["log_loss"])
        summary_path = TRAINED_DIR / "summary_step4.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"ranking": ranking}, f, indent=2, ensure_ascii=False)
        print("\n🏁 Ranking (valid logloss):")
        for i, r in enumerate(ranking, 1):
            print(
                f"{i}. {r['model']}-{r['calibration']}  | "
                f"valid ll={r['valid']['log_loss']:.4f}, test ll={r['test']['log_loss']:.4f}"
            )
        print(f"💾 Resumen → {summary_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="logreg rf xgb (xgb si está instalado)",
    )
    ap.add_argument(
        "--calibration", default="isotonic", choices=["isotonic", "sigmoid"]
    )
    ap.add_argument("--random_state", type=int, default=7)
    args = ap.parse_args()
    main(args)
