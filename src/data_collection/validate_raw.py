from __future__ import annotations
from pathlib import Path
import sys
import yaml
import pandas as pd
import numpy as np

# --- robust import, funciona como script o módulo ---
try:
    from src.utils.schemas import validate_matches, validate_odds
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from src.utils.schemas import validate_matches, validate_odds

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(
    (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
)
RAW_DIR = ROOT / PARAMS["data"]["paths"]["raw_dir"]
OUT_DIR = RAW_DIR / "validated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p, parse_dates=["date"])
    except Exception:
        try:
            df = pd.read_csv(p)
            if "date" in df:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        except Exception:
            return None


def preprocess_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Deja solo partidos con marcador y tipos correctos antes de validar."""
    df = df.copy()

    # 1) filtra filas con goles presentes (los sin jugar vienen con NaN)
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()

    # 2) asegura tipos enteros y matchweek entero (nullable si hace falta)
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")
    df["matchweek"] = pd.to_numeric(df["matchweek"], errors="coerce").astype("Int64")

    # 3) recalcula FTR de forma vectorizada (evita NaN)
    mask = df["home_goals"].notna() & df["away_goals"].notna()
    hg, ag = df["home_goals"], df["away_goals"]
    df["ftr"] = np.select(
        [mask & (hg > ag), mask & (hg < ag), mask & (hg == ag)],
        ["H", "A", "D"],
        default=None,
    ).astype("object")

    # 4) elimina cualquier fila que aún quede con NaN crítico
    df = df.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "ftr",
            "matchweek",
        ]
    )

    # 5) castea a tipos estrictos (int64) ya sin nulos
    df["home_goals"] = df["home_goals"].astype("int64")
    df["away_goals"] = df["away_goals"].astype("int64")
    df["matchweek"] = df["matchweek"].astype("int64")

    return df.reset_index(drop=True)


def preprocess_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica de odds si existen."""
    df = df.copy()
    # descarta cuotas inválidas o nulas
    for c in ["odd_home", "odd_draw", "odd_away"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(
        subset=["date", "home_team", "away_team", "odd_home", "odd_draw", "odd_away"]
    )
    df = df[
        (df["odd_home"] >= 1.01) & (df["odd_draw"] >= 1.01) & (df["odd_away"] >= 1.01)
    ]
    # matchweek a entero si viene como float/str
    if "matchweek" in df:
        df["matchweek"] = pd.to_numeric(df["matchweek"], errors="coerce").astype(
            "Int64"
        )
        df = df.dropna(subset=["matchweek"])
        df["matchweek"] = df["matchweek"].astype("int64")
    return df.reset_index(drop=True)


def main():
    # -------- Matches --------
    m_path = RAW_DIR / "matches.csv"
    m = safe_read_csv(m_path)
    if m is None:
        print("⚠️  No hay matches.csv para validar.")
    else:
        m_clean = preprocess_matches(m)
        mv = validate_matches(m_clean)  # Pandera
        (OUT_DIR / "matches.parquet").write_bytes(
            b""
        )  # asegura creación si hay permisos
        mv.to_parquet(OUT_DIR / "matches.parquet", index=False)
        print(f"✅ Matches validados → {OUT_DIR/'matches.parquet'} ({len(mv)})")

    # -------- Odds (opcional) --------
    o_path = RAW_DIR / "odds.csv"
    o = safe_read_csv(o_path)
    if o is None or o.empty:
        print("⚠️  No hay odds.csv o está vacío. Se omite la validación de cuotas.")
    else:
        o_clean = preprocess_odds(o)
        if o_clean.empty:
            print(
                "⚠️  odds.csv no contiene filas válidas tras limpieza. Se omite validación."
            )
        else:
            ov = validate_odds(o_clean)
            ov.to_parquet(OUT_DIR / "odds.parquet", index=False)
            print(f"✅ Odds validadas → {OUT_DIR/'odds.parquet'} ({len(ov)})")


if __name__ == "__main__":
    main()
