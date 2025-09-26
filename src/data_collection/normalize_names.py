from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(
    (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
)
RAW_DIR = ROOT / PARAMS["data"]["paths"]["raw_dir"]
MAP = yaml.safe_load(
    (ROOT / "config" / "team_name_map.yaml").read_text(encoding="utf-8")
)


def build_alias_to_canonical(map_yaml: dict) -> dict:
    alias2canon = {}
    for canon, aliases in map_yaml.items():
        alias2canon[canon.lower()] = canon
        for a in aliases:
            alias2canon[a.lower()] = canon
    return alias2canon


def normalize(df: pd.DataFrame, alias2canon: dict) -> pd.DataFrame:
    df = df.copy()
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.lower().map(alias2canon).fillna(df[col])
    return df


def safe_read_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        # intenta con parse_dates si la columna existe
        return pd.read_csv(p, parse_dates=["date"])
    except Exception:
        # segundo intento sin parse_dates
        try:
            return pd.read_csv(p)
        except Exception:
            return None


def main():
    alias2canon = build_alias_to_canonical(MAP)

    matches_path = RAW_DIR / "matches.csv"
    odds_path = RAW_DIR / "odds.csv"

    # Matches
    matches = safe_read_csv(matches_path)
    if matches is not None:
        matches = normalize(matches, alias2canon)
        matches.to_csv(matches_path, index=False)
        print(f"✅ Nombres normalizados en {matches_path}")
    else:
        print(f"⚠️  No se pudo leer {matches_path} (inexistente o vacío).")

    # Odds (opcional)
    odds = safe_read_csv(odds_path)
    if odds is not None:
        odds = normalize(odds, alias2canon)
        odds.to_csv(odds_path, index=False)
        print(f"✅ Nombres normalizados en {odds_path}")
    else:
        print(
            f"⚠️  No se pudo leer {odds_path} (inexistente o vacío). Se continúa sin normalizar odds."
        )


if __name__ == "__main__":
    main()
