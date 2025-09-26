from __future__ import annotations
from pathlib import Path
import glob
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(
    (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
)
RAW_DIR = ROOT / PARAMS["data"]["paths"]["raw_dir"]
ODDS_DIR = ROOT / PARAMS["data"]["paths"]["odds_dir"]
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# Columnas por casa en football-data
BOOK_COLS = {
    "Bet365": ("B365H", "B365D", "B365A"),
    "Pinnacle": ("PSH", "PSD", "PSA"),
    "bwin": ("BWH", "BWD", "BWA"),
    # Añade más si las tienes: WH -> ("WHH","WHD","WHA"), VC -> ("VCH","VCD","VCA"), etc.
}


def infer_season_from_date(dt: pd.Timestamp) -> str | None:
    """
    EPL: temporada va aproximadamente de agosto a mayo.
    Regla: si mes >= 7 (julio) -> season empieza ese año; si no, empieza el año anterior.
    """
    if pd.isna(dt):
        return None
    year = int(dt.year)
    start = year if dt.month >= 7 else year - 1
    return f"{start}-{start+1}"


def load_one_csv(path: Path) -> pd.DataFrame:
    # football-data suele venir con latin-1
    df = pd.read_csv(path, encoding="latin-1")
    # columnas mínimas
    if not {"Date", "HomeTeam", "AwayTeam"}.issubset(df.columns):
        return pd.DataFrame()
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # inferir temporada por fecha
    df["season"] = df["date"].map(infer_season_from_date)

    # ordenar y crear matchweek por fecha
    df = df.sort_values(["date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    dates = df["date"].dropna().drop_duplicates().reset_index(drop=True)
    date_to_mw = {d: i + 1 for i, d in enumerate(dates)}
    df["matchweek"] = df["date"].map(date_to_mw).astype("Int64")

    blocks: list[pd.DataFrame] = []
    for book, cols in BOOK_COLS.items():
        h, d, a = cols
        if not set([h, d, a]).issubset(df.columns):
            continue
        b = df[["date", "season", "matchweek", "HomeTeam", "AwayTeam", h, d, a]].copy()
        b = b.rename(
            columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                h: "odd_home",
                d: "odd_draw",
                a: "odd_away",
            }
        )
        b = b.dropna(
            subset=[
                "odd_home",
                "odd_draw",
                "odd_away",
                "date",
                "home_team",
                "away_team",
            ]
        )
        # filtros básicos de sanidad
        b = b[
            (b["odd_home"] >= 1.01) & (b["odd_draw"] >= 1.01) & (b["odd_away"] >= 1.01)
        ]
        b["bookmaker"] = book
        blocks.append(
            b[
                [
                    "date",
                    "season",
                    "matchweek",
                    "bookmaker",
                    "home_team",
                    "away_team",
                    "odd_home",
                    "odd_draw",
                    "odd_away",
                ]
            ]
        )

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def main():
    files = sorted(glob.glob(str(ODDS_DIR / "E0*.csv")))  # usa todos tus E0*.csv
    if not files:
        print(f"⚠️  No hay CSVs en {ODDS_DIR}.")
        return
    frames = []
    for f in files:
        print("Leyendo", f)
        df = load_one_csv(Path(f))
        print("  →", len(df), "filas útiles")
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "odds.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    casas = sorted(out.bookmaker.unique().tolist()) if not out.empty else []
    seasons = sorted(out.season.dropna().unique().tolist()) if not out.empty else []
    print(
        f"✅ Guardado {out_path} ({len(out)} filas) | Casas: {casas} | Seasons: {seasons}"
    )


if __name__ == "__main__":
    main()
