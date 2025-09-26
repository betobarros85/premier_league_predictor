# tools/check_step2.py
from pathlib import Path
import pandas as pd
import yaml


def exists_checks() -> None:
    print("#1 Archivos clave")
    paths = [
        "data/raw/matches.csv",
        "data/raw/odds.csv",
        "data/raw/validated/matches.parquet",
        "data/raw/validated/odds.parquet",
    ]
    for p in paths:
        print(f"{p} → {Path(p).exists()}")


def parquet_checks():
    print("\n#2 Lectura parquet")
    m = pd.read_parquet("data/raw/validated/matches.parquet")
    o = pd.read_parquet("data/raw/validated/odds.parquet")
    print("Matches:", m.shape, "cols:", list(m.columns))
    print("Odds   :", o.shape, "cols:", list(o.columns))
    print("Seasons (matches):", sorted(m.season.unique().tolist()))
    print("Seasons (odds)   :", sorted(o.season.unique().tolist()))
    print("Bookmakers:", o.bookmaker.value_counts().to_dict())
    return m, o


def names_checks() -> None:
    print("\n#3 Normalización de nombres")
    with open("config/team_name_map.yaml", encoding="utf-8") as f:
        mp = yaml.safe_load(f)
    rev = {a.lower(): k for k, v in mp.items() for a in [k, *v]}

    m_csv = pd.read_csv("data/raw/matches.csv")
    bad_h = sorted(
        {t for t in m_csv.home_team.astype(str).str.lower().unique() if t not in rev}
    )
    bad_a = sorted(
        {t for t in m_csv.away_team.astype(str).str.lower().unique() if t not in rev}
    )
    print("No mapeados (home):", bad_h[:10])
    print("No mapeados (away):", bad_a[:10])


def dup_dates_checks(m: pd.DataFrame, o: pd.DataFrame) -> None:
    print("\n#4 Duplicados y rangos de fechas")
    km = ["date", "home_team", "away_team", "season"]
    ko = ["date", "home_team", "away_team", "season", "bookmaker"]
    print("Duplicados matches:", int(m.duplicated(km).sum()))
    print("Duplicados odds   :", int(o.duplicated(ko).sum()))
    print("Rango fechas matches:", m.date.min(), "→", m.date.max())
    print("Rango fechas odds   :", o.date.min(), "→", o.date.max())
    print("Max matchweek por temporada (ejemplo 10):")
    print(m.groupby("season")["matchweek"].max().head(10))


def join_rate(m: pd.DataFrame, o: pd.DataFrame) -> None:
    print("\n#5 Cobertura de odds vs partidos")
    keys = ["date", "home_team", "away_team", "season"]
    mm = m[keys].drop_duplicates().assign(has_match=1)
    oo = o[keys].drop_duplicates().assign(has_odds=1)
    j = mm.merge(oo, on=keys, how="left")
    cover_all = 100 * j["has_odds"].fillna(0).mean()
    print(f"Cobertura global de odds: {cover_all:.1f}%")
    by_season = (
        j.assign(has_odds=j["has_odds"].fillna(0))
        .groupby("season")["has_odds"]
        .mean()
        .mul(100)
        .round(1)
    )
    print("Cobertura por temporada (%):")
    print(by_season.to_string())


def odds_sanity(o: pd.DataFrame) -> None:
    print("\n#6 Sanidad de cuotas")
    cols = ["odd_home", "odd_draw", "odd_away"]
    mins = o[cols].min().to_dict()
    maxs = o[cols].max().to_dict()
    na = o[cols].isna().sum().to_dict()
    overround = (1 / o["odd_home"] + 1 / o["odd_draw"] + 1 / o["odd_away"]) * 100
    print("Min odds:", mins)
    print("Max odds:", maxs)
    print("Na odds :", na)
    print(
        "Overround% (p5, p50, p95):",
        float(overround.quantile(0.05)),
        float(overround.median()),
        float(overround.quantile(0.95)),
    )


if __name__ == "__main__":
    exists_checks()
    m_df, o_df = parquet_checks()
    names_checks()
    dup_dates_checks(m_df, o_df)
    join_rate(m_df, o_df)
    odds_sanity(o_df)
