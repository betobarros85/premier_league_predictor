# src/data_collection/fetch_matches_fbref.py
from __future__ import annotations

from pathlib import Path
import io
import re
import sys
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Config project paths ---
ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(
    (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
)
RAW_DIR = ROOT / PARAMS["data"]["paths"]["raw_dir"]

# --- FBref config ---
LEAGUE_ID = 9  # Premier League
SEASON_URL = "https://fbref.com/en/comps/{lid}/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "Connection": "keep-alive",
}


def make_session() -> requests.Session:
    s = requests.Session()
    rtry = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    ad = HTTPAdapter(max_retries=rtry, pool_connections=5, pool_maxsize=5)
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update(HEADERS)
    return s


def parse_fbref_html_to_df(html: str) -> pd.DataFrame:
    # pandas.read_html usará lxml si está instalado (recomendado)
    tables = pd.read_html(io.StringIO(html))
    df = tables[0].copy()
    # Columnas esperadas
    rename = {
        "Wk": "matchweek",
        "Date": "date",
        "Home": "home_team",
        "Away": "away_team",
        "Score": "score",
    }
    for k, v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    # Filtra filas con fecha y parsea
    df = df[~df["date"].isna()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Extrae goles desde "2–1" o "2-1"
    def split_score(s):
        if isinstance(s, str) and ("-" in s or "–" in s):
            a, b = re.split(r"[-–]", s)
            if a.strip().isdigit() and b.strip().isdigit():
                return int(a), int(b)
        return None, None

    hg, ag = zip(*[split_score(s) for s in df.get("score", [])])
    df["home_goals"] = hg
    df["away_goals"] = ag
    return df


def fetch_one_season_fbref(season: str) -> pd.DataFrame:
    session = make_session()
    url = SEASON_URL.format(lid=LEAGUE_ID, season=season)
    print(f"FBref {season} -> {url}")
    r = session.get(url, timeout=30)
    if r.status_code == 403:
        raise PermissionError("403 from FBref")
    r.raise_for_status()
    df = parse_fbref_html_to_df(r.text)
    df["season"] = season

    # matchweek: usar si existe; si no, enumerar fechas
    if "matchweek" in df.columns:
        df["matchweek"] = pd.to_numeric(df["matchweek"], errors="coerce")
    if "matchweek" not in df.columns or df["matchweek"].isna().all():
        df = df.sort_values(["date", "home_team", "away_team"])
        dates = df["date"].dropna().drop_duplicates().reset_index(drop=True)
        date_to_mw = {d: i + 1 for i, d in enumerate(dates)}
        df["matchweek"] = df["date"].map(date_to_mw)

    # FTR vectorizado
    mask = df["home_goals"].notna() & df["away_goals"].notna()
    hg = df["home_goals"]
    ag = df["away_goals"]
    df["ftr"] = np.select(
        [mask & (hg > ag), mask & (hg < ag), mask & (hg == ag)],
        ["H", "A", "D"],
        default=None,
    ).astype("object")

    cols = [
        "date",
        "season",
        "matchweek",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "ftr",
    ]
    return (
        df[cols]
        .dropna(subset=["date", "home_team", "away_team"])
        .reset_index(drop=True)
    )


# ---- Fallback OpenFootball (GitHub raw) ----
OPENFOOTBALL_CANDIDATES = [
    # Repo principal por país
    "https://raw.githubusercontent.com/openfootball/eng-england/master/{slug}/1-premierleague.csv",
    # Alternativa histórica
    "https://raw.githubusercontent.com/openfootball/football.csv/master/england/{slug}/1-premierleague.csv",
]


def season_slug_openfootball(season: str) -> str:
    # "2018-2019" -> "2018-19"; "2024-2025" -> "2024-25"
    a, b = season.split("-")
    return f"{a}-{b[-2:]}"


def fetch_one_season_openfootball(season: str) -> pd.DataFrame:
    slug = season_slug_openfootball(season)
    content = None
    last_err = None
    used_url = None

    for base in OPENFOOTBALL_CANDIDATES:
        url = base.format(slug=slug)
        print(f"OpenFootball {season} -> {url}")
        try:
            r = requests.get(
                url, timeout=30, headers={"User-Agent": HEADERS["User-Agent"]}
            )
            r.raise_for_status()
            content = r.text
            used_url = url
            break
        except Exception as e:
            last_err = e
            continue

    if content is None:
        raise RuntimeError(
            f"No se pudo obtener CSV de OpenFootball para {season}. Último error: {last_err}"
        )

    df = pd.read_csv(io.StringIO(content))

    # Columnas típicas: date, home/away o team1/team2, score o ft, round
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(names)

    date_col = pick("date")
    home_col = pick("home", "home_team", "team1")
    away_col = pick("away", "away_team", "team2")
    score_col = pick("score", "ft", "result")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    out["home_team"] = df[home_col].astype(str)
    out["away_team"] = df[away_col].astype(str)

    def split_score(s: str):
        if isinstance(s, str) and ("-" in s or "–" in s):
            a, b = re.split(r"[-–]", s)
            if a.strip().isdigit() and b.strip().isdigit():
                return int(a), int(b)
        return None, None

    hg, ag = zip(*[split_score(str(s)) for s in df[score_col]])
    out["home_goals"] = hg
    out["away_goals"] = ag
    out["season"] = season

    # matchweek por fechas (aprox si no hay "round")
    out = out.sort_values(["date", "home_team", "away_team"])
    dates = out["date"].dropna().drop_duplicates().reset_index(drop=True)
    date_to_mw = {d: i + 1 for i, d in enumerate(dates)}
    out["matchweek"] = out["date"].map(date_to_mw)

    # FTR vectorizado
    mask = out["home_goals"].notna() & out["away_goals"].notna()
    hg = out["home_goals"]
    ag = out["away_goals"]
    out["ftr"] = np.select(
        [mask & (hg > ag), mask & (hg < ag), mask & (hg == ag)],
        ["H", "A", "D"],
        default=None,
    ).astype("object")

    cols_out = [
        "date",
        "season",
        "matchweek",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "ftr",
    ]
    print(f"Usado OpenFootball URL: {used_url}")
    return (
        out[cols_out]
        .dropna(subset=["date", "home_team", "away_team"])
        .reset_index(drop=True)
    )


# ---- Orquestador: FBref -> OpenFootball fallback ----
def fetch_one_season(season: str) -> pd.DataFrame:
    try:
        time.sleep(0.7)  # pequeña pausa
        return fetch_one_season_fbref(season)
    except Exception as e:
        print(f"FBref falló: {e}. Usando fallback OpenFootball…")
        return fetch_one_season_openfootball(season)


def main(seasons: list[str]) -> None:
    frames = [fetch_one_season(s) for s in seasons]
    out = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW_DIR / "matches.csv", index=False, encoding="utf-8")
    print(f"✅ matches.csv listo ({len(out)} filas)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seasons",
        nargs="+",
        required=False,
        default=PARAMS["data"]["seasons"]["train"]
        + PARAMS["data"]["seasons"]["valid"]
        + PARAMS["data"]["seasons"]["test"]
        + PARAMS["data"]["seasons"]["prod"],
    )
    args = p.parse_args()
    try:
        main(args.seasons)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
