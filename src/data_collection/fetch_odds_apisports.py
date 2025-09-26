from __future__ import annotations
from pathlib import Path
import sys
import argparse
import time
import yaml
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(
    (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
)
SECRETS = yaml.safe_load((ROOT / PARAMS["secrets"]).read_text(encoding="utf-8"))
RAW_DIR = ROOT / PARAMS["data"]["paths"]["raw_dir"]

BASE = SECRETS["apisports"]["base_url"].rstrip("/")
HEADERS = {"x-apisports-key": SECRETS["apisports"]["api_key"]}

LEAGUE_ID = 39  # Premier League


def start_year(season_str: str) -> int:
    return int(season_str.split("-")[0])


def one_season_odds(year: int, pause: float = 0.6) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        params = {"league": LEAGUE_ID, "season": year, "page": page}
        r = requests.get(f"{BASE}/odds", headers=HEADERS, params=params, timeout=40)
        if r.status_code == 429:
            print("⏳ Rate limit. Esperando 10s…")
            time.sleep(10)
            continue
        if r.status_code != 200:
            print(f"⚠️  API error {r.status_code}: {r.text[:200]}")
            break
        js = r.json()
        resp = js.get("response", [])
        print(f"  página {page}: {len(resp)} items")
        if not resp:
            break

        for item in resp:
            fix = item.get("fixture", {})
            teams = item.get("teams", {})
            books = item.get("bookmakers", [])
            if not books:
                continue
            for bmk in books:
                name = bmk.get("name")
                # apostar por nombre de mercado estándar
                bets = [
                    b
                    for b in bmk.get("bets", [])
                    if str(b.get("name", "")).lower()
                    in ("match winner", "match-winner", "1x2")
                ]
                if not bets:
                    continue
                for bet in bets:
                    vals = {v.get("value"): v.get("odd") for v in bet.get("values", [])}
                    try:
                        rows.append(
                            {
                                "date": pd.to_datetime(
                                    fix.get("date"), errors="coerce"
                                ),
                                "season": f"{year}-{year+1}",
                                "matchweek": (
                                    None
                                    if not fix.get("round")
                                    else str(fix["round"])
                                    .split(" - ")[-1]
                                    .replace("Round ", "")
                                ),
                                "bookmaker": name,
                                "home_team": teams.get("home", {}).get("name"),
                                "away_team": teams.get("away", {}).get("name"),
                                "odd_home": float(
                                    vals.get("Home") or vals.get("1") or "nan"
                                ),
                                "odd_draw": float(
                                    vals.get("Draw")
                                    or vals.get("X")
                                    or vals.get("0")
                                    or "nan"
                                ),
                                "odd_away": float(
                                    vals.get("Away") or vals.get("2") or "nan"
                                ),
                            }
                        )
                    except Exception:
                        continue

        # paginación
        paging = js.get("paging", {})
        if paging.get("current", 1) >= paging.get("total", 1):
            break
        page += 1
        time.sleep(pause)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(
            subset=[
                "date",
                "home_team",
                "away_team",
                "odd_home",
                "odd_draw",
                "odd_away",
            ]
        )
        df["matchweek"] = pd.to_numeric(df["matchweek"], errors="coerce").astype(
            "Int64"
        )
    return df


def main(seasons: list[str]) -> None:
    years = [start_year(s) for s in seasons]
    frames = []
    for y in years:
        print(f"Descargando odds API-Sports {y}-{y+1}…")
        df = one_season_odds(y)
        print(f"  → {len(df)} filas")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "odds.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    casas = sorted(out.bookmaker.dropna().unique().tolist()) if not out.empty else []
    print(f"✅ Guardado {out_path} ({len(out)} filas) | Casas: {casas}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_seasons = (
        PARAMS["data"]["seasons"]["train"]
        + PARAMS["data"]["seasons"]["valid"]
        + PARAMS["data"]["seasons"]["test"]
        + PARAMS["data"]["seasons"]["prod"]
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=default_seasons,
        help="Lista de temporadas ej. 2024-2025 2025-2026",
    )
    args = parser.parse_args()
    try:
        main(args.seasons)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
