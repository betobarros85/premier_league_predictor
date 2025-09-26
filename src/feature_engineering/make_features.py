from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd


# -------------------- util paths/config --------------------
ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(
    (ROOT / "config" / "parameters.yaml").read_text(encoding="utf-8")
)

RAW_DIR = ROOT / PARAMS["data"]["paths"]["raw_dir"]
VALID_DIR = RAW_DIR / "validated"
PROC_DIR = ROOT / PARAMS["data"]["paths"]["processed_dir"]
PROC_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = PARAMS["data"]["seasons"]
FEAT = PARAMS["features"]
BOOK_PRIORITY = PARAMS.get("production", {}).get("bookmakers_priority", [])


# -------------------- helpers: splits --------------------
def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def where_season(s):
        if s in SEASONS["train"]:
            return "train"
        if s in SEASONS["valid"]:
            return "valid"
        if s in SEASONS["test"]:
            return "test"
        if s in SEASONS["prod"]:
            return "prod"
        return "other"

    df["split"] = df["season"].map(where_season)
    return df[df["split"] != "other"].reset_index(drop=True)


# -------------------- base table + target --------------------
def build_base() -> pd.DataFrame:
    mpath = VALID_DIR / "matches.parquet"
    if not mpath.exists():
        raise FileNotFoundError(
            f"No existe {mpath}. Corre src/data_collection/validate_raw.py primero."
        )
    m = pd.read_parquet(mpath)

    # orden canónico
    m = m.sort_values(["season", "date", "home_team", "away_team"]).reset_index(
        drop=True
    )

    # target ya viene en ftr (H/D/A). Por si acaso, lo recalculamos seguro.
    mask = m["home_goals"].notna() & m["away_goals"].notna()
    hg, ag = m["home_goals"], m["away_goals"]
    m["y"] = np.select(
        [mask & (hg > ag), mask & (hg < ag), mask & (hg == ag)],
        ["H", "A", "D"],
        default=None,
    )

    # llave estable del partido (fecha + equipos + temporada)
    m["match_id"] = (
        m["season"].astype(str)
        + "_"
        + m["date"].dt.strftime("%Y%m%d")
        + "_"
        + m["home_team"].astype(str)
        + "_vs_"
        + m["away_team"].astype(str)
    )

    m = add_split(m)
    return m


# -------------------- rolling features por equipo --------------------
def add_rolling_features(m: pd.DataFrame, windows=(3, 5, 8)) -> pd.DataFrame:
    df = m.copy()

    # puntos por partido para el local, desde el POV del local
    df["home_points"] = np.select(
        [df["y"] == "H", df["y"] == "D", df["y"] == "A"], [3, 1, 0], default=np.nan
    )
    df["away_points"] = np.select(
        [df["y"] == "A", df["y"] == "D", df["y"] == "H"], [3, 1, 0], default=np.nan
    )

    # construimos una tabla “long” por equipo para computar rollings simétricos
    def team_long(role: str) -> pd.DataFrame:
        is_home = role == "home"
        out = pd.DataFrame(
            {
                "season": df["season"],
                "date": df["date"],
                "team": df["home_team"] if is_home else df["away_team"],
                "gf": df["home_goals"] if is_home else df["away_goals"],
                "ga": df["away_goals"] if is_home else df["home_goals"],
                "pts": df["home_points"] if is_home else df["away_points"],
            }
        )
        out = out.sort_values(["team", "season", "date"])
        return out

    long_home = team_long("home")
    long_away = team_long("away")
    long_all = pd.concat([long_home, long_away], ignore_index=True)

    # shift para que el rolling NO use el partido actual (solo pasado)
    long_all[["gf", "ga", "pts"]] = long_all[["gf", "ga", "pts"]].astype("float")
    long_all[["gf", "ga", "pts"]] = long_all.groupby(["team", "season"])[
        ["gf", "ga", "pts"]
    ].shift(1)

    # computa rollings por ventana dentro de cada temporada y equipo
    for w in windows:
        grp = long_all.groupby(["team", "season"], group_keys=False)
        long_all[f"gf_avg_{w}"] = (
            grp["gf"]
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        long_all[f"ga_avg_{w}"] = (
            grp["ga"]
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        long_all[f"pts_avg_{w}"] = (
            grp["pts"]
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        long_all[f"gd_avg_{w}"] = long_all[f"gf_avg_{w}"] - long_all[f"ga_avg_{w}"]

    # separamos a home/away de vuelta y mergeamos al partido
    def suffix_cols(prefix: str) -> list[str]:
        cols = []
        for w in windows:
            cols += [f"gf_avg_{w}", f"ga_avg_{w}", f"gd_avg_{w}", f"pts_avg_{w}"]
        return cols

    home_feats = long_all.loc[
        long_all.index[: len(long_home)],  # primeros N son home por cómo concatenamos
        ["team", "season", "date"] + suffix_cols("home"),
    ].copy()
    home_feats = home_feats.rename(columns={"team": "home_team"})
    away_feats = long_all.loc[
        long_all.index[len(long_home) :],
        ["team", "season", "date"] + suffix_cols("away"),
    ].copy()
    away_feats = away_feats.rename(columns={"team": "away_team"})

    # renombrar columnas para distinguir
    for w in windows:
        home_feats = home_feats.rename(
            columns={
                f"gf_avg_{w}": f"home_gf_avg_{w}",
                f"ga_avg_{w}": f"home_ga_avg_{w}",
                f"gd_avg_{w}": f"home_gd_avg_{w}",
                f"pts_avg_{w}": f"home_pts_avg_{w}",
            }
        )
        away_feats = away_feats.rename(
            columns={
                f"gf_avg_{w}": f"away_gf_avg_{w}",
                f"ga_avg_{w}": f"away_ga_avg_{w}",
                f"gd_avg_{w}": f"away_gd_avg_{w}",
                f"pts_avg_{w}": f"away_pts_avg_{w}",
            }
        )

    out = df.merge(home_feats, on=["season", "date", "home_team"], how="left").merge(
        away_feats, on=["season", "date", "away_team"], how="left"
    )
    return out


# -------------------- Elo baseline --------------------
def add_elo_features(m: pd.DataFrame, k=20, home_adv=60) -> pd.DataFrame:
    df = m.sort_values(["season", "date"]).copy()
    teams = pd.Index(sorted(set(df["home_team"]).union(set(df["away_team"]))))
    elo = pd.Series(1500.0, index=teams)

    rows = []
    cur_season = None
    for r in df.itertuples(index=False):
        if r.season != cur_season:
            # reinicio suave por temporada (puedes ajustar si prefieres carry-over parcial)
            cur_season = r.season
            # nada: mantenemos Elo acumulado; si quisieras reiniciar, descomenta:
            # elo[:] = 1500.0

        eh = elo.get(r.home_team, 1500.0) + home_adv
        ea = elo.get(r.away_team, 1500.0)
        # expectativa local
        ph = 1.0 / (1 + 10 ** (-(eh - ea) / 400))

        # resultado real (1, 0.5, 0)
        if r.y == "H":
            rh = 1.0
        elif r.y == "D":
            rh = 0.5
        elif r.y == "A":
            rh = 0.0
        else:
            rh = np.nan

        rows.append(
            {
                "match_id": r.match_id,
                "home_elo_pre": elo.get(r.home_team, 1500.0),
                "away_elo_pre": elo.get(r.away_team, 1500.0),
                "elo_diff": elo.get(r.home_team, 1500.0) - elo.get(r.away_team, 1500.0),
                "elo_winprob_home": ph,
            }
        )

        # update post-partido si hay resultado
        if not np.isnan(rh):
            elo_h_pre = elo.get(r.home_team, 1500.0)
            elo_a_pre = elo.get(r.away_team, 1500.0)

            # expectativa sin ventaja de casa para el update
            ph_noha = 1.0 / (1 + 10 ** (-(elo_h_pre - elo_a_pre) / 400))
            delta = k * (rh - ph_noha)
            elo[r.home_team] = elo_h_pre + delta
            elo[r.away_team] = elo_a_pre - delta

    elo_df = pd.DataFrame(rows)
    out = df.merge(elo_df, on="match_id", how="left")
    return out


# -------------------- merge odds (opcional) --------------------
def merge_odds_if_available(m: pd.DataFrame) -> pd.DataFrame:
    opath = VALID_DIR / "odds.parquet"
    if not opath.exists():
        print("ℹ️  No hay odds.parquet; continúo sin cuotas.")
        return m

    o = pd.read_parquet(opath)
    # agregamos por partido y preferimos bookmaker prioritarios
    # key de unión:
    keys = ["season", "date", "home_team", "away_team"]

    # si hay prioridad, probamos a elegir por orden; si no, mediana
    if BOOK_PRIORITY:
        o["bookmaker_rank"] = o["bookmaker"].map(
            {b: i for i, b in enumerate(BOOK_PRIORITY)}
        )
        o["bookmaker_rank"] = o["bookmaker_rank"].fillna(999)
        o = o.sort_values(keys + ["bookmaker_rank"])

        # nos quedamos con la primera por partido
        keep = o.groupby(keys, as_index=False).first()
        keep = keep.drop(columns=["bookmaker_rank"])
        merged = m.merge(
            keep[keys + ["odd_home", "odd_draw", "odd_away", "bookmaker"]],
            on=keys,
            how="left",
        )
    else:
        agg = o.groupby(keys, as_index=False)[
            ["odd_home", "odd_draw", "odd_away"]
        ].median()
        merged = m.merge(agg, on=keys, how="left")
        merged["bookmaker"] = np.nan

    return merged


# -------------------- main --------------------
def main():
    base = build_base()
    base = add_rolling_features(base, windows=FEAT["rolling_windows"])
    base = add_elo_features(
        base, k=FEAT["elo"]["k"], home_adv=FEAT["elo"]["home_advantage"]
    )
    base = merge_odds_if_available(base)

    # orden de columnas: clave, target, odds, features
    id_cols = [
        "match_id",
        "date",
        "season",
        "matchweek",
        "home_team",
        "away_team",
        "split",
    ]
    target_cols = ["y"]
    odds_cols = [
        c
        for c in ["odd_home", "odd_draw", "odd_away", "bookmaker"]
        if c in base.columns
    ]
    feat_cols = [c for c in base.columns if c not in id_cols + target_cols + odds_cols]

    base = base[id_cols + target_cols + odds_cols + sorted(feat_cols)]

    # guardar por split
    for sp in ["train", "valid", "test", "prod"]:
        out = base[base["split"] == sp].reset_index(drop=True)
        out_path = PROC_DIR / f"features_{sp}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"✅ Guardado {out_path} ({len(out)} filas, {out.shape[1]} cols)")
    print("Listo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # hoy no necesitamos flags; dejamos hook para futuro
    _ = parser.parse_args()
    main()
