from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Field
from pandera.typing import DataFrame, Series


# ========== ESQUEMA DE PARTIDOS ==========
class MatchesSchema(pa.DataFrameModel):
    date: Series[pa.DateTime] = Field(nullable=False)
    season: Series[str] = Field(nullable=False)
    matchweek: Series[int] = Field(ge=1, nullable=False)

    home_team: Series[str] = Field(nullable=False)
    away_team: Series[str] = Field(nullable=False)

    home_goals: Series[int] = Field(ge=0, nullable=False)
    away_goals: Series[int] = Field(ge=0, nullable=False)

    # H=local, D=empate, A=visita
    ftr: Series[str] = Field(isin=["H", "D", "A"], nullable=False)


# ========== ESQUEMA DE ODDS ==========
class OddsSchema(pa.DataFrameModel):
    date: Series[pa.DateTime] = Field(nullable=False)
    season: Series[str] = Field(nullable=False)
    matchweek: Series[int] = Field(ge=1, nullable=False)

    bookmaker: Series[str] = Field(nullable=False)
    home_team: Series[str] = Field(nullable=False)
    away_team: Series[str] = Field(nullable=False)

    odd_home: Series[float] = Field(ge=1.01, nullable=False)
    odd_draw: Series[float] = Field(ge=1.01, nullable=False)
    odd_away: Series[float] = Field(ge=1.01, nullable=False)


# ========== VALIDADORES ==========
def validate_matches(df: pd.DataFrame) -> DataFrame[MatchesSchema]:
    # asegurar tipo de fecha antes de validar
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    out: DataFrame[MatchesSchema] = MatchesSchema.validate(df, lazy=True)
    if (out["home_team"] == out["away_team"]).any():
        raise ValueError("Hay filas con home_team == away_team.")
    return out


def validate_odds(df: pd.DataFrame) -> DataFrame[OddsSchema]:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    out: DataFrame[OddsSchema] = OddsSchema.validate(df, lazy=True)
    if (out["home_team"] == out["away_team"]).any():
        raise ValueError("Hay filas con home_team == away_team en odds.")
    return out
