from pathlib import Path
import pandas as pd

for sp in ["train", "valid", "test", "prod"]:
    p = Path(f"data/processed/features_{sp}.parquet")
    if not p.exists():
        print(sp, "→ falta archivo")
        continue
    df = pd.read_parquet(p)
    print(f"\n[{sp}] {df.shape}")
    print("cols ejemplo:", df.columns[:12].tolist())
    print("target distrib:", df["y"].value_counts(normalize=True).round(3).to_dict())
    if {"odd_home", "odd_draw", "odd_away"}.issubset(df.columns):
        print(
            "tiene odds:",
            df[["odd_home", "odd_draw", "odd_away"]].notna().mean().round(3).to_dict(),
        )
