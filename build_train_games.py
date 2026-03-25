# build_train_games.py
import pandas as pd
import numpy as np

INPUT_CSV = "df_jornades_all.csv"
OUTPUT_CSV = "train_games.csv"

# 1) Load
df = pd.read_csv(INPUT_CSV)

# 2) Jornada_num
if "Jornada_num" not in df.columns:
    if "Jornada" in df.columns:
        df["Jornada_num"] = (
            df["Jornada"].astype(str).str.replace("J", "", regex=False).astype(int)
        )
    else:
        raise ValueError("Falta 'Jornada_num' o 'Jornada'.")

# 3) Check columnes clau (ara amb TEAM_id)
required = ["TEAM_id", "Rival_idR", "Jornada_num", "Resultat"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Falten columnes: {missing}")

# 4) Target
df["win"] = (df["Resultat"].astype(str).str.upper() == "W").astype(int)

# 5) Mètriques disponibles
metrics = ["NET", "OER", "DER", "eFG", "ORB", "AST", "TOV", "pos_part"]
metrics = [m for m in metrics if m in df.columns]
if len(metrics) == 0:
    raise ValueError("No he trobat cap mètrica (NET/OER/DER/eFG/ORB/AST/TOV/pos_part).")

# 6) Tipus consistents
df["TEAM_id"] = df["TEAM_id"].astype(str).str.strip()
df["Rival_idR"] = df["Rival_idR"].astype(str).str.strip()
df["Jornada_num"] = pd.to_numeric(df["Jornada_num"], errors="coerce")

# Treu files sense IDs / jornada
df = df.dropna(subset=["TEAM_id", "Rival_idR", "Jornada_num"]).copy()
df["Jornada_num"] = df["Jornada_num"].astype(int)

# IMPORTANT: jornada 1 no té històric (pre = NaN) -> fora
df = df[df["Jornada_num"] >= 2].copy()

# 7) Ordena per equip i jornada
df = df.sort_values(["TEAM_id", "Jornada_num"]).copy()

# 8) Features pre-partit (evitar leakage): mitjana acumulada fins t-1
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")
    df[f"{m}_pre"] = (
        df.groupby("TEAM_id")[m]
          .transform(lambda s: s.shift(1).expanding().mean())
    )

# 9) Dataset A (team_a) i dataset B (team_b) amb mètriques pre
A = df.rename(columns={"TEAM_id": "team_a", "Rival_idR": "team_b"}).copy()

B_cols = ["TEAM_id", "Jornada_num"] + [f"{m}_pre" for m in metrics]
B = df[B_cols].rename(columns={"TEAM_id": "team_b"}).copy()
B = B.add_prefix("B_")
B = B.rename(columns={"B_team_b": "team_b", "B_Jornada_num": "Jornada_num"})

# 10) Merge per obtenir stats pre del rival a la mateixa jornada
train = A.merge(B, on=["team_b", "Jornada_num"], how="left")

# 11) Diffs (A - B)
for m in metrics:
    train[f"{m}_diff"] = train[f"{m}_pre"] - train[f"B_{m}_pre"]

# 12) Selecció final + dropna “intel·ligent”
keep = ["Jornada_num", "team_a", "team_b", "win"] + [f"{m}_diff" for m in metrics]
diff_cols = [f"{m}_diff" for m in metrics]

train_games = (
    train[keep]
    .dropna(subset=diff_cols + ["win"])  # no eliminis per altres NAs irrellevants
    .sort_values("Jornada_num")
    .reset_index(drop=True)
)

# 13) Guarda
train_games.to_csv(OUTPUT_CSV, index=False)
print("✅ Creat", OUTPUT_CSV, "->", train_games.shape)
print("Columns:", train_games.columns.tolist())

# Diagnòstic extra si queda buit
if len(train_games) == 0:
    print("\n[DIAGNÒSTIC]")
    print("Rows df (després filtres):", len(df))
    print("Rows train (després merge):", len(train))
    if "B_NET_pre" in train.columns:
        print("NA rate B_NET_pre:", train["B_NET_pre"].isna().mean())

    # ids de rival que NO existeixen com a TEAM_id (això trenca el merge)
    all_teams = set(df["TEAM_id"].unique())
    miss = [x for x in train["team_b"].dropna().unique() if x not in all_teams]
    print("Exemple team_b que no matxen TEAM_id (max 20):", miss[:20])

