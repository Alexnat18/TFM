# train_model.py
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

INPUT_CSV = "train_games.csv"
MODEL_OUT = "winprob_model.joblib"

df = pd.read_csv(INPUT_CSV)

# Features: totes les *_diff
FEATURES = [c for c in df.columns if c.endswith("_diff")]
if len(FEATURES) == 0:
    raise ValueError("No hi ha features *_diff. Has generat bé train_games.csv?")

if "win" not in df.columns:
    raise ValueError("Falta la columna 'win' a train_games.csv")

# Si encara és massa petit, para i avisa
n = len(df)
if n < 10:
    raise ValueError(f"train_games.csv massa petit: {n} files. Revisa build_train_games.py / merges.")

X = df[FEATURES].astype(float)
y = df["win"].astype(int)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000))
])

# Validació temporal (n_splits dinàmic)
n_splits = min(5, n - 1)   # evita errors si hi ha poques files
tscv = TimeSeriesSplit(n_splits=n_splits)

aucs, briers = [], []

for tr, te in tscv.split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    aucs.append(roc_auc_score(y.iloc[te], p))
    briers.append(brier_score_loss(y.iloc[te], p))

print(f"N samples:  {n}")
print(f"N splits:   {n_splits}")
print(f"AUC mean:   {np.mean(aucs):.3f}")
print(f"Brier mean: {np.mean(briers):.3f}")

# Fit final + save pack
model.fit(X, y)
joblib.dump({"model": model, "features": FEATURES}, MODEL_OUT)
print("✅ Guardat:", MODEL_OUT)
