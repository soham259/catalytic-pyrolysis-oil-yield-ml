"""
Trains the final oil-yield model and evaluates it on the unseen papers.

Uses the tuned settings in src/config.py instead of repeating the full model search.

Run from the project folder:
    python -m src.train          # train + unseen-paper test (about 1 minute)
    python -m src.train --cv     # also run source-aware cross-validation (about 5-10 minutes)
"""
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.features import CATS, GROUP, TARGET, clean, engineer, predict, remove_compiled_duplicates

warnings.filterwarnings("ignore")


def load_data():
    raw = pd.read_excel(config.TRAIN_FILE, sheet_name="ML_Dataset")
    raw_u = pd.read_excel(config.UNSEEN_FILE, sheet_name="Validation_Dataset")
    df, log = clean(raw)
    df, n_dup = remove_compiled_duplicates(df, config.COMPILED_GROUPS)
    log["after cross-group duplicate removal"] = len(df)
    uns, _ = clean(raw_u)
    for k, v in log.items():
        print(f"  {k:40s} {v}")
    return df, uns


def fit(X, y):
    """Train CatBoost and LightGBM on the given rows and return the model bundle."""
    levels = {c: list(pd.Categorical(X[c]).categories) for c in CATS}
    cb = CatBoostRegressor(**config.CB_PARAMS, cat_features=CATS).fit(X, y)
    Xl = X.copy()
    for c in CATS:
        Xl[c] = pd.Categorical(Xl[c], categories=levels[c])
    lg = lgb.LGBMRegressor(**config.LGB_PARAMS).fit(Xl, y)
    return dict(models={"CatBoost": cb, "LightGBM": lg}, weights=config.WEIGHTS, features=list(X.columns),
                cats=CATS, cat_levels=levels, cb_params=config.CB_PARAMS, lgb_params=config.LGB_PARAMS)


def scores(y, p):
    return f"R2 = {r2_score(y, p):.3f} | MAE = {mean_absolute_error(y, p):.2f} wt%"


def source_aware_cv(df, X, y):
    """5-fold StratifiedGroupKFold by paper, 3 seeds; outlier rule applied to training folds only."""
    g = df[GROUP].values
    catalytic = X["has_catalyst"].values == 1
    r_all, r_cat = [], []
    for seed in config.SEEDS:
        bins = pd.qcut(df[TARGET], 5, labels=False, duplicates="drop")
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(df, bins, groups=g):
            assert not set(g[tr]) & set(g[te])                      # no paper on both sides
            keep = config.keep_for_training(y[tr])
            oof[te] = predict(fit(X.iloc[tr][keep], y[tr][keep]), X.iloc[te])
        r_all.append(r2_score(y, oof)); r_cat.append(r2_score(y[catalytic], oof[catalytic]))
        print(f"  seed {seed}: R2 all = {r_all[-1]:.3f}, catalytic = {r_cat[-1]:.3f}")
    print(f"  mean: R2 all = {np.mean(r_all):.3f} ± {np.std(r_all, ddof=1):.3f}, catalytic = {np.mean(r_cat):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train the oil-yield model")
    parser.add_argument("--cv", action="store_true", help="also run source-aware cross-validation")
    args = parser.parse_args()

    print("1. Loading and cleaning data")
    df, uns = load_data()

    print("2. Building the 47 model inputs")
    X, XU = engineer(df), engineer(uns)
    y, yu = df[TARGET].values, uns[TARGET].values
    print(f"  training {X.shape}, unseen {XU.shape}")

    if args.cv:
        print("3. Source-aware cross-validation")
        source_aware_cv(df, X, y)

    print("4. Training the final model")
    keep = config.keep_for_training(y)
    bundle = fit(X[keep], y[keep])
    config.MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, config.MODEL_FILE)
    print(f"  trained on {keep.sum()} experiments, saved to {config.MODEL_FILE.relative_to(config.ROOT)}")

    print("5. Test on the 8 unseen papers")
    pu = predict(bundle, XU)
    base = XU["has_thermal_baseline"].values == 1
    print(f"  all {len(yu)} experiments:           {scores(yu, pu)}")
    print(f"  with reference run ({base.sum()}):       {scores(yu[base], pu[base])}")

    out = uns[[GROUP, "Research_Paper_Name"]].copy()
    out["Actual oil (wt%)"] = yu
    out["Predicted oil (wt%)"] = pu.round(2)
    out["Reference run used"] = np.where(base, "yes", "no")
    config.REPORTS.mkdir(exist_ok=True)
    out.to_excel(config.REPORTS / "unseen_predictions.xlsx", index=False)
    print("  predictions saved to reports/unseen_predictions.xlsx")


if __name__ == "__main__":
    main()
