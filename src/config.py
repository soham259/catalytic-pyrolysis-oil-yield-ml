"""Paths and model settings used by the scripts and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # project folder

TRAIN_FILE = ROOT / "data" / "raw" / "Final_Source_Aware_ML_Dataset_v050_Batch49.xlsx"
UNSEEN_FILE = ROOT / "data" / "raw" / "Validation_Unseen_Dataset_v001.xlsx"
MODEL_FILE = ROOT / "models" / "oil_yield_model_FINAL.joblib"
REPORTS = ROOT / "reports"

SEEDS = [42, 7, 2026]                        # source-aware cross-validation seeds
COMPILED_GROUPS = [4, 5, 50, 53, 88, 92]     # literature-compilation groups checked for duplicates

# Tuned settings found by random search
CB_PARAMS = dict(iterations=1200, learning_rate=0.05, depth=8, l2_leaf_reg=12, rsm=1.0, random_strength=1,
                 bagging_temperature=0.5, verbose=0, random_seed=42, thread_count=-1)
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=60, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=0, verbose=-1, random_state=42)
WEIGHTS = {"CatBoost": 0.4, "LightGBM": 0.6}   # ensemble weights


def keep_for_training(oil):
    """Outlier rule: train only on experiments with 1 < oil < 99 wt%."""
    return (oil > 1) & (oil < 99)
