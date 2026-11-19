"""
Predicts oil yield for every row of an Excel or CSV file in the training-data column format.

Run from the project folder:
    python -m src.predict data/raw/Validation_Unseen_Dataset_v001.xlsx
    python -m src.predict my_experiments.xlsx --out reports/my_predictions.xlsx

A row without catalyst that has a measured oil yield is used as the reference run for the other rows
with the same Group_ID and the same plastic.
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src import config
from src.features import INPUT_COLUMNS, TARGET, engineer, predict, prepare

warnings.filterwarnings("ignore")


def read_excel_data(path, sheet=None):
    """Read the given sheet, or the first sheet that contains the model's input columns."""
    if sheet is not None:
        return pd.read_excel(path, sheet_name=sheet)
    for name, df in pd.read_excel(path, sheet_name=None).items():
        if sum(c in df.columns for c in INPUT_COLUMNS) >= len(INPUT_COLUMNS) // 2:
            print(f"Using sheet '{name}'")
            return df
    raise ValueError("No sheet with the dataset columns was found; use --sheet to choose one.")


def main():
    parser = argparse.ArgumentParser(description="Predict oil yield for a file of experiments")
    parser.add_argument("file", help="Excel (.xlsx) or CSV file")
    parser.add_argument("--sheet", default=None,
                        help="sheet name for Excel files (default: the first sheet that has the data columns)")
    parser.add_argument("--out", default=None, help="output Excel file (default: reports/<name>_predictions.xlsx)")
    args = parser.parse_args()

    path = Path(args.file)
    data = pd.read_csv(path) if path.suffix.lower() == ".csv" else read_excel_data(path, args.sheet)
    bundle = joblib.load(config.MODEL_FILE)

    d = prepare(data)
    X = engineer(d)
    data["Predicted Oil Yield (wt%)"] = predict(bundle, X).round(2)
    data["Reference run used"] = np.where(X["has_thermal_baseline"] == 1, "yes", "no")

    actual = d[TARGET].values
    known = np.isfinite(actual)
    print(f"Predicted {len(data)} rows ({int(X['has_thermal_baseline'].sum())} with a reference run).")
    if known.sum() >= 3:
        p = data["Predicted Oil Yield (wt%)"].values
        print(f"Rows with an actual value: {known.sum()} | R2 = {r2_score(actual[known], p[known]):.3f} | "
              f"MAE = {mean_absolute_error(actual[known], p[known]):.2f} wt%")

    out = Path(args.out) if args.out else config.REPORTS / f"{path.stem}_predictions.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_excel(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
