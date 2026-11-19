# Oil Yield Predictor for Catalytic Plastic Pyrolysis

Streamlit app and Python code that estimate the liquid oil yield (wt%) of a catalytic plastic pyrolysis run
from the plastic feed, process conditions, reactor and catalyst. Model: CatBoost (0.4) + LightGBM (0.6).

**Contributors:** Soham Tanaji Raut (230008031), Srajan Kewat (230008033)
**Supervisor:** Dr. Rajan Singh, Discipline of Chemical Engineering, IIT Indore

## Files

| Path | Purpose |
|---|---|
| `app/app.py` | Streamlit web app |
| `src/features.py` | data cleaning and model inputs |
| `src/train.py` | retrains the model |
| `src/predict.py` | predicts every row of an Excel/CSV file |
| `src/config.py` | file paths and model settings |
| `models/oil_yield_model_FINAL.joblib` | trained model |
| `data/raw/` | training data and unseen-paper test data |
| `data/Test_rows_*.xlsx` | example inputs for the app |

## Run on your computer (Python 3.10–3.13)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app/app.py
```

Other commands:

```bash
python -m src.train                                              # retrain (about 1 min)
python -m src.predict data/raw/Validation_Unseen_Dataset_v001.xlsx   # batch prediction
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to share.streamlit.io, sign in with GitHub and click **Create app**.
3. Repository: this repo, branch: `main`, main file path: `app/app.py`.
4. Click **Deploy**.

## License

MIT (see `LICENSE`).
