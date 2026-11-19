"""
Data preparation and feature engineering for the oil-yield model.

Shared by the training script, the prediction script and the app, so all three build exactly
the 47 inputs the model was trained on.
"""
import re
import numpy as np
import pandas as pd

TARGET = "Oil Yield (wt%)"
GROUP = "Group_ID"

NUM_RAW = ["HDPE (wt%)", "LDPE (wt%)", "LLDPE (wt%)", "PP (wt%)", "PS (wt%)", "PVC (wt%)", "PET (wt%)",
           "Heating Rate (C/min)", "Reaction Temperature (C)", "Reaction Time (min)", "Plastic Particle Size (mm)",
           "Catalyst Loading (wt%)", "Plastic/Catalyst Ratio (P/C)", "Total Acidity (umol/g)", "Si/Al Ratio", "SBET (m2/g)"]
CAT_RAW = ["Plastic Source (Virgin/Waste)", "Operation Mode", "Reactor Type", "Catalytic Mode (In-situ/Ex-situ)"]
COMP = NUM_RAW[:7]
INPUT_COLUMNS = [GROUP, "Research_Paper_Name"] + COMP + ["Plastic Source (Virgin/Waste)", "Heating Rate (C/min)",
                 "Reaction Temperature (C)", "Reaction Time (min)", "Plastic Particle Size (mm)", "Operation Mode",
                 "Reactor Type", "Catalyst Loading (wt%)", "Plastic/Catalyst Ratio (P/C)", "Total Acidity (umol/g)",
                 "Si/Al Ratio", "SBET (m2/g)", "Catalytic Mode (In-situ/Ex-situ)", TARGET]

CATS = ["Plastic Source (Virgin/Waste)", "Operation Mode", "Reactor Family", "Catalytic Mode (In-situ/Ex-situ)"]

CATALYST_FAMILIES = {
    "cf_ZSM5": r"zsm-?5|\bmfi\b|hzsm|h-zsm|zsm-?11",
    "cf_Y": r"\busy\b|faujasite|\by[- ]zeolite|zeolite[- ]y\b|\bhy\b|\bnay\b|\bh-y\b|\bus-y\b|\bre-?y\b|cbv ?[0-9]",
    "cf_beta": r"\bbeta\b|\bbea\b|hβ|h-β",
    "cf_mordenite": r"mordenite|\bmor\b",
    "cf_FCC": r"\bfcc\b|equilibrium catalyst|\be-?cat\b",
    "cf_mesoporous": r"mcm-?41|sba-?15|kit-?6|mcm-?48|mesopor",
    "cf_clay": r"clay|bentonite|kaolin|montmorillonite|\bk10\b|illite|halloysite|sepiolite|pillared|attapulgite",
    "cf_silica_alumina": r"silica[- ]alumina|sio2[-/]al2o3|amorphous",
    "cf_alumina": r"alumina|al2o3",
    "cf_metal_oxide": r"\bcao\b|\bmgo\b|\bzno\b|tio2|fe2o3|\bcuo\b|ceo2|zro2|metal[- ]oxide|\bnio\b|co3o4|red mud|bauxite|dolomite|limestone|calcium oxide",
    "cf_carbon": r"activated carbon|biochar|carbon",
    "cf_natural_zeolite": r"natural zeolite|clinoptilolite|\bnz\b|\bta-nz\b",
    "cf_metal_loaded": r"\b(?:ni|fe|co|cu|zn|ga|pt|pd|ru|mo|ce|la)\s?[/-]|nickel|\bgallo|metal[- ]loaded|metal[- ]incorporated|impregnat"}

# Catalyst choices shown in the app -> text that the family patterns above recognise
CATALYST_CHOICES = {
    "ZSM-5 / HZSM-5": "zsm-5", "Y / USY zeolite": "hy zeolite", "Beta zeolite": "beta", "Mordenite": "mordenite",
    "FCC catalyst": "fcc", "Natural zeolite": "natural zeolite", "Mesoporous silica (MCM-41, SBA-15, KIT-6)": "mcm-41",
    "Clay (bentonite, kaolin, montmorillonite)": "clay", "Silica-alumina": "silica-alumina", "Alumina": "alumina",
    "Metal oxide (CaO, MgO, red mud ...)": "metal oxide", "Activated carbon / biochar": "activated carbon",
    "Metal-loaded (Ni, Fe, Ga ...)": "metal-loaded"}

# Reactor choices shown in the app -> text that reactor_family() maps to the right family
REACTOR_CHOICES = {"Fixed bed / tubular": "fixed bed", "Batch reactor": "batch reactor",
                   "Stirred tank / autoclave": "stirred autoclave", "Fluidized / spouted bed": "fluidized bed",
                   "Microwave reactor": "microwave", "Other": "other reactor", "Not reported": "Unknown"}


def to_num(value):
    """'450-500' -> 475 (midpoint), '<2' -> 2, other text -> NaN."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"^[<>~≈≤≥]=?\s*", "", s)
    if re.fullmatch(r"-?\d+(\.\d+)?\s*-\s*\d+(\.\d+)?", s):
        a, b = [float(x) for x in s.split("-") if x.strip() != ""][:2]
        return (a + b) / 2
    try:
        return float(s)
    except ValueError:
        return np.nan


def prepare(d):
    """Same conversions as clean(); rows without an oil yield are kept (they are predicted)."""
    d = d.copy()
    for c in INPUT_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    for c in NUM_RAW:
        d[c] = d[c].map(to_num).astype(float)
    for c in CAT_RAW:
        d[c] = d[c].astype("object").where(d[c].notna(), "Unknown").astype(str).str.strip()
    d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce")
    d[GROUP] = d[GROUP].fillna(-1)
    d["Research_Paper_Name"] = d["Research_Paper_Name"].fillna("")
    return d.reset_index(drop=True)


def reactor_family(s):
    s = str(s).lower()
    if "fluid" in s or "spouted" in s: return "fluidized/spouted"
    if "microwave" in s: return "microwave"
    if any(k in s for k in ["fixed", "packed", "tube", "tubular"]): return "fixed/tube"
    if any(k in s for k in ["stir", "autoclave", "stl", "pressure"]): return "stirred/autoclave"
    if any(k in s for k in ["batch", "micro", "flask", "pot", "reactor w/"]): return "batch"
    return "Unknown" if s == "unknown" else "other"


def op_family(s):
    s = str(s).lower()
    return "Semi-Batch" if "semi" in s else "Continuous" if "contin" in s else "Batch" if ("batch" in s or "pot" in s) else "Unknown"


def engineer(d):
    """Builds the 47 model inputs."""
    X = pd.DataFrame(index=d.index)
    comp = d[COMP]; comp0 = comp.fillna(0)
    for c in COMP: X[c] = comp[c]
    X["comp_known"] = comp.notna().any(axis=1).astype(int)
    X["n_polymers"] = (comp0 > 0).sum(axis=1)
    X["comp_sum"] = comp0.sum(axis=1)
    X["max_component"] = comp0.max(axis=1)
    for c in ["Heating Rate (C/min)", "Reaction Temperature (C)", "Reaction Time (min)", "Plastic Particle Size (mm)",
              "Total Acidity (umol/g)", "Si/Al Ratio", "SBET (m2/g)"]:
        X[c] = d[c]
    load = d["Catalyst Loading (wt%)"].where(d["Catalyst Loading (wt%)"].notna(),
                                             100 / d["Plastic/Catalyst Ratio (P/C)"].replace(0, np.nan))
    X["cat_loading_unified"] = load
    has_cat = ((load.fillna(0) > 0) | d["Catalytic Mode (In-situ/Ex-situ)"].isin(["In-situ", "Ex-situ"])
               | d[["Total Acidity (umol/g)", "Si/Al Ratio", "SBET (m2/g)"]].notna().any(axis=1))
    X["has_catalyst"] = has_cat.astype(int)
    X["acid_x_load"] = d["Total Acidity (umol/g)"] * load / 100
    X["sbet_x_load"] = d["SBET (m2/g)"] * load / 100
    X["acid_per_area"] = d["Total Acidity (umol/g)"] / d["SBET (m2/g)"].replace(0, np.nan)
    X["T_x_logtime"] = d["Reaction Temperature (C)"] * np.log1p(d["Reaction Time (min)"])
    X["n_catalyst_desc"] = d[["Total Acidity (umol/g)", "Si/Al Ratio", "SBET (m2/g)"]].notna().sum(axis=1)
    X["n_missing_process"] = d[["Heating Rate (C/min)", "Reaction Time (min)", "Plastic Particle Size (mm)"]].isna().sum(axis=1)
    X["Plastic Source (Virgin/Waste)"] = d["Plastic Source (Virgin/Waste)"]
    X["Operation Mode"] = d["Operation Mode"].map(op_family)
    X["Reactor Family"] = d["Reactor Type"].map(reactor_family)
    mode = d["Catalytic Mode (In-situ/Ex-situ)"].copy(); mode[~has_cat] = "No catalyst"
    X["Catalytic Mode (In-situ/Ex-situ)"] = mode
    text = d["Research_Paper_Name"].astype(str).str.lower()
    for k, pattern in CATALYST_FAMILIES.items():
        X[k] = text.str.contains(pattern, regex=True).astype(int) * X["has_catalyst"]
    X["cf_n_families"] = X[list(CATALYST_FAMILIES)].sum(axis=1)

    # thermal baseline: same group, no-catalyst run, same feed (within 1 wt%), nearest temperature, never itself
    comp_v = comp0.values; temp = d["Reaction Temperature (C)"].values
    grp = d[GROUP].values; oil = d[TARGET].values
    thermal = (X["has_catalyst"].values == 0) & np.isfinite(oil)
    base = np.full(len(d), np.nan); dT = np.full(len(d), np.nan)
    for i in range(len(d)):
        cand = [j for j in np.where(thermal & (grp == grp[i]) & (np.arange(len(d)) != i))[0]
                if np.abs(comp_v[j] - comp_v[i]).max() <= 1.0]
        if not cand:
            continue
        gaps = [abs(temp[j] - temp[i]) if not (np.isnan(temp[j]) or np.isnan(temp[i])) else 1e9 for j in cand]
        k = int(np.argmin(gaps))
        base[i] = oil[cand[k]]
        dT[i] = (temp[i] - temp[cand[k]]) if gaps[k] < 1e9 else np.nan
    X["thermal_baseline_oil"] = base
    X["dT_vs_baseline"] = dT
    X["has_thermal_baseline"] = np.isfinite(base).astype(int)
    return X


def predict(bundle, X):
    """Weighted CatBoost + LightGBM prediction, clipped to 0-100 wt%."""
    X = X[bundle["features"]]
    parts = {}
    if "CatBoost" in bundle["models"]:
        parts["CatBoost"] = bundle["models"]["CatBoost"].predict(X)
    if "LightGBM" in bundle["models"]:
        Xl = X.copy()
        for c in bundle["cats"]:
            Xl[c] = pd.Categorical(Xl[c], categories=bundle["cat_levels"][c])
        parts["LightGBM"] = bundle["models"]["LightGBM"].predict(Xl)
    return sum(bundle["weights"][k] * np.clip(v, 0, 100) for k, v in parts.items())


# ----------------------------------------------------------------------------- cleaning of the training data
def clean(d):
    """Convert text to numbers, drop rows without oil yield, check range, drop exact copies."""
    d = prepare(d)
    log = {"rows in": len(d)}
    d = d[d[TARGET].notna()];                         log["after dropping missing target"] = len(d)
    d = d[(d[TARGET] >= 0) & (d[TARGET] <= 100)];    log["after target range check 0-100"] = len(d)
    d = d.drop_duplicates(subset=NUM_RAW + CAT_RAW + [TARGET, GROUP]); log["after exact-duplicate removal"] = len(d)
    return d.reset_index(drop=True), log


def remove_compiled_duplicates(d, compiled_groups):
    """Drop rows of literature compilations that repeat an original-paper experiment."""
    comp = d[COMP].fillna(0).values
    temp = d["Reaction Temperature (C)"].values
    oil = d[TARGET].values
    load = d["Catalyst Loading (wt%)"].values
    grp = d[GROUP].values
    is_reference = ~np.isin(grp, compiled_groups)
    drop = []
    for i in np.where(np.isin(grp, compiled_groups))[0]:
        cand = np.where(is_reference & (grp != grp[i]) & (np.abs(oil - oil[i]) <= 0.3) & (np.abs(temp - temp[i]) <= 1.0))[0]
        cand = [j for j in cand
                if np.abs(comp[j] - comp[i]).max() <= 1.0
                and (np.isnan(load[i]) or np.isnan(load[j]) or abs(load[i] - load[j]) <= 0.5)]
        if cand:
            drop.append(i)
        else:
            is_reference[i] = True
    return d.drop(index=d.index[drop]).reset_index(drop=True), len(drop)
