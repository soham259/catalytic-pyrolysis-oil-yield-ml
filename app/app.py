"""
Oil Yield Predictor - Streamlit app for the catalytic plastic pyrolysis model.

Run from the project folder:  streamlit run app/app.py
Needs models/oil_yield_model_FINAL.joblib (created by `python -m src.train`).
"""
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project folder, so that `src` can be imported

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from catboost import Pool

from src import config
from src.features import (CATALYST_CHOICES, COMP, GROUP, INPUT_COLUMNS, REACTOR_CHOICES, TARGET,
                      engineer, predict, prepare)

st.set_page_config(page_title="Oil Yield Predictor", page_icon="🛢️", layout="wide")

# ----------------------------------------------------------------------------- style
TEAL, TEAL_DARK, AMBER, RED, GREEN, GREY = "#0f766e", "#134e4a", "#d97706", "#dc2626", "#16a34a", "#64748b"
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1350px;}}
.hero {{background: linear-gradient(120deg, {TEAL_DARK} 0%, {TEAL} 60%, #14b8a6 100%); color: white;
        padding: 1.4rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;}}
.hero h1 {{color: white; font-size: 1.9rem; margin: 0 0 .3rem 0; font-weight: 700;}}
.hero p {{margin: 0; opacity: .92; font-size: 1.0rem;}}
.card {{background: var(--secondary-background-color); border: 1px solid rgba(100,116,139,.18);
        border-radius: 14px; padding: 1rem 1.2rem; height: 100%;}}
.card .label {{font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: {GREY}; margin-bottom: .2rem;}}
.card .value {{font-size: 1.9rem; font-weight: 700; line-height: 1.15;}}
.card .sub {{font-size: .86rem; color: {GREY}; margin-top: .25rem;}}
.result {{border-radius: 18px; padding: 1.3rem 1.5rem; color: white;
          background: linear-gradient(135deg, {TEAL_DARK}, {TEAL});}}
.result .big {{font-size: 3.2rem; font-weight: 800; line-height: 1;}}
.result .unit {{font-size: 1.2rem; font-weight: 500; opacity: .85;}}
.result .range {{font-size: .98rem; opacity: .92; margin-top: .45rem;}}
.badge {{display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
         margin-top: .7rem; background: rgba(255,255,255,.18);}}
.section {{font-size: 1.05rem; font-weight: 700; margin: .4rem 0 .2rem 0;}}
div[data-testid="stExpander"] details summary p {{font-weight: 600;}}
</style>""", unsafe_allow_html=True)


def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def y_range(values, current):
    v = list(values) + [current]
    return [max(0, min(v) - 10), min(100, max(v) + 10)]


def style_fig(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      font=dict(size=13), legend=dict(orientation="h", y=-0.2), hoverlabel=dict(font_size=13))
    return fig


# ----------------------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_FILE)


try:
    bundle = load_model()
except FileNotFoundError:
    st.error("Model file `models/oil_yield_model_FINAL.joblib` not found. Run `python -m src.train` first.")
    st.stop()

# Final validation results, shown on the Model page
RESULTS = {"unseen_all": (0.312, 14.5, 73), "unseen_ref": (0.461, 12.4, 55), "cv_cat": (0.338, 15.1, 2447),
           "cv_with_ref": (0.478, 13.2), "cv_without_ref": (0.246, 16.6)}
JOURNEY = [("Random 80/20 split\n(papers shared)", 0.723), ("Random split model\non unseen papers", -0.215),
           ("Source-aware CV\nraw inputs", 0.191), ("+ engineered &\nbaseline features", 0.310),
           ("+ outlier rule,\ntuning, ensemble", 0.347), ("Final model\nunseen papers", 0.312),
           ("Final model, unseen\nwith reference run", 0.461)]

# ----------------------------------------------------------------------------- header
st.markdown("""<div class="hero"><h1>Oil Yield Predictor for Catalytic Plastic Pyrolysis</h1>
<p>Machine-learning model trained on 2,965 experiments from 304 published studies and validated on 8 unseen papers.
Enter your feedstock, process conditions and catalyst to estimate the liquid oil yield.</p></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.markdown("**Model:** weighted ensemble of CatBoost (0.4) and LightGBM (0.6)\n\n"
                "**Inputs:** 47 features built from feed, process, reactor and catalyst data\n\n"
                "**Validation:** source-aware (whole papers held out) and 8 unseen papers")
    st.markdown("### Best accuracy")
    st.info("Add the oil yield of one **no-catalyst run** on the same set-up. It tells the model the level of your "
            "reactor and condenser and raises R² on unseen papers from 0.31 to 0.46.")
    st.markdown("### Project")
    st.caption("B.Tech Project, Chemical Engineering, IIT Indore\n\nSynthesis and Application of Hierarchical "
               "Zeolite for Plastic Waste Pyrolysis")

tab_pred, tab_batch, tab_model = st.tabs(["Single prediction", "Batch prediction", "Model performance"])


NICE = {"thermal_baseline_oil": "Reference-run oil yield", "dT_vs_baseline": "Temperature gap to reference run",
        "has_thermal_baseline": "Reference run given", "cat_loading_unified": "Catalyst loading (wt%)",
        "T_x_logtime": "Temperature x log(time)", "acid_x_load": "Acidity x loading", "sbet_x_load": "SBET x loading",
        "acid_per_area": "Acid-site density", "n_polymers": "Number of polymers", "max_component": "Main polymer share",
        "comp_sum": "Composition total", "comp_known": "Composition reported", "has_catalyst": "Catalyst used",
        "n_catalyst_desc": "Catalyst properties given", "n_missing_process": "Process values missing",
        "cf_n_families": "Number of catalyst families", "cf_ZSM5": "ZSM-5", "cf_Y": "Y zeolite", "cf_beta": "Beta zeolite",
        "cf_mordenite": "Mordenite", "cf_FCC": "FCC", "cf_mesoporous": "Mesoporous silica", "cf_clay": "Clay",
        "cf_silica_alumina": "Silica-alumina", "cf_alumina": "Alumina", "cf_metal_oxide": "Metal oxide",
        "cf_carbon": "Carbon", "cf_natural_zeolite": "Natural zeolite", "cf_metal_loaded": "Metal-loaded"}
CAT_FLAGS = [k for k in NICE if k.startswith("cf_") and k != "cf_n_families"] + ["has_thermal_baseline", "has_catalyst", "comp_known"]


# ============================================================================= helpers
def experiment_row(inp, temperature=None, loading=None):
    """One row in the training-file format, built from the form inputs."""
    r = {c: np.nan for c in INPUT_COLUMNS}
    r.update({GROUP: 1, "Research_Paper_Name": inp["catalyst_text"], TARGET: np.nan})
    for c, v in inp["comp"].items():
        r[c] = v
    r.update({"Plastic Source (Virgin/Waste)": inp["source"], "Heating Rate (C/min)": inp["heating"],
              "Reaction Temperature (C)": inp["T"] if temperature is None else temperature,
              "Reaction Time (min)": inp["time"], "Plastic Particle Size (mm)": inp["particle"],
              "Operation Mode": inp["op_mode"], "Reactor Type": inp["reactor"],
              "Catalyst Loading (wt%)": inp["loading"] if loading is None else loading,
              "Total Acidity (umol/g)": inp["acidity"], "Si/Al Ratio": inp["sial"], "SBET (m2/g)": inp["sbet"],
              "Catalytic Mode (In-situ/Ex-situ)": inp["cat_mode"]})
    return r


def reference_row(inp):
    """The user's no-catalyst run on the same set-up (used for the thermal-baseline features)."""
    r = experiment_row(inp)
    r.update({"Research_Paper_Name": "", "Reaction Temperature (C)": inp["ref_T"], "Catalyst Loading (wt%)": np.nan,
              "Total Acidity (umol/g)": np.nan, "Si/Al Ratio": np.nan, "SBET (m2/g)": np.nan,
              "Catalytic Mode (In-situ/Ex-situ)": np.nan, TARGET: inp["ref_oil"]})
    return r


def features_for(inp, rows):
    """Engineer features for a list of experiment rows, adding the reference run when it is given."""
    frame = pd.DataFrame(rows + ([reference_row(inp)] if inp["use_ref"] else []))
    X = engineer(prepare(frame))
    return X.iloc[:len(rows)]


# ============================================================================= single prediction
with tab_pred:
    left, right = st.columns([1.05, 1.25], gap="large")

    with left:
        st.markdown('<div class="section">Experiment details</div>', unsafe_allow_html=True)

        with st.expander("1. Feedstock", expanded=True):
            feed = st.selectbox("Plastic", ["LDPE", "HDPE", "LLDPE", "PP", "PS", "PET", "Mixture (enter wt%)"])
            comp = {c: 0.0 for c in COMP}          # same as the training data: other polymers stored as 0
            if feed.startswith("Mixture"):
                cols = st.columns(4)
                for i, c in enumerate(COMP):
                    comp[c] = cols[i % 4].number_input(c.replace(" (wt%)", " wt%"), 0.0, 100.0, 0.0, 5.0, key=c)
                total = sum(comp.values())
                if abs(total - 100) > 1:
                    st.warning(f"Composition adds up to {total:.0f} wt% (should be about 100).")
            else:
                comp[f"{feed} (wt%)"] = 100.0
            source = st.radio("Plastic source", ["Waste", "Virgin", "Unknown"], horizontal=True)

        with st.expander("2. Process conditions", expanded=True):
            T = st.slider("Reaction temperature (°C)", 300, 750, 450, 5)
            c1, c2 = st.columns(2)
            time_min = c1.number_input("Reaction time (min)", 1.0, 600.0, 60.0, 5.0)
            heating = c2.number_input("Heating rate (°C/min)", 1.0, 1000.0, value=None, step=1.0,
                                      placeholder="not reported")
            c3, c4 = st.columns(2)
            reactor = c3.selectbox("Reactor", list(REACTOR_CHOICES), index=1)
            op_mode = c4.selectbox("Operation mode", ["Batch", "Semi-Batch", "Continuous", "Unknown"])
            particle = st.number_input("Plastic particle size (mm)", 0.01, 100.0, value=None, step=0.5,
                                       placeholder="not reported")

        with st.expander("3. Catalyst", expanded=True):
            catalysts = st.multiselect("Catalyst type", list(CATALYST_CHOICES), default=["ZSM-5 / HZSM-5"])
            cat_mode = st.radio("Contact mode", ["In-situ", "Ex-situ"], horizontal=True)
            amount_as = st.radio("Catalyst amount given as", ["Loading (wt% of plastic)", "Plastic/catalyst ratio (P/C)"],
                                 horizontal=True)
            if amount_as.startswith("Loading"):
                loading = st.number_input("Catalyst loading (wt% of plastic)", 0.1, 200.0, 10.0, 1.0)
            else:
                pc = st.number_input("Plastic/catalyst ratio (g plastic per g catalyst)", 0.5, 1000.0, 10.0, 1.0)
                loading = 100.0 / pc                     # same conversion as in the model: loading = 100 / (P/C)
                st.caption(f"Equivalent catalyst loading: {loading:.2f} wt% of plastic")
            st.caption("Catalyst properties (leave empty if not measured)")
            c1, c2, c3 = st.columns(3)
            sbet = c1.number_input("SBET (m²/g)", 1.0, 2000.0, value=309.46, step=10.0, placeholder="not measured")
            sial = c2.number_input("Si/Al ratio", 0.1, 1000.0, value=None, step=0.1, placeholder="not measured")
            acidity = c3.number_input("Total acidity (µmol/g)", 1.0, 5000.0, value=None, step=10.0,
                                      placeholder="not measured")

        with st.expander("4. No-catalyst reference run (recommended)", expanded=True):
            use_ref = st.toggle("I have a run without catalyst on the same set-up and plastic", value=False)
            c1, c2 = st.columns(2)
            ref_T = c1.number_input("Its temperature (°C)", 250, 900, T, 5, disabled=not use_ref)
            ref_oil = c2.number_input("Its oil yield (wt%)", 0.0, 100.0, 60.0, 0.5, disabled=not use_ref)

    inp = dict(comp=comp, source=source, T=T, time=time_min, heating=heating if heating is not None else np.nan,
               particle=particle if particle is not None else np.nan, reactor=REACTOR_CHOICES[reactor],
               op_mode=op_mode, loading=loading, cat_mode=cat_mode,
               sbet=sbet if sbet is not None else np.nan, sial=sial if sial is not None else np.nan,
               acidity=acidity if acidity is not None else np.nan,
               catalyst_text=" ".join(CATALYST_CHOICES[c] for c in catalysts),
               use_ref=use_ref, ref_T=ref_T, ref_oil=ref_oil)

    X1 = features_for(inp, [experiment_row(inp)])
    pred = float(predict(bundle, X1)[0])
    mae = RESULTS["unseen_ref"][1] if use_ref else RESULTS["cv_without_ref"][1]
    lo, hi = max(0, pred - mae), min(100, pred + mae)

    with right:
        st.markdown('<div class="section">Predicted result</div>', unsafe_allow_html=True)
        conf = ("Higher confidence: reference run used (R² ≈ 0.46 on unseen papers)" if use_ref
                else "Moderate confidence: no reference run (add one for a better estimate)")
        st.markdown(f"""<div class="result"><div style="opacity:.85">Predicted oil yield</div>
            <div><span class="big">{pred:.1f}</span> <span class="unit">wt%</span></div>
            <div class="range">Typical error range: {lo:.0f} – {hi:.0f} wt% (± {mae:.1f} wt%, mean absolute error)</div>
            <span class="badge">{conf}</span></div>""", unsafe_allow_html=True)
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            card("Gas + solid", f"{100 - pred:.0f} wt%", "remainder of the mass balance")
        with k2:
            if use_ref:
                d = pred - ref_oil
                card("Catalyst effect", f"{d:+.1f} wt%", f"vs. {ref_oil:.1f} wt% without catalyst")
            else:
                card("Catalyst effect", "—", "needs the reference run")
        with k3:
            card("Catalyst families", f"{int(X1['cf_n_families'].iloc[0])}",
                 ", ".join(c.split(' (')[0] for c in catalysts) if catalysts else "none selected")

        notes = []
        if not catalysts:
            notes.append("No catalyst type selected: the model uses only the loading and properties.")
        if T < 350 or T > 650:
            notes.append("Temperature is outside 350–650 °C, where most training experiments lie.")
        if use_ref and abs(ref_T - T) > 50:
            notes.append("The reference run is more than 50 °C away from this experiment; the estimate is less certain.")
        for n in notes:
            st.warning(n)

        st.markdown('<div class="section">What-if analysis</div>', unsafe_allow_html=True)
        w1, w2 = st.tabs(["Temperature", "Catalyst loading"])
        with w1:
            temps = list(range(350, 655, 10))
            yt = predict(bundle, features_for(inp, [experiment_row(inp, temperature=t) for t in temps]))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=temps, y=yt, mode="lines+markers", line=dict(color=TEAL, width=3),
                                     name="Predicted oil yield", hovertemplate="%{x} °C: %{y:.1f} wt%<extra></extra>"))
            fig.add_trace(go.Scatter(x=[T], y=[pred], mode="markers", marker=dict(size=14, color=AMBER),
                                     name="Current input", hovertemplate="%{x} °C: %{y:.1f} wt%<extra></extra>"))
            fig.update_xaxes(title="Reaction temperature (°C)"); fig.update_yaxes(title="Oil yield (wt%)", range=y_range(yt, pred))
            st.plotly_chart(style_fig(fig), width="stretch")
            best_t = temps[int(np.argmax(yt))]
            st.caption(f"Highest predicted oil yield in this range: {yt.max():.1f} wt% at {best_t} °C, "
                       "with all other inputs kept the same. Tree-based models give step-like curves; "
                       "read the overall trend rather than single steps.")
        with w2:
            loads = [1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50]
            yl = predict(bundle, features_for(inp, [experiment_row(inp, loading=l) for l in loads]))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=loads, y=yl, mode="lines+markers", line=dict(color=TEAL, width=3),
                                     name="Predicted oil yield", hovertemplate="%{x} wt%: %{y:.1f} wt% oil<extra></extra>"))
            fig.add_trace(go.Scatter(x=[loading], y=[pred], mode="markers", marker=dict(size=14, color=AMBER),
                                     name="Current input"))
            fig.update_xaxes(title="Catalyst loading (wt% of plastic)", type="log")
            fig.update_yaxes(title="Oil yield (wt%)", range=y_range(yl, pred))
            st.plotly_chart(style_fig(fig), width="stretch")

        st.markdown('<div class="section">What drives this prediction</div>', unsafe_allow_html=True)
        cb = bundle["models"]["CatBoost"]
        sv = cb.get_feature_importance(Pool(X1[bundle["features"]], cat_features=bundle["cats"]), type="ShapValues")[0]
        contrib = pd.Series(sv[:-1], index=bundle["features"])
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
        def label(f):
            v = X1[f].iloc[0]
            name = NICE.get(f, f.replace("_", " "))
            if f in CAT_FLAGS:
                return f"{name}: {'yes' if v == 1 else 'no'}"
            if isinstance(v, str):
                return f"{name}: {v}"
            return f"{name}: not given" if pd.isna(v) else f"{name}: {v:g}"
        fig = go.Figure(go.Bar(x=top.values, y=[label(f) for f in top.index], orientation="h",
                               marker_color=[GREEN if v > 0 else RED for v in top.values],
                               hovertemplate="%{y}: %{x:+.1f} wt%<extra></extra>"))
        fig.add_vline(x=0, line_color=GREY)
        fig.update_xaxes(title="Effect on predicted oil yield (wt%)")
        st.plotly_chart(style_fig(fig, 320), width="stretch")
        st.caption(f"SHAP values of the CatBoost part of the ensemble. Green raises and red lowers the yield "
                   f"compared with the average prediction ({sv[-1]:.1f} wt%).")


# ============================================================================= batch prediction
with tab_batch:
    st.markdown("Upload an Excel or CSV file in the same column format as the training dataset. Every row is predicted. "
                "A row **without catalyst that has an oil yield** acts as the reference run for the other rows with the "
                "same `Group_ID` and the same plastic. If actual oil yields are given, the accuracy is shown as well.")
    template = pd.DataFrame([{**{c: np.nan for c in INPUT_COLUMNS}, GROUP: 1, "Research_Paper_Name": "my study, HZSM-5",
                              "LDPE (wt%)": 100, "Reaction Temperature (C)": 450, "Reaction Time (min)": 60,
                              "Reactor Type": "batch reactor", "Operation Mode": "Batch", "Catalyst Loading (wt%)": 10,
                              "SBET (m2/g)": 309.46, "Catalytic Mode (In-situ/Ex-situ)": "In-situ"},
                             {**{c: np.nan for c in INPUT_COLUMNS}, GROUP: 1, "Research_Paper_Name": "my study, no catalyst",
                              "LDPE (wt%)": 100, "Reaction Temperature (C)": 450, "Reaction Time (min)": 60,
                              "Reactor Type": "batch reactor", "Operation Mode": "Batch", TARGET: 60.0}])[INPUT_COLUMNS]
    buf = BytesIO(); template.to_excel(buf, index=False)
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download template", buf.getvalue(), "oil_yield_template.xlsx", width="stretch")
    use_sample = c2.button("Use sample data", width="stretch")
    up = st.file_uploader("Upload file", type=["xlsx", "csv"])

    data = None
    if up is not None:
        if up.name.endswith(".csv"):
            data = pd.read_csv(up)
        else:
            sheets = pd.ExcelFile(up).sheet_names
            sheet = st.selectbox("Sheet", sheets)
            data = pd.read_excel(up, sheet_name=sheet)
    elif use_sample or st.session_state.get("sample"):
        st.session_state["sample"] = True
        data = pd.read_excel(config.UNSEEN_FILE, sheet_name="Validation_Dataset")

    if data is not None:
        missing = [c for c in INPUT_COLUMNS if c not in data.columns]
        if missing:
            st.warning("Columns not found (treated as not reported): " + ", ".join(missing))
        d = prepare(data)
        X = engineer(d)
        actual = d[TARGET].values
        out = data.copy()
        out["Predicted Oil Yield (wt%)"] = predict(bundle, X).round(2)
        out["Reference run used"] = np.where(X.has_thermal_baseline == 1, "yes", "no")
        out["Catalytic run"] = np.where(X.has_catalyst == 1, "yes", "no")

        has_actual = np.isfinite(actual)
        k1, k2, k3, k4 = st.columns(4)
        with k1: card("Rows", f"{len(out)}", f"{int((X.has_catalyst == 1).sum())} catalytic")
        with k2: card("With reference run", f"{int(X.has_thermal_baseline.sum())}", "thermal baseline found")
        if has_actual.sum() >= 3:
            from sklearn.metrics import mean_absolute_error, r2_score
            a, p = actual[has_actual], out["Predicted Oil Yield (wt%)"].values[has_actual]
            with k3: card("R² (rows with actual value)", f"{r2_score(a, p):.3f}", f"{has_actual.sum()} rows")
            with k4: card("MAE", f"{mean_absolute_error(a, p):.1f} wt%", "mean absolute error")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=a, y=p, mode="markers", marker=dict(size=9, color=TEAL, opacity=.75),
                                     hovertemplate="actual %{x:.1f}, predicted %{y:.1f}<extra></extra>", name="experiments"))
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color=GREY), name="perfect"))
            fig.update_xaxes(title="Actual oil yield (wt%)", range=[-2, 102])
            fig.update_yaxes(title="Predicted oil yield (wt%)", range=[-2, 102])
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        view = pd.DataFrame({"Group": out[GROUP] if GROUP in out else "",
                             "Study": out.get("Research_Paper_Name", pd.Series([""] * len(out))).astype(str).str.split(r" \(").str[0].str[:40],
                             "T (°C)": d["Reaction Temperature (C)"], "Loading (wt%)": X["cat_loading_unified"].round(2),
                             "Catalytic": out["Catalytic run"], "Reference run": out["Reference run used"],
                             "Actual oil (wt%)": d[TARGET], "Predicted oil (wt%)": out["Predicted Oil Yield (wt%)"]})
        if has_actual.any():
            view["Error (wt%)"] = (view["Predicted oil (wt%)"] - view["Actual oil (wt%)"]).round(2)
        st.dataframe(view, width="stretch", height=380, hide_index=True)
        res = BytesIO(); out.to_excel(res, index=False)
        st.download_button("Download predictions (Excel)", res.getvalue(), "oil_yield_predictions.xlsx", type="primary")


# ============================================================================= model performance
with tab_model:
    k1, k2, k3, k4 = st.columns(4)
    with k1: card("Unseen papers", f"R² {RESULTS['unseen_all'][0]:.2f}", f"73 experiments, 8 papers · MAE {RESULTS['unseen_all'][1]} wt%")
    with k2: card("Unseen, with reference run", f"R² {RESULTS['unseen_ref'][0]:.2f}", f"55 experiments · MAE {RESULTS['unseen_ref'][1]} wt%")
    with k3: card("Source-aware CV", f"R² {RESULTS['cv_cat'][0]:.2f}", f"catalytic runs · MAE {RESULTS['cv_cat'][1]} wt%")
    with k4: card("Training data", "2,965", "experiments from 304 papers")

    st.markdown("")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.markdown('<div class="section">How the model was developed</div>', unsafe_allow_html=True)
        labels, vals = zip(*JOURNEY[::-1])
        colours = [GREY, RED, TEAL, TEAL, TEAL_DARK, AMBER, AMBER][::-1]
        fig = go.Figure(go.Bar(y=[l.replace("\n", " ") for l in labels], x=vals, orientation="h", marker_color=colours,
                               text=[f"{v:.2f}" for v in vals], textposition="outside",
                               hovertemplate="%{y}: R² %{x:.3f}<extra></extra>"))
        fig.add_vline(x=0, line_color=GREY)
        fig.update_xaxes(title="R²", range=[-0.35, 0.9])
        st.plotly_chart(style_fig(fig, 400), width="stretch")
        st.caption("A random 80/20 split looks strong (0.72) because experiments from one paper sit in both training "
                   "and test sets; the same model fails on new papers. Every later score holds out whole papers.")
    with c2:
        st.markdown('<div class="section">Most important inputs</div>', unsafe_allow_html=True)
        imp = pd.Series(bundle["models"]["CatBoost"].get_feature_importance(), index=bundle["features"])
        imp = imp.sort_values(ascending=False).head(12)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[NICE.get(f, f) for f in imp.index], orientation="h",
                               marker_color=TEAL, hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
        fig.update_xaxes(title="Importance (CatBoost, %)")
        st.plotly_chart(style_fig(fig, 400), width="stretch")

    with st.expander("Validation details and limitations"):
        st.markdown(f"""
- **Source-aware validation:** 5-fold `StratifiedGroupKFold` grouped by paper, repeated with 3 seeds; no paper is ever
  in both training and test data.
- **External test:** 73 experiments from 8 papers published in 2025–2026, never used for training, tuning or model choice.
- **Reference run:** with a no-catalyst run of the same paper, cross-validation R² is {RESULTS['cv_with_ref'][0]:.2f}
  (MAE {RESULTS['cv_with_ref'][1]} wt%); without it {RESULTS['cv_without_ref'][0]:.2f} (MAE {RESULTS['cv_without_ref'][1]} wt%).
- **Missing values** are not filled; the gradient-boosting models handle them directly (KNN imputation lowered R²).
- **Scope:** polyolefins, PS and PET with zeolite and other solid-acid catalysts at about 300–700 °C. Predictions far
  outside this range, or for unusual reactors, are less reliable.
- The largest remaining source of error is laboratory-to-laboratory variation (reactor, condenser, definition of oil).
""")
