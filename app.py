from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import build_feature_matrix
from database import (
    initialize_database,
    insert_upload,
    log_analysis_history,
    save_detection_results,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "lightgbm_sgcc_research.joblib"
META_PATH = ROOT / "model_metadata.json"
CONFIG_PATH = ROOT / "app_config.json"
DATE_PATH = ROOT / "raw_date_columns.json"
SAMPLE_PATH = ROOT / "sample_sgcc_input.csv"

initialize_database()

st.set_page_config(
    page_title="Electricity Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 1.6rem 1.7rem; border: 1px solid #dce8e5; border-radius: 18px;
           background: linear-gradient(120deg, #eff8f5 0%, #f7fafc 100%); margin-bottom: 1.35rem;}
    .eyebrow {font-size: .76rem; text-transform: uppercase; letter-spacing: .12em;
              font-weight: 750; color: #24705f; margin-bottom: .5rem;}
    .hero h1 {font-size: 2.1rem; line-height: 1.15; letter-spacing: -.04em;
              color: #153c35; margin: 0 0 .6rem 0;}
    .hero p {color: #49645e; font-size: 1rem; line-height: 1.55; margin: 0; max-width: 760px;}
    .section-title {font-size: 1.05rem; font-weight: 700; color: #203d38; margin: .7rem 0 .3rem;}
    div[data-testid="stMetric"] {border: 1px solid #e1e9e7; border-radius: 13px; padding: .9rem 1rem; background: #fff;}
    div[data-testid="stFileUploader"] {border: 1px dashed #a8c8bf; border-radius: 14px; padding: .5rem; background: #fbfdfc;}
    .stButton button, .stDownloadButton button {border-radius: 10px; min-height: 2.65rem; font-weight: 650;}
    .footnote {font-size: .84rem; color: #60716d; line-height: 1.5;}
    @media (prefers-color-scheme: dark) {
      .hero {background: linear-gradient(120deg, #18332e 0%, #202a30 100%); border-color: #344943;}
      .hero h1 {color: #e3f4ef;} .hero p {color: #c0d3cd;} .eyebrow {color: #8bd1bc;}
      .section-title {color: #d8e9e4;} div[data-testid="stMetric"] {background: #1b2428; border-color: #364448;}
      div[data-testid="stFileUploader"] {background: #1b2428; border-color: #52756a;}
      .footnote {color: #b6c6c1;}
    }
    </style>
    <div class="hero">
      <div class="eyebrow">SGCC · LightGBM research prototype</div>
      <h1>Electricity Anomaly Detector</h1>
      <p>Upload SGCC customer electricity-consumption data, run the model, and download the records flagged for review. The workflow is kept simple: upload, analyze, review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the LightGBM model…")
def load_assets():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_dates = json.loads(DATE_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    if len(feature_columns) != int(config["feature_count"]):
        raise RuntimeError("Model feature configuration does not match.")
    if getattr(model, "n_features_in_", None) != len(feature_columns):
        raise RuntimeError("The saved model does not match its feature configuration.")
    if len(raw_dates) != int(config["raw_date_column_count"]):
        raise RuntimeError("The daily-date schema is incomplete.")
    return model, metadata, config, raw_dates


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    normalized = []
    for column in frame.columns:
        name = str(column).replace("\ufeff", "").strip()
        try:
            parsed = pd.to_datetime(name, format="%m/%d/%Y", errors="raise")
            name = f"{parsed.month}/{parsed.day}/{parsed.year}"
        except (ValueError, TypeError):
            pass
        normalized.append(name)
    frame.columns = normalized
    duplicated = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError("Duplicate column names were found after date normalization.")
    return frame


def read_upload(uploaded):
    payload = uploaded.getvalue()
    if uploaded.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
            if not csv_names:
                raise ValueError("The ZIP file does not contain a CSV file.")
            selected = next((name for name in csv_names if Path(name).name.lower() == "data set.csv"), csv_names[0])
            with archive.open(selected) as stream:
                frame = pd.read_csv(stream, low_memory=False)
            source_name = Path(selected).name
    else:
        frame = pd.read_csv(io.BytesIO(payload), low_memory=False)
        source_name = uploaded.name
    frame = normalize_columns(frame)
    if frame.empty:
        raise ValueError("The uploaded file has column names but no data rows.")
    return frame, source_name


try:
    model, metadata, config, raw_dates = load_assets()
except Exception as exc:
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

st.markdown('<div class="section-title">1. Upload your data</div>', unsafe_allow_html=True)
st.write("Upload a CSV containing SGCC daily electricity-consumption history, or a ZIP containing the original SGCC CSV. The saved model requires the complete set of daily columns used during training.")
left, right = st.columns([3, 1], vertical_alignment="center")
with left:
    uploaded = st.file_uploader("Choose a CSV or ZIP file", type=["csv", "zip"], label_visibility="collapsed")
with right:
    if SAMPLE_PATH.exists():
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_PATH.read_bytes(),
            file_name="sample_sgcc_input.csv",
            mime="text/csv",
            use_container_width=True,
        )

if uploaded is None:
    st.info("Start by uploading your SGCC CSV. You can use the sample file to test the interface.")
    with st.expander("About this model"):
        cv = config.get("nested_cv_summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Cross-validation F1", f"{cv.get('F1_mean', float('nan')):.4f}")
        c2.metric("Cross-validation PR-AUC", f"{cv.get('PR_AUC_mean', float('nan')):.4f}")
        c3.metric("Cross-validation ROC-AUC", f"{cv.get('ROC_AUC_mean', float('nan')):.4f}")
        st.caption("These are historical development estimates, not results from a newly uploaded file.")
        st.warning("The model was trained on SGCC customer-level electricity-theft labels. It has not been validated for prepaid vending, token/recharge transaction fraud, or operational decisions.")
    st.stop()

try:
    with st.spinner("Reading your file…"):
        raw, source_name = read_upload(uploaded)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

missing_dates = [column for column in raw_dates if column not in raw.columns]
matched_dates = len(raw_dates) - len(missing_dates)

st.markdown('<div class="section-title">2. Check the uploaded data</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Records", f"{len(raw):,}")
m2.metric("Daily columns matched", f"{matched_dates:,} / {len(raw_dates):,}")
m3.metric("File", source_name, help="The selected CSV inside a ZIP is shown here.")

if missing_dates:
    st.error("This file does not contain the complete SGCC daily-history schema required by the LightGBM model, so predictions were not run.")
    st.write(f"Found **{matched_dates:,} of {len(raw_dates):,}** required daily columns. Please upload the full SGCC-format dataset. This app will not silently switch to a different detector.")
    with st.expander("What data is required?"):
        st.write("The model expects 1,034 daily consumption columns from the SGCC training schema, plus an optional `CONS_NO` identifier. A `FLAG` label is not required for prediction and is not used as a model input.")
        st.write("Example missing date columns: " + ", ".join(missing_dates[:8]))
    st.stop()

st.success("Data format matches the SGCC model. Ready to analyze.")
with st.expander("Preview uploaded records"):
    st.dataframe(raw.head(6), use_container_width=True, hide_index=True)

threshold = float(config.get("nested_cv_median_threshold", 0.560202))
st.markdown('<div class="section-title">3. Run detection</div>', unsafe_allow_html=True)
st.caption(f"The model uses a provisional score threshold of {threshold:.4f}. A flagged record should be reviewed; it is not proof of theft or fraud.")
if st.button("Analyze uploaded data", type="primary", use_container_width=True):
    try:
        with st.spinner("Preparing features and scoring records…"):
            features, ids, _, warnings_out = build_feature_matrix(raw, raw_dates, metadata["feature_columns"])
            scores = model.predict_proba(features)[:, 1]
            predictions = (scores >= threshold).astype(int)
            results = pd.DataFrame({
                "customer_id": ids,
                "model_score": np.round(scores, 4),
                "result": np.where(predictions == 1, "Flagged for review", "Not flagged"),
            })
            results = results.sort_values("model_score", ascending=False).reset_index(drop=True)
            upload_id = insert_upload(source_name, len(raw))
            records_saved, records_flagged = save_detection_results(
                upload_id=upload_id,
                records=(
                    {
                        "user_id": str(user_id),
                        "prediction": int(prediction),
                        "risk_score": float(score),
                    }
                    for user_id, prediction, score in zip(ids, predictions, scores)
                ),
                risk_threshold=threshold,
            )
            log_analysis_history(
                upload_id=upload_id,
                model_name="LightGBM SGCC research",
                records_analysed=records_saved,
                records_flagged=records_flagged,
            )
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.stop()

    flagged = int(predictions.sum())
    st.markdown('<div class="section-title">4. Results</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Records analyzed", f"{len(results):,}")
    r2.metric("Flagged for review", f"{flagged:,}")
    r3.metric("Flagged share", f"{flagged / max(len(results), 1) * 100:.1f}%")
    st.dataframe(results, use_container_width=True, hide_index=True, height=420)
    st.download_button(
        "Download results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="electricity_anomaly_results.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    for warning in warnings_out:
        st.caption(warning)
    st.caption("Results are sorted by model score, highest first. Scores are model outputs, not calibrated probabilities of fraud.")

st.divider()
st.markdown('<p class="footnote">Research prototype · Uses the saved SGCC LightGBM model locally. Uploaded data is not sent to an external API by this app. The current model is for customer-level SGCC consumption histories, not prepaid transaction fraud.</p>', unsafe_allow_html=True)
