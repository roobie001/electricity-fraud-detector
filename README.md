# Electricity Anomaly Detector — VS Code Project

A small local Streamlit app using the agreed SGCC + LightGBM model. Open this folder in Visual Studio Code to read, edit, and run the source code.

## Requirements

- Python 3.11, 3.12, or 3.13
- Visual Studio Code
- The Python extension for VS Code
- Internet access for the first-time package installation

## Open in VS Code

1. Extract this ZIP file to a folder on your computer.
2. In VS Code, choose **File → Open Folder…** and select `Electricity_Anomaly_Detector_VSCode`.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment and install the packages using the commands for your operating system below.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks environment activation, skip the activation command and run these instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

5. Streamlit prints a local address, usually `http://localhost:8501`. Open that address in your browser.
6. Upload an SGCC-compatible CSV, or click **Download sample CSV** in the app to test the interface.

## Files in this folder

- `app.py` — user interface and upload/prediction workflow.
- `feature_engineering.py` — converts SGCC daily consumption data into the model's 66 features.
- `lightgbm_sgcc_research.joblib` — saved LightGBM model.
- `model_metadata.json`, `app_config.json`, `feature_columns.json`, `raw_date_columns.json` — model and input-schema configuration.
- `sample_sgcc_input.csv` — small sample for testing the application workflow.
- `verify_app.py` — basic checks for the model and sample input.
- `requirements.txt` — required Python packages.

## Optional model check

After installing the requirements, run:

```bash
python verify_app.py
```

## Important limitation

The model expects the SGCC customer-level daily electricity-consumption format, including 1,034 daily columns, and was trained using customer-level electricity-theft labels. It has not been validated for prepaid token, recharge, or vending-transaction fraud. A flagged record is a review signal, not proof of theft or fraud. The included sample is for a software smoke test, not an independent evaluation set.

The app processes uploads locally and does not send them to an external API.
