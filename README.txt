# EV Depot Demand Forecaster — Executive Operations Manual

This centralized production application provisions real-time time-series predictive modeling, grid headroom diagnostics, and automated capacity constraint telemetry tracking.

## System Architecture Blueprint
The application engine is built entirely on a single-file automated paradigm. It runs a production-ready `XGBRegressor` machine learning model on historical energy load vectors. To ensure absolute operational stability, the engine engineers lag observations and rolling moving averages exclusively using past context data frames, completely preventing data leakage during training splits.

## Inbound Telemetry Dataset Schemas
When manually uploading operational logging files into the dashboard portal, inbound file feeds (CSV or Excel formats) must contain two core column vectors:
* **Chronological Time Axis:** Structured series entries formatting date and hours exactly as `YYYY-MM-DD HH:MM:SS`.
* **Target Load Array:** Quantitative continuous load indices matching the expected `Demand_kW` metric parameters.

## Workspace Terminal Initialization Sequence
Deploy and launch the complete integrated system array natively using standard command-line tools:

1. Provision an isolated local virtual environment sandbox:
   ```bash
   python -m venv venv
   ```
2. Engage the active terminal shell route alignment:
   ```bash
   # Windows Command Prompt / PowerShell:
   venv\Scripts\activate
   # macOS or Linux Terminal Shell:
   source venv/bin/activate
   ```
3. Install dependencies cleanly to prevent library configuration mismatch errors:
   ```bash
   pip install -r requirements.txt
   ```
4. To initialize the interactive web application dashboard instantly:
   ```bash
   streamlit run app.py
   ```
5. To run the standalone project report training pipeline and export all 6 analytical validation chart plots dynamically as high-resolution images:
   ```bash
   python train_report_pipeline.py
   ```
## Production Security Credentials
Access administrative configuration matrices and model summary registers via the default master keys:
* **Corporate ID Node:** operations@ukpowernetworks.co.uk
* **Security Password String:** ••••••••••••