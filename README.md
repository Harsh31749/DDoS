# DDoS Detection Pipeline

This project trains and deploys machine-learning models to detect DDoS traffic from tabular network-flow features.

## Quick start

1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Train models and generate deployment artifacts:
   - `python main.py --mode train`
3. Start realtime detector:
   - `python website_monitor.py`
4. Open dashboard:
   - `streamlit run app.py`

## Key outputs

- `outputs/inference_bundle.pkl`: deployment-ready model bundle
- `outputs/model_performance.csv`: model metrics
- `outputs/cv_results.csv`: sampled CV stability
- `outputs/pipeline_results.json`: dashboard payload
- `outputs/preprocess_report.json`: preprocessing and validation report
- `outputs/live_alerts.json`: realtime alert feed
- `outputs/live_metrics.json`: detector runtime health metrics

## Test

- `python -m unittest discover -s tests -p "test_*.py"`
