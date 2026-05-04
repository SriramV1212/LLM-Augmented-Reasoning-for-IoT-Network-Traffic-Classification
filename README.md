# LLM-Augmented Reasoning for IoT Network Traffic Classification

Course project (CSE 534) over **CICFlowMeter** features from the CIC-BCCC-NRC TabularIoTAttack-2024 setting.

Two **classification modes** (pick one per command):

- **`ml-agent` (default)** — Pre-trained **Random Forest** / **XGBoost**, **SHAP**, and an optional **LLM** explanation of the outcome (OpenRouter).
- **`zeroshot`** — **LLM-only** label + chain-of-thought / rationale (no tree model).

Trained models and EDA live in this repo (`models/`, `eda.ipynb`, `ml_classifier.ipynb`).

## Setup

```bash
cd LLM-Augmented-Reasoning-for-IoT-Network-Traffic-Classification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add OPENROUTER_API_KEY
```

Run from this directory so paths `models/` and `prompts/` resolve correctly (or pass `--models-dir` / `--prompts-dir`).

### Train the ML stack (outside Jupyter)

The notebook `ml_classifier.ipynb` is mirrored by `llm_agent/train_ml.py`. With the four class CSVs under `data/` (filenames must match exactly), run:

```bash
python cli.py train --data-dir data --output-dir models
```

Expected files in `data/`:

- `Benign Traffic.csv`
- `DDoS UDP Flood.csv`
- `DoS TCP Flood.csv` (only the first `--dos-tcp-rows` rows are read, default `100000`, matching the notebook)
- `Recon Port Scan.csv`

This writes `random_forest.joblib`, `xgboost.json`, `label_encoder.joblib`, `feature_cols.joblib`, `training_metrics.json`, and optional PNG plots (omit with `--no-plots`). Use `--skip-rf` or `--skip-xgb` to train a single backend.

If you do not have a `flow.json` yet, generate one from the trained feature list:

```bash
python cli.py sample-flow -o flow.json
python cli.py sample-flow -o flow.json --demo   # a few non-zero toy fields
```

## Flow JSON format

`classify` expects a JSON file with a `features` object keyed by **exact** CICFlowMeter column names (same order as training is not required — missing keys default to `0`).

```json
{
  "flow_id": "optional-string",
  "features": {
    "Flow Duration": 123.4,
    "Total Fwd Packets": 10,
    "Total Backward Packets": 8
  }
}
```

Export a full row from your dataframe, e.g. `row.to_dict()` for the feature columns saved in `models/feature_cols.joblib`.

## CLI

```bash
# Default: ML + SHAP + LLM explainer (same as ml-agent)
python cli.py classify -i flow.json
python cli.py classify -i flow.json --mode ml-agent --ml-backend rf

# ML + SHAP + rule-based text only (no OpenRouter call for explainer)
python cli.py classify -i flow.json --skip-llm

# LLM-only classification + rationale
python cli.py classify -i flow.json --mode zeroshot --model tencent/hy3-preview:free

# Evaluate ML vs labels; default also runs OpenRouter SHAP explainer on first 25 rows (needs API key)
python cli.py evaluate -d path/to/test.csv -o eval_report.json --max-rows 500 --mode ml-agent

# ML metrics only (no OpenRouter calls for explainer)
python cli.py evaluate -d path/to/test.csv -o eval_report.json --mode ml-agent --no-llm-explainer

# Evaluate zero-shot LLM vs labels (one API call per row; needs key)
python cli.py evaluate -d path/to/test.csv -o eval_report.json --max-rows 50 --mode zeroshot

# Verbose OpenRouter logs (request URL/model, timings, message previews → stderr)
python cli.py classify -i flow.json -v
python cli.py evaluate -d data/eval_sample.csv -o eval.json --mode ml-agent -v
python cli.py list-models -v

# Longer previews in those logs
python cli.py classify -i flow.json --debug-api

# List models available to your OpenRouter key
python cli.py list-models --filter llama

# Full pipeline: train (if CSVs exist) + classify + optional CSV evaluation
python cli.py pipeline --train --input flow.json --eval-dataset path/to/test.csv -o pipeline_report.json
python cli.py pipeline -i flow.json   # classify only; prints JSON (includes `why`)
```

`classify` returns JSON with **`classification_mode`**, **`final_prediction`**, **`zeroshot`** and **`ml_agent`** (one is always `null`), and **`why`** (unit-of-analysis, structured evidence, **`narrative`**). In **`ml_agent`**, the LLM explainer fills **`why.ml_agent.plausibility_llm`** (intuitive vs counterintuitive, SOC-style) plus **`llm_explanation`** / **`security_takeaway_llm`**, grounded in SHAP and serialized flow stats — not invented IPs or geolocation (this CIC tabular export has no IP columns).

## Python API (sketch)

```python
from pathlib import Path
from llm_agent.config import load_config
from llm_agent.openrouter_client import OpenRouterClient
from llm_agent.track_b_ml_explainer import load_ml_models, run_track_b

cfg = load_config()
client = OpenRouterClient(cfg)
art = load_ml_models(Path("models"))
result = run_track_b(your_feature_dict, art, client, cfg, ml_backend="xgboost")
print(result.prediction.predicted_class, result.explanation)
```

## Project layout

| Path | Role |
|------|------|
| `llm_agent/` | Library code (OpenRouter client, ML path, zero-shot, merger, evaluator) |
| `prompts/` | Editable system/user templates |
| `cli.py` | Entry point (`classify`, `evaluate`, `train`, `pipeline`, `sample-flow`, `list-models`) |
| `llm_agent/flow_explanation.py` | Builds `why` / narrative for a classification |
| `models/` | `random_forest.joblib`, `xgboost.json`, encoders, `feature_cols.joblib` (from `train` or checked-in) |
| `data/` | Optional local folder for the four training CSVs (often gitignored) |

### Get the training CSVs from Kaggle

1. Authenticate with Kaggle (pick one):
   - **Recommended:** save `kaggle.json` as `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`.
   - **Optional:** add `KAGGLE_USERNAME` and `KAGGLE_KEY` to this repo’s `.env` (`.env` is gitignored). The download script loads it before calling Kaggle.
2. Install the client: `pip install kaggle`
3. From the repo root:

```bash
python3 scripts/download_kaggle_data.py --data-dir data
```

This pulls [CIC-BCCC-NRC-TabularIoTAttacks-2024](https://www.kaggle.com/datasets/kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024) and copies the four files expected under `data/`. If filenames in the archive differ, the script prints what it found so you can rename manually.

### Build a small labeled test CSV and run `evaluate`

From the four `data/*.csv` files and `models/feature_cols.joblib`:

```bash
python3 scripts/make_eval_sample.py --per-class 500 -o data/eval_sample.csv
python3 cli.py evaluate -d data/eval_sample.csv -o data/eval_report.json --mode ml-agent
```

`--per-class` is only for **`make_eval_sample.py`** (rows per label in the built CSV). **`evaluate`** uses **`--max-rows`** to randomly subsample rows after loading that CSV (not per-class). In **`ml-agent`** mode, metrics use **all** loaded rows; the JSON report also includes **`ml_agent_llm_explainer_samples`** for the first **`--explain-max-rows`** rows (default 25) when **`OPENROUTER_API_KEY`** is set. Use **`--no-llm-explainer`** for metrics-only without API calls. For zero-shot LLM metrics (slow, needs API key): `--mode zeroshot --max-rows 20`.

## Notes

- Default LLM id is **`tencent/hy3-preview:free`** (OpenRouter free tier). Override with `--model` or `OPENROUTER_DEFAULT_MODEL` in `.env`.
- Default mode is **`ml_agent`**. Set **`LLM_AGENT_CLASSIFY_MODE`** to `ml_agent` or `zeroshot` (legacy **`LLM_AGENT_TRACK_MODE`**: `b`/`both` → ml_agent, `a` → zeroshot).
- OpenRouter uses the **OpenAI-compatible** HTTP API (`OPENROUTER_BASE_URL` defaults to `https://openrouter.ai/api/v1`).
- Training-time preprocessing for ML is **`sign(x) * log1p(|x|)`** on every feature (applied in the ML path).
- Dataset CSVs are gitignored; place your own CSV for `evaluate`.
