from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_utils import OUTPUTS_DIR, TARGET_COLUMN, save_text


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    task1_dir = OUTPUTS_DIR / "task1_ingestion_schema_missingness"
    task2_dir = OUTPUTS_DIR / "task2_eda_insights"
    task3_dir = OUTPUTS_DIR / "task3_baseline_modeling"
    task4_dir = OUTPUTS_DIR / "task4_debug_broken_pipeline"

    schema = read_json(task1_dir / "schema_report.json")
    task1_summary = read_json(task1_dir / "run_summary.json")
    metrics = pd.read_csv(task3_dir / "metrics.csv")
    best_model = metrics.sort_values("rmse").iloc[0]
    debug_summary = read_json(task4_dir / "fix_confirmation.json")
    failure_summary = read_json(task4_dir / "broken_pipeline_failure.json")

    report = f"""# Benchmark Report

## Task 1: Dataset Ingestion + Schema Checks + Missingness Handling

- Specification: Load the CSV safely, inspect schema and missingness, detect obvious issues, document a missing-value strategy, and save a cleaned dataset.
- Success criteria: dataset loads, schema report exists, missingness is reported before/after handling, cleaned CSV is saved.
- Result: Success. Rows={schema['row_count']}, columns={schema['column_count']}, duplicates={schema['duplicate_rows']}, remaining missing values after handling={task1_summary['remaining_missing_values']}.
- Failures: No runtime failure. Data-quality issues were detected through profiling: missing categorical values in `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home`.
- Corrections: Filled numeric nulls with medians and categorical nulls with the mode. In practice only categorical columns required filling.
- Reproducibility: Deterministic file input, fixed seed, saved schema and missingness reports, and saved cleaned CSV under `outputs/task1_ingestion_schema_missingness/`.

## Task 2: EDA and Insight Generation

- Specification: Produce summary statistics, save informative plots, and write concise insights grounded in those outputs.
- Success criteria: summaries saved, plots saved, insights trace back to saved artifacts.
- Result: Success. Numeric summaries, categorical counts, correlation outputs, and four plots were generated under `outputs/task2_eda_insights/`.
- Failures: One environment failure occurred during execution development: Matplotlib initially attempted to use a Tk GUI backend and raised a `TclError` in this headless environment. It was detected from the plotting exception trace.
- Corrections: Forced the non-interactive `Agg` backend before importing `pyplot`, then reran the task successfully.
- Reproducibility: Plots are rendered with a fixed script path and saved directly to disk; no interactive notebook state is involved.

## Task 3: Baseline Model Training + Evaluation Harness

- Specification: Predict `{TARGET_COLUMN}` with leakage-safe preprocessing, a fixed train/test split, baseline models, and saved metrics/predictions.
- Success criteria: target documented, baseline and stronger model trained, metrics saved, predictions saved, config saved.
- Result: Success. Best model={best_model['model']} with MAE={best_model['mae']:.3f}, RMSE={best_model['rmse']:.3f}, R2={best_model['r2']:.3f}. The target choice was `{TARGET_COLUMN}` because it is the clear numeric outcome column.
- Failures: Two failures were detected during development by traceback inspection. First, pandas `string` dtype introduced `pd.NA`, which caused `SimpleImputer` to fail with `TypeError: boolean value of NA is ambiguous`. Second, `RandomForestRegressor(n_jobs=-1)` triggered `PermissionError: [WinError 5] Access is denied` in the sandbox when opening worker resources.
- Corrections: Normalized categoricals while preserving `np.nan` compatibility instead of pandas `string` dtype, and changed the random forest baseline to `n_jobs=1`. The final evaluation harness then ran successfully.
- Reproducibility: Train/test split uses fixed seed 42, preprocessing stays inside sklearn pipelines to prevent leakage, and metrics/config/predictions are saved under `outputs/task3_baseline_modeling/`.

## Task 4: Debugging a Deliberately Broken Pipeline

- Specification: Build a small broken pipeline on this dataset, capture the failure, diagnose it, fix it, and confirm the fix.
- Success criteria: failure evidence saved, diagnosis stated, corrected pipeline runs successfully.
- Result: Success after correction. The broken pipeline failure was captured, diagnosed, fixed, and rerun successfully.
- Failures: Deliberate failure detected by exception during `LinearRegression.fit`, caused by categorical strings reaching the estimator without encoding.
- Corrections: Added `OneHotEncoder(handle_unknown='ignore')` to the categorical preprocessing branch and reran the pipeline successfully.
- Reproducibility: The broken and fixed pipelines are both defined in code, use the same dataset and split, and save the traceback plus sample predictions for verification.

## Global Reproducibility Notes

- Python scripts only; no notebooks were required.
- Fixed random seed: 42.
- All outputs, logs, metrics, plots, and predictions are saved under `outputs/`.
- Full reproducibility still depends on installed package versions and sandbox permissions, but the workflow avoids network calls and external state.

## Evidence

- Schema report: `outputs/task1_ingestion_schema_missingness/schema_report.json`
- Missingness reports: `outputs/task1_ingestion_schema_missingness/missingness_before.csv`, `outputs/task1_ingestion_schema_missingness/missingness_after.csv`
- EDA insights: `outputs/task2_eda_insights/insights.txt`
- Modeling metrics: `outputs/task3_baseline_modeling/metrics.csv`
- Broken pipeline traceback: `{failure_summary['traceback_path']}`
- Fix confirmation: `outputs/task4_debug_broken_pipeline/fix_confirmation.json`
"""
    save_text(report, OUTPUTS_DIR / "benchmark_report.md")


if __name__ == "__main__":
    main()
