from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_utils import (
    OUTPUTS_DIR,
    TARGET_COLUMN,
    TaskDefinition,
    apply_missing_value_strategy,
    coerce_object_columns,
    ensure_dir,
    missingness_summary,
    read_dataset,
    save_json,
    seed_everything,
    setup_task_logger,
    write_task_metadata,
)


TASK_SLUG = "task1_ingestion_schema_missingness"


def detect_schema_issues(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    unnamed_columns = [col for col in df.columns if str(col).lower().startswith("unnamed")]
    if unnamed_columns:
        issues.append(f"Unnamed columns detected: {unnamed_columns}")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        normalized = df[col].dropna().astype(str).str.strip()
        if normalized.empty:
            continue
        if (normalized != df[col].dropna().astype(str)).any():
            issues.append(f"Whitespace inconsistency detected in column '{col}'")

    if TARGET_COLUMN not in df.columns:
        issues.append(f"Expected target column '{TARGET_COLUMN}' is missing")

    if df.duplicated().any():
        issues.append(f"Duplicate rows detected: {int(df.duplicated().sum())}")

    for col in df.columns:
        if df[col].isna().all():
            issues.append(f"Column '{col}' is entirely null")

    return issues


def main() -> None:
    seed_everything()
    logger, log_path = setup_task_logger(TASK_SLUG)
    task_dir = ensure_dir(OUTPUTS_DIR / TASK_SLUG)

    definition = TaskDefinition(
        task_name="Dataset ingestion + schema checks + missingness handling",
        specification=(
            "Load the CSV safely, inspect schema and missingness, detect obvious data-quality "
            "issues, apply a documented missing-value strategy, and save reports plus a cleaned dataset."
        ),
        success_criteria=[
            "Dataset loads from data/StudentPerformanceFactors.csv without manual intervention.",
            "Schema report includes shape, columns, dtypes, duplicates, and obvious schema warnings.",
            "Missingness report is saved before and after handling.",
            "A cleaned CSV with handled missing values is written under outputs/.",
        ],
    )
    write_task_metadata(definition, task_dir / "task_definition.json", {"log_path": str(log_path)})

    failures: list[dict[str, str]] = []
    try:
        raw_df = read_dataset(logger)
        raw_df = coerce_object_columns(raw_df, logger)

        schema_report = {
            "row_count": int(raw_df.shape[0]),
            "column_count": int(raw_df.shape[1]),
            "column_names": raw_df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in raw_df.dtypes.items()},
            "duplicate_rows": int(raw_df.duplicated().sum()),
            "schema_issues": detect_schema_issues(raw_df),
        }
        save_json(schema_report, task_dir / "schema_report.json")
        logger.info("Saved schema report to %s", task_dir / "schema_report.json")

        missing_before = missingness_summary(raw_df)
        missing_before.to_csv(task_dir / "missingness_before.csv", index=False)
        logger.info("Saved pre-imputation missingness report")

        cleaned_df, strategy = apply_missing_value_strategy(raw_df, logger, target=TARGET_COLUMN)

        missing_after = missingness_summary(cleaned_df)
        missing_after.to_csv(task_dir / "missingness_after.csv", index=False)
        cleaned_df.to_csv(task_dir / "cleaned_student_performance.csv", index=False)
        save_json(strategy, task_dir / "missing_value_strategy.json")
        logger.info("Saved cleaned dataset and missing-value strategy")

        success = {
            "status": "success",
            "target_column_choice": TARGET_COLUMN,
            "remaining_missing_values": int(cleaned_df.isna().sum().sum()),
            "cleaned_dataset_path": str(task_dir / "cleaned_student_performance.csv"),
        }
        save_json(success, task_dir / "run_summary.json")
    except Exception as exc:
        logger.exception("Task failed")
        failures.append(
            {
                "failure": str(exc),
                "detected_by": "Python exception raised during ingestion/schema/missingness task execution.",
                "corrected": "no",
            }
        )
        save_json({"status": "failed", "failures": failures}, task_dir / "run_summary.json")
        raise


if __name__ == "__main__":
    main()
