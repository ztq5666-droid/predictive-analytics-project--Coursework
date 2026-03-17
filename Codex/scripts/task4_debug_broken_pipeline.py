from __future__ import annotations

import traceback
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_utils import (
    OUTPUTS_DIR,
    SEED,
    TARGET_COLUMN,
    TaskDefinition,
    coerce_object_columns,
    ensure_dir,
    read_dataset,
    save_json,
    save_text,
    seed_everything,
    setup_task_logger,
    split_columns,
    write_task_metadata,
)


TASK_SLUG = "task4_debug_broken_pipeline"


def build_broken_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]), categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def build_fixed_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def main() -> None:
    seed_everything()
    logger, log_path = setup_task_logger(TASK_SLUG)
    task_dir = ensure_dir(OUTPUTS_DIR / TASK_SLUG)

    definition = TaskDefinition(
        task_name="Debugging a deliberately broken pipeline",
        specification=(
            "Construct a small broken regression pipeline using this dataset, capture the failure, diagnose the "
            "root cause, fix the pipeline, and confirm the repaired version runs successfully."
        ),
        success_criteria=[
            "The broken pipeline fails with saved evidence.",
            "The cause is identified clearly.",
            "A corrected pipeline is executed successfully with saved confirmation output.",
        ],
    )
    write_task_metadata(definition, task_dir / "task_definition.json", {"log_path": str(log_path)})

    df = read_dataset(logger)
    df = coerce_object_columns(df, logger)
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    numeric_cols, categorical_cols = split_columns(df, target=TARGET_COLUMN)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )

    failure_record: dict[str, str] = {}

    try:
        broken = build_broken_pipeline(numeric_cols, categorical_cols)
        logger.info("Running deliberately broken pipeline")
        broken.fit(x_train, y_train)
    except Exception as exc:
        failure_record = {
            "failure": str(exc),
            "detected_by": "Pipeline fit raised an exception because categorical strings reached LinearRegression unchanged.",
            "diagnosis": (
                "The categorical branch only imputes values and does not encode strings, so the linear model receives "
                "non-numeric feature values."
            ),
            "traceback_path": str(task_dir / "broken_pipeline_traceback.txt"),
        }
        save_text(traceback.format_exc(), task_dir / "broken_pipeline_traceback.txt")
        save_json(failure_record, task_dir / "broken_pipeline_failure.json")
        logger.exception("Broken pipeline failed as expected")

    fixed = build_fixed_pipeline(numeric_cols, categorical_cols)
    logger.info("Running fixed pipeline")
    fixed.fit(x_train, y_train)
    preds = fixed.predict(x_test)
    confirmation = pd.DataFrame(
        {
            "row_index": y_test.index[:20],
            "actual": y_test.iloc[:20].values,
            "prediction": preds[:20],
        }
    )
    confirmation.to_csv(task_dir / "fixed_pipeline_sample_predictions.csv", index=False)

    result_summary = {
        "broken_pipeline_failed": bool(failure_record),
        "fix_applied": "Added OneHotEncoder(handle_unknown='ignore') to the categorical preprocessing branch.",
        "fixed_pipeline_status": "success",
        "sample_prediction_rows_saved": int(confirmation.shape[0]),
    }
    save_json(result_summary, task_dir / "fix_confirmation.json")
    logger.info("Saved failure evidence and fix confirmation")


if __name__ == "__main__":
    main()
