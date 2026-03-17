from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

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
    seed_everything,
    setup_task_logger,
    split_columns,
    write_task_metadata,
)


TASK_SLUG = "task3_baseline_modeling"


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def evaluate_model(name: str, model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = {
        "model": name,
        "mae": round(float(mean_absolute_error(y_test, preds)), 6),
        "rmse": round(rmse(y_test, preds), 6),
        "r2": round(float(r2_score(y_test, preds)), 6),
    }
    prediction_frame = pd.DataFrame(
        {
            "row_index": y_test.index,
            "actual": y_test.values,
            "prediction": preds,
            "model": name,
        }
    )
    return metrics, prediction_frame


def main() -> None:
    seed_everything()
    logger, log_path = setup_task_logger(TASK_SLUG)
    task_dir = ensure_dir(OUTPUTS_DIR / TASK_SLUG)

    definition = TaskDefinition(
        task_name="Baseline model training + evaluation harness",
        specification=(
            "Use Exam_Score as a regression target, build a leakage-safe preprocessing and evaluation harness, "
            "train one simple and one stronger baseline, and save metrics, config, and predictions."
        ),
        success_criteria=[
            "Target selection is stated and justified.",
            "Train/test split uses a fixed random seed.",
            "Pipelines impute and encode inside the estimator flow to avoid leakage.",
            "Metrics, predictions, and run configuration are saved.",
        ],
    )
    write_task_metadata(definition, task_dir / "task_definition.json", {"log_path": str(log_path)})

    try:
        df = read_dataset(logger)
        df = coerce_object_columns(df, logger)

        x = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        numeric_cols, categorical_cols = split_columns(df, target=TARGET_COLUMN)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=SEED
        )
        logger.info("Train/test split complete: train=%s, test=%s", x_train.shape, x_test.shape)

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        models: dict[str, Pipeline] = {
            "dummy_mean_regressor": Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", DummyRegressor(strategy="mean")),
                ]
            ),
            "ridge_regression": Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            "random_forest_regressor": Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", RandomForestRegressor(n_estimators=250, random_state=SEED, n_jobs=1)),
                ]
            ),
        }

        metrics_records = []
        prediction_frames = []
        for name, model in models.items():
            logger.info("Training %s", name)
            metrics, preds = evaluate_model(name, model, x_train, x_test, y_train, y_test)
            metrics_records.append(metrics)
            prediction_frames.append(preds)
            logger.info("Completed %s with metrics=%s", name, metrics)

        metrics_df = pd.DataFrame(metrics_records).sort_values("rmse")
        metrics_df.to_csv(task_dir / "metrics.csv", index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(task_dir / "predictions.csv", index=False)

        config = {
            "target_column": TARGET_COLUMN,
            "task_type": "regression",
            "seed": SEED,
            "test_size": 0.2,
            "feature_columns": x.columns.tolist(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "leakage_prevention": (
                "Imputation, scaling, and one-hot encoding are fit inside sklearn pipelines on the training split only."
            ),
            "models": list(models.keys()),
        }
        save_json(config, task_dir / "run_config.json")

        best_row = metrics_df.iloc[0].to_dict()
        save_json({"best_model": best_row}, task_dir / "best_model_summary.json")
        logger.info("Saved metrics, predictions, and configuration")
    except Exception:
        logger.exception("Task failed")
        raise


if __name__ == "__main__":
    main()
