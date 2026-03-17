from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "StudentPerformanceFactors.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SEED = 42
TARGET_COLUMN = "Exam_Score"


@dataclass
class TaskDefinition:
    task_name: str
    specification: str
    success_criteria: list[str]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_task_logger(task_slug: str) -> tuple[logging.Logger, Path]:
    ensure_dir(OUTPUTS_DIR)
    log_path = OUTPUTS_DIR / f"{task_slug}.log"
    logger = logging.getLogger(task_slug)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def read_dataset(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading dataset from %s", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset loaded with shape=%s", df.shape)
    return df


def split_columns(df: pd.DataFrame, target: str = TARGET_COLUMN) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[target])
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(4),
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    )
    return summary.sort_values(["missing_count", "column"], ascending=[False, True])


def schema_summary(df: pd.DataFrame) -> dict[str, Any]:
    dtype_counts = {}
    for dtype in df.dtypes.astype(str):
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "unnamed_columns": [col for col in df.columns if str(col).lower().startswith("unnamed")],
        "dtype_counts": dtype_counts,
    }


def coerce_object_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object"]).columns:
        non_null = cleaned[col].notna()
        cleaned.loc[non_null, col] = cleaned.loc[non_null, col].astype(str).str.strip()
        cleaned[col] = cleaned[col].astype("object")
        logger.info("Normalized string column '%s' while preserving null compatibility", col)
    return cleaned


def apply_missing_value_strategy(
    df: pd.DataFrame,
    logger: logging.Logger,
    target: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = cleaned.select_dtypes(exclude=["number"]).columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    strategy: dict[str, Any] = {"numeric": {}, "categorical": {}}

    for col in numeric_cols:
        median_value = float(cleaned[col].median())
        missing_count = int(cleaned[col].isna().sum())
        if missing_count:
            cleaned[col] = cleaned[col].fillna(median_value)
            logger.warning(
                "Filled %s missing values in numeric column '%s' with median=%s",
                missing_count,
                col,
                median_value,
            )
        strategy["numeric"][col] = {"method": "median", "fill_value": median_value}

    for col in categorical_cols:
        modes = cleaned[col].mode(dropna=True)
        fill_value = "Unknown" if modes.empty else str(modes.iloc[0])
        missing_count = int(cleaned[col].isna().sum())
        if missing_count:
            cleaned[col] = cleaned[col].fillna(fill_value)
            logger.warning(
                "Filled %s missing values in categorical column '%s' with mode='%s'",
                missing_count,
                col,
                fill_value,
            )
        strategy["categorical"][col] = {"method": "mode", "fill_value": fill_value}

    remaining_missing = int(cleaned.isna().sum().sum())
    logger.info("Remaining missing values after handling: %s", remaining_missing)
    return cleaned, strategy


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_task_metadata(
    definition: TaskDefinition,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = asdict(definition)
    if extra:
        payload.update(extra)
    save_json(payload, path)
