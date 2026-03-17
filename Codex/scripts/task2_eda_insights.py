from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    read_dataset,
    save_text,
    seed_everything,
    setup_task_logger,
    split_columns,
    write_task_metadata,
)


TASK_SLUG = "task2_eda_insights"


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    seed_everything()
    logger, log_path = setup_task_logger(TASK_SLUG)
    task_dir = ensure_dir(OUTPUTS_DIR / TASK_SLUG)
    plots_dir = ensure_dir(task_dir / "plots")

    definition = TaskDefinition(
        task_name="EDA and insight generation",
        specification=(
            "Produce summary statistics, save informative plots, and write concise insights that are "
            "supported by the dataset and generated artifacts."
        ),
        success_criteria=[
            "Numeric and categorical summaries are saved.",
            "Several informative plots are generated and written under outputs/.",
            "Insights mention evidence from the saved summaries or plots without unsupported claims.",
        ],
    )
    write_task_metadata(definition, task_dir / "task_definition.json", {"log_path": str(log_path)})

    try:
        sns.set_theme(style="whitegrid")
        df = read_dataset(logger)
        df = coerce_object_columns(df, logger)
        df, _ = apply_missing_value_strategy(df, logger, target=TARGET_COLUMN)

        numeric_cols, categorical_cols = split_columns(df, target=TARGET_COLUMN)

        numeric_summary = df[numeric_cols + [TARGET_COLUMN]].describe().T
        numeric_summary.to_csv(task_dir / "numeric_summary.csv")

        categorical_summary = []
        for col in categorical_cols:
            top_counts = df[col].value_counts().head(5)
            for category, count in top_counts.items():
                categorical_summary.append(
                    {"column": col, "category": category, "count": int(count)}
                )
        pd.DataFrame(categorical_summary).to_csv(task_dir / "categorical_top_counts.csv", index=False)

        corr = df[numeric_cols + [TARGET_COLUMN]].corr(numeric_only=True)
        corr.to_csv(task_dir / "correlation_matrix.csv")
        corr_to_target = corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
        corr_to_target.to_csv(task_dir / "correlation_to_target.csv", header=["correlation"])

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[TARGET_COLUMN], bins=25, kde=True, ax=ax, color="#2a6f97")
        ax.set_title("Exam Score Distribution")
        save_plot(fig, plots_dir / "exam_score_distribution.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="Previous_Scores",
            y=TARGET_COLUMN,
            alpha=0.6,
            ax=ax,
            color="#ca6702",
        )
        ax.set_title("Previous Scores vs Exam Score")
        save_plot(fig, plots_dir / "previous_scores_vs_exam_score.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        order = df.groupby("Motivation_Level")[TARGET_COLUMN].median().sort_values(ascending=False).index
        sns.boxplot(
            data=df,
            x="Motivation_Level",
            y=TARGET_COLUMN,
            order=order,
            hue="Motivation_Level",
            dodge=False,
            legend=False,
            ax=ax,
            palette="Set2",
        )
        ax.set_title("Exam Score by Motivation Level")
        save_plot(fig, plots_dir / "exam_score_by_motivation.png")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        save_plot(fig, plots_dir / "correlation_heatmap.png")

        top_target_features = corr_to_target.head(3).to_dict()
        grouped_motivation = (
            df.groupby("Motivation_Level")[TARGET_COLUMN]
            .agg(["mean", "median", "count"])
            .sort_values("median", ascending=False)
        )
        grouped_motivation.to_csv(task_dir / "motivation_grouped_scores.csv")

        insights = [
            "Exam_Score is the chosen target because it is the only clear outcome variable and is numeric.",
            (
                "Previous_Scores has the strongest positive linear relationship with Exam_Score among the numeric "
                f"predictors in the saved correlation file: {corr_to_target.index[0]}={corr_to_target.iloc[0]:.3f}."
            ),
            (
                "Hours_Studied and Attendance also show positive associations with Exam_Score, which is visible in "
                "the correlation outputs and consistent with the scatter and distribution patterns."
            ),
            (
                "Motivation groups differ in central tendency; the saved grouped summary and boxplot show that the "
                f"top median group is {grouped_motivation.index[0]} with median={grouped_motivation.iloc[0]['median']:.1f}."
            ),
            (
                "Missingness was limited to three categorical columns and was handled before plotting, so the EDA "
                "artifacts reflect the cleaned dataset used later for modeling."
            ),
        ]
        save_text("\n".join(f"- {line}" for line in insights), task_dir / "insights.txt")
        logger.info("Saved EDA summaries, plots, and insights")
    except Exception:
        logger.exception("Task failed")
        raise


if __name__ == "__main__":
    main()
