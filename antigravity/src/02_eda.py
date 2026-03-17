import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(cleaned_data_path, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading cleaned data from {cleaned_data_path}")
    df = pd.read_csv(cleaned_data_path)

    # 2. Compute Summary Statistics
    summary_stats = df.describe(include="all")
    stats_path = os.path.join(outputs_dir, "summary_statistics.csv")
    summary_stats.to_csv(stats_path)
    print(f"Saved summary statistics to {stats_path}")

    # 3. Generate Insights Text Output
    insights = [
        "### Exploratory Data Analysis Insights ###\n",
        f"Analyzed {len(df)} records.",
        f"Mean Exam Score: {df['Exam_Score'].mean():.2f}",
        f"Median Exam Score: {df['Exam_Score'].median():.2f}",
        f"Max Exam Score: {df['Exam_Score'].max()}",
        f"Min Exam Score: {df['Exam_Score'].min()}\n",
    ]

    # 4. Generate Plots
    sns.set_theme(style="whitegrid")

    # Plot 1: Distribution of Exam Scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Exam_Score"], kde=True, color="blue", bins=30)
    plt.title("Distribution of Exam Scores")
    plt.xlabel("Exam Score")
    plt.ylabel("Frequency")
    plot1_path = os.path.join(outputs_dir, "plot1_exam_score_dist.png")
    plt.savefig(plot1_path)
    plt.close()
    insights.append(
        "- The exam score distribution appears plotted and saved (plot1_exam_score_dist.png)."
    )

    # Plot 2: Correlation Matrix for Numeric Features
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    plot2_path = os.path.join(outputs_dir, "plot2_correlation_matrix.png")
    plt.savefig(plot2_path)
    plt.close()

    # Identify highest correlation with target
    if "Exam_Score" in corr.columns:
        corrs = corr["Exam_Score"].drop("Exam_Score").abs().sort_values(ascending=False)
        highest_corr_feat = corrs.index[0]
        highest_corr_val = corrs.iloc[0]
        insights.append(
            f"- Numeric feature with highest absolute correlation to Exam_Score is {highest_corr_feat} "
            f"({highest_corr_val:.2f}). plotted as (plot2_correlation_matrix.png)."
        )

    # Plot 3: Boxplot of Exam Scores by Parental Involvement
    if "Parental_Involvement" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Parental_Involvement", y="Exam_Score", data=df)
        plt.title("Exam Score by Parental Involvement")
        plt.xlabel("Parental Involvement")
        plt.ylabel("Exam Score")
        plot3_path = os.path.join(outputs_dir, "plot3_score_by_parental_inv.png")
        plt.savefig(plot3_path)
        plt.close()
        insights.append(
            "- Exam score boxplot grouped by Parental_Involvement saved (plot3_score_by_parental_inv.png)."
        )

    # 5. Save Insights Text
    insights_path = os.path.join(outputs_dir, "eda_insights.txt")
    with open(insights_path, "w") as f:
        f.write("\n".join(insights))
    print(f"Saved EDA insights to {insights_path}")


if __name__ == "__main__":
    CLEANED_DATA_PATH = "outputs/cleaned_data.csv"
    OUTPUTS_DIR = "outputs"
    run_eda(CLEANED_DATA_PATH, OUTPUTS_DIR)
