"""
Task 2: Exploratory Data Analysis (EDA) and Insight Generation
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set up logging
os.makedirs('outputs', exist_ok=True)
logging.basicConfig(filename='outputs/task2_eda.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

TASK_SPEC = """
Perform EDA: summary statistics, informative plots, and concise, data-grounded insights.
"""
SUCCESS_CRITERIA = """
- Summary statistics saved
- Plots generated and saved
- Insights written and saved
"""

def main():
    # Load cleaned data
    try:
        df = pd.read_csv('outputs/cleaned_data.csv')
        logging.info('Cleaned data loaded.')
    except Exception as e:
        logging.error(f'Failed to load cleaned data: {e}')
        raise

    # Summary statistics
    summary = df.describe(include='all').transpose()
    summary.to_csv('outputs/eda_summary_statistics.csv')
    logging.info('Summary statistics saved.')

    # Plots
    plot_paths = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        path = f'outputs/hist_{col}.png'
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Value counts of {col}')
        path = f'outputs/bar_{col}.png'
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)
    logging.info(f'Plots saved: {plot_paths}')

    # Correlation heatmap
    plt.figure(figsize=(10,8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    logging.info('Correlation heatmap saved.')

    # Insights
    insights = []
    # Example: Highest missing column
    missing = df.isnull().sum()
    if missing.max() > 0:
        col = missing.idxmax()
        insights.append(f'Column with most missing values: {col} ({missing[col]})')
    # Example: Most variable column
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_cols.empty:
        var_col = numeric_cols.var().idxmax()
        insights.append(f'Most variable numeric column: {var_col}')
    # Example: Most frequent value in categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        mode = df[col].mode().iloc[0]
        freq = df[col].value_counts().iloc[0]
        insights.append(f'Most frequent value in {col}: {mode} ({freq} times)')
    with open('outputs/eda_insights.txt', 'w') as f:
        for line in insights:
            f.write(line + '\n')
    logging.info(f'Insights written: {insights}')

if __name__ == '__main__':
    main()
