import pandas as pd
import os


def check_schema_and_clean(file_path, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Load the CSV safely
    print(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Extract basic info
    row_count, col_count = df.shape
    cols = list(df.columns)
    dtypes = df.dtypes.to_dict()
    duplicates = df.duplicated().sum()
    missing_vals = df.isnull().sum()

    # 3. Create Schema Report
    schema_report = [
        "### Schema Report ###\n",
        f"Row count: {row_count}",
        f"Column count: {col_count}",
        f"Columns: {cols}",
        f"Duplicates: {duplicates}\n",
        "Data Types:",
    ]
    for col, dtype in dtypes.items():
        schema_report.append(f"  {col}: {dtype}")

    # Check for obvious schema issues
    schema_issues = []
    if "Unnamed: 0" in cols or any(c.startswith("Unnamed") for c in cols):
        schema_issues.append("Warning: Found Unnamed columns (possible index).")

    if schema_issues:
        schema_report.append("\nSchema Issues:")
        schema_report.extend(schema_issues)
    else:
        schema_report.append("\nSchema Issues: None detected.")

    schema_report_path = os.path.join(outputs_dir, "schema_report.txt")
    with open(schema_report_path, "w") as f:
        f.write("\n".join(schema_report))
    print(f"Saved {schema_report_path}")

    # 4. Create Missingness Report and Strategy
    missingness_report = ["### Missingness Report ###\n", "Missing values per column:"]
    for col, missing in missing_vals.items():
        if missing > 0:
            missingness_report.append(
                f"  {col}: {missing} missing ({(missing/row_count)*100:.2f}%)"
            )

    if sum(missing_vals) == 0:
        missingness_report.append("  No missing values found.")

    missingness_report.append("\nImputation Strategy:")
    missingness_report.append(
        "- Numeric columns: Impute with median to handle skewness."
    )
    missingness_report.append(
        "- Categorical columns: Impute with mode (most frequent value)."
    )

    missingness_report_path = os.path.join(outputs_dir, "missingness_report.txt")
    with open(missingness_report_path, "w") as f:
        f.write("\n".join(missingness_report))
    print(f"Saved {missingness_report_path}")

    # 5. Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # 6. Save Cleaned Data
    cleaned_path = os.path.join(outputs_dir, "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned data to {cleaned_path}")


if __name__ == "__main__":
    DATA_PATH = "data/StudentPerformanceFactors.csv"
    OUTPUTS_DIR = "outputs"
    check_schema_and_clean(DATA_PATH, OUTPUTS_DIR)
