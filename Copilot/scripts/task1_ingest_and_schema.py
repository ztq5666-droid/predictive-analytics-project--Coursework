"""
Task 1: Dataset ingestion, schema checks, and missingness handling
"""
import pandas as pd
import numpy as np
import logging
import os

# Set up logging
os.makedirs('outputs', exist_ok=True)
logging.basicConfig(filename='outputs/task1_ingestion.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Task specification
TASK_SPEC = """
Load the dataset, check schema, handle missing values.
"""
SUCCESS_CRITERIA = """
- Dataset loaded without error
- Schema and missingness reports saved
- Missing values handled with documented strategy
"""

def main():
    # Load dataset
    try:
        df = pd.read_csv('data/StudentPerformanceFactors.csv')
        logging.info('Dataset loaded successfully.')
    except Exception as e:
        logging.error(f'Failed to load dataset: {e}')
        raise

    # Report basic info
    row_count, col_count = df.shape
    col_names = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    duplicate_count = df.duplicated().sum()
    missing = df.isnull().sum().to_dict()
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    inconsistent_types = []
    for col in df.columns:
        if df[col].apply(type).nunique() > 1:
            inconsistent_types.append(col)

    # Save schema report
    schema_report = {
        'row_count': row_count,
        'col_count': col_count,
        'col_names': col_names,
        'dtypes': dtypes,
        'duplicate_count': int(duplicate_count),
        'unnamed_cols': unnamed_cols,
        'inconsistent_types': inconsistent_types
    }
    pd.Series(schema_report).to_json('outputs/schema_report.json', indent=2)
    logging.info(f'Schema report saved. {schema_report}')

    # Save missingness report
    pd.Series(missing).to_json('outputs/missingness_report.json', indent=2)
    logging.info(f'Missingness report saved. {missing}')

    # Handle missing values
    missing_strategy = {}
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64]:
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                missing_strategy[col] = f'Filled with median ({fill_value})'
            else:
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'missing'
                df[col].fillna(fill_value, inplace=True)
                missing_strategy[col] = f'Filled with mode ({fill_value})'
    pd.Series(missing_strategy).to_json('outputs/missingness_strategy.json', indent=2)
    logging.info(f'Missing values handled: {missing_strategy}')

    # Save cleaned data
    df.to_csv('outputs/cleaned_data.csv', index=False)
    logging.info('Cleaned data saved.')

if __name__ == '__main__':
    main()
