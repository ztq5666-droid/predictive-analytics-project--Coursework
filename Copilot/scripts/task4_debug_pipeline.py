"""
Task 4: Debugging a Deliberately Broken Pipeline
"""
import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set up logging
os.makedirs('outputs', exist_ok=True)
logging.basicConfig(filename='outputs/task4_debug.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

TASK_SPEC = """
Create or use a broken pipeline, show the failure, diagnose, fix, and confirm the fix.
"""
SUCCESS_CRITERIA = """
- Broken pipeline created and failure shown
- Cause diagnosed and fixed
- Fix confirmed with evidence
"""

RANDOM_SEED = 42

def broken_pipeline():
    # Deliberately broken: tries to fit on non-numeric data
    df = pd.read_csv('outputs/cleaned_data.csv')
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    # No encoding of categoricals (intentional bug)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logging.info(f'Broken pipeline accuracy: {acc}')
    except Exception as e:
        logging.error(f'Pipeline failed: {e}')
        with open('outputs/broken_pipeline_failure.txt', 'w') as f:
            f.write(str(e))
        return False
    return True

def fixed_pipeline():
    # Fix: encode categoricals
    df = pd.read_csv('outputs/cleaned_data.csv')
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == object or y.dtype.name == 'category':
        y = pd.factorize(y)[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    with open('outputs/fixed_pipeline_accuracy.txt', 'w') as f:
        f.write(f'Accuracy after fix: {acc}\n')
    logging.info(f'Fixed pipeline accuracy: {acc}')
    return True

def main():
    # Run broken pipeline
    logging.info('Running broken pipeline...')
    failed = not broken_pipeline()
    if failed:
        logging.info('Diagnosing and fixing pipeline...')
        fixed = fixed_pipeline()
        if fixed:
            logging.info('Pipeline fixed and confirmed.')
        else:
            logging.error('Pipeline fix failed.')
    else:
        logging.warning('Broken pipeline did not fail as expected.')

if __name__ == '__main__':
    main()
