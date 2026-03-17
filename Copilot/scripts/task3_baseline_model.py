"""
Task 3: Baseline Model Training and Evaluation Harness
"""
import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report

# Set up logging
os.makedirs('outputs', exist_ok=True)
logging.basicConfig(filename='outputs/task3_modeling.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

TASK_SPEC = """
Select a sensible prediction target, build baseline models, evaluate, and save all outputs.
"""
SUCCESS_CRITERIA = """
- Target column selected and justified
- Models trained and evaluated
- Metrics, config, and predictions saved
"""

RANDOM_SEED = 42

def main():
    # Load cleaned data
    try:
        df = pd.read_csv('outputs/cleaned_data.csv')
        logging.info('Cleaned data loaded.')
    except Exception as e:
        logging.error(f'Failed to load cleaned data: {e}')
        raise

    # Target selection (choose last column if ambiguous)
    target = df.columns[-1]
    logging.info(f'Target column selected: {target}')
    with open('outputs/model_target.txt', 'w') as f:
        f.write(f'Target column: {target}\n')

    # Check if classification or regression
    if df[target].nunique() <= 10 and df[target].dtype in [object, 'category', 'bool', np.int64]:
        task_type = 'classification'
    else:
        task_type = 'regression'
    logging.info(f'Task type: {task_type}')
    with open('outputs/model_task_type.txt', 'w') as f:
        f.write(f'Task type: {task_type}\n')

    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == object or y.dtype.name == 'category':
        y = pd.factorize(y)[0]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    logging.info(f'Train/test split: {X_train.shape}, {X_test.shape}')

    # Baseline model
    if task_type == 'classification':
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {}
    if task_type == 'classification':
        y_pred_label = (y_pred > 0.5).astype(int) if y_pred.ndim else y_pred
        metrics['accuracy'] = accuracy_score(y_test, y_pred_label)
        metrics['f1'] = f1_score(y_test, y_pred_label, average='weighted')
        metrics['classification_report'] = classification_report(y_test, y_pred_label)
    else:
        # For older sklearn, no 'squared' argument; RMSE = sqrt(MSE)
        mse = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(mse)
    pd.Series(metrics).to_json('outputs/model_metrics.json', indent=2)
    logging.info(f'Metrics saved: {metrics}')

    # Save predictions
    pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv('outputs/model_predictions.csv', index=False)
    logging.info('Predictions saved.')

    # Save config
    config = {
        'random_seed': RANDOM_SEED,
        'model': str(model),
        'task_type': task_type,
        'target': target
    }
    pd.Series(config).to_json('outputs/model_config.json', indent=2)
    logging.info(f'Config saved: {config}')

if __name__ == '__main__':
    main()
