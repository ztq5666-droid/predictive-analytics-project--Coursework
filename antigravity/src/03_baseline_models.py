import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def build_and_evaluate(cleaned_data_path, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading cleaned data from {cleaned_data_path}")
    df = pd.read_csv(cleaned_data_path)

    # 2. Define Features and Target
    target = "Exam_Score"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify column types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # 3. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # 4. Train/Test Split (Avoid Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Build Models
    models = {
        "DummyRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        ),
    }

    metrics = {}
    predictions_df = pd.DataFrame({"Actual": y_test})

    # 6. Train and Evaluate
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        predictions_df[f"{name}_Pred"] = preds

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        metrics[name] = {"RMSE": float(rmse), "R2": float(r2)}
        print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # 7. Save Metrics, Configuration, and Predictions
    config = {
        "target": target,
        "features": {"numeric": numeric_features, "categorical": categorical_features},
        "test_size": 0.2,
        "random_state": 42,
        "models_used": list(models.keys()),
    }

    output_data = {"config": config, "metrics": metrics}

    metrics_path = os.path.join(outputs_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved metrics and config to {metrics_path}")

    predictions_path = os.path.join(outputs_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    CLEANED_DATA_PATH = "outputs/cleaned_data.csv"
    OUTPUTS_DIR = "outputs"
    build_and_evaluate(CLEANED_DATA_PATH, OUTPUTS_DIR)
