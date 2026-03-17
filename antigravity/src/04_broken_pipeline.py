import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def run_debugging_demo(cleaned_data_path, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)
    log_path = os.path.join(outputs_dir, "debugging_log.txt")
    logs = []

    def log_print(msg):
        print(msg)
        logs.append(msg)

    log_print("### Task 4: Debugging a Broken Pipeline ###\n")

    # 1. Load Data
    df = pd.read_csv(cleaned_data_path)
    # Simplify by only using numeric columns to demonstrate the point clearly
    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

    X = numeric_df.drop(columns=["Exam_Score"])
    y = numeric_df["Exam_Score"]

    log_print("SCENARIO: A developer tries to scale the features using StandardScaler,")
    log_print(
        "but they mistakenly apply fit_transform to the ENTIRE dataset before splitting it."
    )
    log_print(
        "This causes DATA LEAKAGE because information about the test set distribution"
    )
    log_print("influences the scaling of the training set.\n")

    # --- BROKEN PIPELINE ---
    log_print("--- Running Broken Pipeline ---")
    try:
        scaler_broken = StandardScaler()
        # ERROR: Scaling before split!
        X_scaled_broken = scaler_broken.fit_transform(X)

        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_scaled_broken, y, test_size=0.2, random_state=42
        )

        model_broken = LinearRegression()
        model_broken.fit(X_train_b, y_train_b)
        preds_broken = model_broken.predict(X_test_b)
        rmse_broken = root_mean_squared_error(y_test_b, preds_broken)

        log_print(f"Broken Pipeline RMSE: {rmse_broken:.4f}")
        log_print(
            "Diagnosis: The RMSE might look 'fine' or artificially low, but the methodology is flawed."
        )
        log_print(
            "The test set was not truly held out since its mean/variance influenced the scaler.\n"
        )
    except Exception as e:
        log_print(f"Pipeline failed with error: {e}\n")

    # --- FIXED PIPELINE ---
    log_print("--- Running Fixed Pipeline ---")
    try:
        # Correctly split first
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit scaler ONLY on training data
        scaler_fixed = StandardScaler()
        X_train_scaled_fixed = scaler_fixed.fit_transform(X_train_f)

        # Transform test data using the fitted scaler
        X_test_scaled_fixed = scaler_fixed.transform(X_test_f)

        model_fixed = LinearRegression()
        model_fixed.fit(X_train_scaled_fixed, y_train_f)
        preds_fixed = model_fixed.predict(X_test_scaled_fixed)
        rmse_fixed = root_mean_squared_error(y_test_f, preds_fixed)

        log_print(f"Fixed Pipeline RMSE: {rmse_fixed:.4f}")
        log_print("Fix Confirmed: Scaler was fitted exclusively on training data.")
        log_print(
            "This ensures evaluating the model reflects true generalization to unseen data.\n"
        )
    except Exception as e:
        log_print(f"Fixed Pipeline failed with error: {e}\n")

    with open(log_path, "w") as f:
        f.write("\n".join(logs))
    log_print(f"Log saved to {log_path}")


if __name__ == "__main__":
    CLEANED_DATA_PATH = "outputs/cleaned_data.csv"
    OUTPUTS_DIR = "outputs"
    run_debugging_demo(CLEANED_DATA_PATH, OUTPUTS_DIR)
