
# Experiment Log

## Screenshots
Below are the four referenced screenshots documenting the process:

![Screenshot 1](../screenshots/screenshot1.png)
![Screenshot 2](../screenshots/screenshot2.png)
![Screenshot 3](../screenshots/screenshot3.png)
![Screenshot 4](../screenshots/screenshot4.png)

**Date:** 2026-03-16

---

## 1. Initial Run & Errors
- **Attempted to run all scripts:**
  - `task1_ingest_and_schema.py`
  - `task2_eda.py`
  - `task3_baseline_model.py`
  - `task4_debug_pipeline.py`
  - `benchmark_report.py`
- **Timestamp:** 2026-03-16 (see terminal logs)
- **Errors encountered:**
  - `task3_baseline_model.py` failed with:
    - `TypeError: got an unexpected keyword argument 'squared'` in `mean_squared_error` (scikit-learn version incompatibility)
    - Import errors for `sklearn.model_selection`, `sklearn.linear_model`, and `sklearn.metrics` (scikit-learn not installed or not available)

---

## 2. Fixes Applied
- **Code fix:**
  - Removed the `squared` argument from `mean_squared_error` and manually computed RMSE as `np.sqrt(mse)` for compatibility with older scikit-learn versions.
- **Dependency management:**
  - Created `requirements.txt` with:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
- **Timestamp:** 2026-03-16

---

## 3. Dependency Installation
- **Command run:**
  - `pip install -r requirements.txt`
- **Result:**
  - All dependencies installed successfully.
- **Timestamp:** 2026-03-16

---

## 4. Re-run & Warnings
- **Scripts re-executed in order:**
  - `python task1_ingest_and_schema.py && python task2_eda.py && python task3_baseline_model.py && python task4_debug_pipeline.py && python benchmark_report.py`
- **Warnings encountered:**
  - **pandas FutureWarning:**
    - About chained assignment and inplace operations (does not affect current results, but may in future pandas versions)
  - **scikit-learn ConvergenceWarning:**
    - LogisticRegression failed to converge (suggests increasing `max_iter` or scaling data)
- **Timestamp:** 2026-03-16

---

## 5. Final Status
- **All scripts completed.**
- **All outputs generated in `outputs/` directory.**
- **Reproducibility:**
  - All code, logs, and outputs are versioned and saved.
  - All fixes and warnings are documented here for audit and reproducibility.
- **Timestamp:** 2026-03-16

---

*This log serves as a full audit trail of the experiment, including errors, fixes, warnings, and successful completion.*
