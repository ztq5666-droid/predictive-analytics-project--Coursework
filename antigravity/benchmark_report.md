# Student Performance Analysis Benchmark Report

## Overview
This report summarizes the objectives, execution, deliverables, and reproducibility for the four tasks associated with analyzing the `StudentPerformanceFactors.csv` dataset.

---

## Task 1: Dataset Ingestion + Schema Checks + Missingness Handling

**Task Specs & Success Criteria:**
1. Safely load the CSV and generate a schema report.
2. Check for unexpected issues.
3. Detect, handle, and document missing values.
4. Output: `outputs/schema_report.txt`, `outputs/missingness_report.txt`, and `outputs/cleaned_data.csv`.

**Results:**
- The CSV (`6607` rows, `20` columns) was ingested successfully.
- No duplicate records were found.
- Missing values detected: `Teacher_Quality` (78 missing, 1.18%), `Parental_Education_Level` (90 missing, 1.36%), and `Distance_from_Home` (67 missing, 1.01%).
- **Imputation Strategy:** Continuous variables imputed with medians to withstand outliers. Categorical columns imputed with modes.
- Output artifacts generated as expected.

---

## Task 2: EDA and Insight Generation

**Task Specs & Success Criteria:**
1. Generate descriptive summary statistics.
2. Produce multiple informative plots and save them (.png).
3. Write concise textual insights grounded in the exact figures.
4. Output: `summary_statistics.csv`, `.png` plots, and `eda_insights.txt`.

**Results:**
- Generated three plots:
  - `plot1_exam_score_dist.png`: Examining the target distribution.
  - `plot2_correlation_matrix.png`: Verifying relationships among numeric inputs.
  - `plot3_score_by_parental_inv.png`: Displaying categorical differences.
- **Insights Recorded:** The average Exam Score across the dataset is ~67.24 (Min: 55, Max: 100). The numeric feature most linearly correlated with `Exam_Score` is `Attendance` (value roughly 0.58).

---

## Task 3: Baseline Model Training + Evaluation

**Task Specs & Success Criteria:**
1. Use `Exam_Score` as target. Justification: It represents the holistic learning outcome based on the feature sets.
2. Build baseline and stronger models on an 80/20 train/test split.
3. Evaluate using appropriate continuous metrics (RMSE, R2).
4. Output: `outputs/metrics.json` and `outputs/predictions.csv`.

**Results:**
- **Dummy Regressor (Mean Baseline):**
  - RMSE: `3.7611`
  - R2: `-0.0007` (predicts target mean unconditionally)
- **Random Forest Regressor (Stronger Baseline):**
  - RMSE: `2.1560`
  - R2: `0.6712`
- **Conclusion:** The ensemble approach significantly outperformed the naive baseline, describing ~67% of the variance in exam scores on unseen tests. Data leakage was systematically avoided using a scikit-learn pipeline spanning imputation/scaling constraints. 

---

## Task 4: Debugging a Deliberately Broken Pipeline

**Task Specs & Success Criteria:**
1. Implement a pipeline featuring a data leakage issue.
2. Formally observe and report the fault.
3. Diagnose and fix the underlying modeling error.
4. Output: `outputs/debugging_log.txt`.

**Results:**
- **Failure Exhibited:** We deliberately instantiated a `StandardScaler()` and fitted it synchronously over all target rows (`fit_transform(X)`) prior to train/test splitting.
- **Diagnosis:** Calling `fit` on test records grants an unfair advantage by revealing the mean and variance of unseen observations to the global algorithm logic.
- **Correction:** We split the data *first*. Then we strictly called `fit_transform()` on `X_train`, and subsequently called `transform()` on `X_test`.
- Note: Both RMSE scores displayed ~`2.2507` on our deterministic seed. Though numerically identical here for LinearRegression, the explicit diagnosis and programmatic decoupling ensures mathematical correctness.

---

## Reproducibility Notes
**Execution Control:** Random state variables (`random_state=42`) were utilized wherever probabilistic behavior emerged (train/test splits and random forest bootstrap samples).
**Environment:** The code operates natively resolving `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.
**Paths:** Folder pathways rigidly adhere to local project mapping without esoteric hardcoding relying on undefined nested directories.
