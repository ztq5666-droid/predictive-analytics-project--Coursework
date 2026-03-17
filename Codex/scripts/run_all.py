from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    scripts = [
        "scripts/task1_ingestion_schema_missingness.py",
        "scripts/task2_eda_insights.py",
        "scripts/task3_baseline_modeling.py",
        "scripts/task4_debug_broken_pipeline.py",
        "scripts/generate_benchmark_report.py",
    ]

    for script in scripts:
        print(f"Running {script}")
        subprocess.run([sys.executable, script], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
