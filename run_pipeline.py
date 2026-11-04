"""
End-to-End Execution Pipeline for Bike-Sharing Demand Analysis & Warehousing
"""

import subprocess
import sys
import os

PIPELINE_STEPS = [
    ("1. Data Preprocessing", "src/preprocessing.py"),
    ("2. Data Normalization", "src/normalization.py"),
    ("3. Data Reduction (PCA & Feature Selection)", "src/data_reduction.py"),
    ("4. Data Warehouse (Star Schema in SQLite)", "src/datawarehouse.py"),
    ("5. OLAP Operations (Roll-up, Drill-down, Slice, Dice)", "src/olapoperations.py"),
    ("6. Demand Classification (KNN, Decision Tree, Rule-based)", "src/demand_classification.py"),
    ("7. Association Rule Mining (Apriori & FP-Growth)", "src/association_mining.py"),
    ("8. Usage Clustering (K-Means, Hierarchical, DBSCAN)", "src/clustering.py"),
]

def main():
    print("=" * 60)
    print("STARTING BIKE SHARING DEMAND ANALYSIS PIPELINE")
    print("=" * 60)

    for step_num, (title, script_path) in enumerate(PIPELINE_STEPS, 1):
        print(f"\n[{step_num}/{len(PIPELINE_STEPS)}] Running {title}...")
        print("-" * 50)
        
        result = subprocess.run([sys.executable, script_path])
        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline failed at step: {title} ({script_path})")
            sys.exit(result.returncode)
        
        print(f"[SUCCESS] Finished {title}")

    print("\n" + "=" * 60)
    print("ALL PIPELINE MODULES EXECUTED SUCCESSFULLY!")
    print("Results and outputs are available in the 'outputs/' directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
