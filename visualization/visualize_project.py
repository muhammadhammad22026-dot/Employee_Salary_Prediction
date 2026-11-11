from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Messy_Employee_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def create_project_visualizations(data_path=RAW_DATA_PATH):
    df = pd.read_csv(data_path)

    # Split combined department and region for analysis.
    split_values = df["Department_Region"].astype(str).str.split("-", n=1, expand=True)
    df["Department"] = split_values[0].str.strip()
    df["Region"] = split_values[1].str.strip()

    # 1. Missing values overview.
    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(missing.index, missing.values)
    ax.invert_yaxis()
    ax.set(title="Missing Values by Column", xlabel="Missing value count", ylabel="Column")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=160)
    plt.close(fig)

    # 2. Salary distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    salary_values = pd.to_numeric(df["Salary"], errors="coerce").dropna()
    ax.hist(salary_values, bins=30, edgecolor="black")
    ax.set(title="Employee Salary Distribution", xlabel="Salary", ylabel="Employee count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "salary_distribution.png", dpi=160)
    plt.close(fig)

    # 3. Salary by department.
    fig, ax = plt.subplots(figsize=(10, 5))
    department_groups = [group["Salary"].dropna().values for _, group in df.groupby("Department")]
    department_names = [name for name, _ in df.groupby("Department")]
    ax.boxplot(department_groups, tick_labels=department_names)
    ax.set(title="Salary Distribution by Department", xlabel="Department", ylabel="Salary")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "salary_by_department.png", dpi=160)
    plt.close(fig)

    # 4. Salary by performance category.
    performance_order = ["Poor", "Average", "Good", "Excellent"]
    fig, ax = plt.subplots(figsize=(9, 5))
    performance_groups = [
        df.loc[df["Performance_Score"] == score, "Salary"].dropna().values
        for score in performance_order
    ]
    ax.boxplot(performance_groups, tick_labels=performance_order)
    ax.set(title="Salary by Performance Score", xlabel="Performance score", ylabel="Salary")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "salary_by_performance.png", dpi=160)
    plt.close(fig)

    print(f"Visualizations saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    create_project_visualizations()
