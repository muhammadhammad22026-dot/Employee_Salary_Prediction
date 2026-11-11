import pandas as pd
from pathlib import Path

# 1. Load The file to 'Preprocess for Machine Learning'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Messy_Employee_dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Cleaned_data.csv"

df = pd.read_csv(RAW_DATA_PATH)
df_clean = df.copy()

# 2. Date cleaning & dropping invalid dates
df_clean["Join_Date"] = pd.to_datetime(
    df_clean["Join_Date"], errors="coerce"
)
df_clean = df_clean.dropna(subset=["Join_Date"])

# 3. Drop unnecessary/sensitive columns
col_to_drop = [
    'Employee_ID', 
    'First_Name', 
    'Last_Name', 
    'Email', 
    'Phone'
]
df_clean = df_clean.drop(
    columns=col_to_drop, 
    errors='ignore'
)

# 4. Numeric conversions
df_clean["Age"] = pd.to_numeric(
    df_clean["Age"], errors="coerce"
)

df_clean["Salary"] = pd.to_numeric(
    df_clean["Salary"], errors="coerce"
)

# 5. Handle missing values (Choose either dropping or imputation)
df_clean = df_clean.dropna(subset=["Salary"])

# 6. Remove duplicate rows
df_clean = df_clean.drop_duplicates()

# 7. Optional: Split Department_Region into separate columns
df_clean[["Department", "Region"]] = (
    df_clean["Department_Region"]
    .str.split("-", n=1, expand=True)
)

df_clean = df_clean.drop(
    columns=['Department_Region']
)

# 8. Boolean conversion (Ensure proper boolean type)
df_clean['Remote_Work'] = df_clean['Remote_Work'].astype(bool)

# 9. To save the data
df_clean.to_csv("data/Cleaned_data.csv", index=False)

# 10. Display results
print("Dataset cleaning completed!")

print("\nDataset Shape:")
print(df_clean.shape)

print("\nMissing Values:")
print(df_clean.isnull().sum())

print("\nDataset Information:")
print(df_clean.info())

print("\nStatistical Summary:")
print(df_clean.describe())
