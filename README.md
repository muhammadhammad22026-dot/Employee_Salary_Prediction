# Employee Salary Prediction

A machine learning project that explores employee data and trains a **Random Forest Regression** model to estimate employee salary from available employee attributes.

> **Data privacy:** The source CSV contains employee-style identifiers and contact fields. It is intentionally excluded from Git tracking. Do not publish raw employee records or personal information. Place your authorized local copy at `data/raw/Messy_Employee_dataset.csv`.

## Project objectives

- Inspect and clean a messy employee dataset.
- Remove direct identifiers and contact information from model inputs.
- Derive employee tenure from `Join_Date`.
- Explore salary patterns with visualizations.
- Train and evaluate a salary regression model using the observed `Salary` column.

## Repository structure

```text
Employee-Salary-Prediction/
├── data/
│   ├── raw/                 # Put Messy_Employee_dataset.csv here (not committed)
│   └── processed/           # Cleaned_data.csv is generated here (not committed)
├── notebooks/
│   └── Employee_Salary_Analysis.ipynb
├── src/
│   ├── Cleaning.py
│   └── Model.py
├── models/                  # Trained model artifact (not committed)
├── visualizations/
│   └── visualize_project.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Dataset

The uploaded dataset inspected during project setup has 1,020 rows and 12 columns. It includes:

- `Age`
- `Department_Region`
- `Status`
- `Join_Date`
- `Salary` (prediction target)
- `Performance_Score`
- `Remote_Work`
- Direct identifiers/contact fields, which are removed during cleaning

The inspected file had missing values in `Age` (211) and `Salary` (24). The cleaning script drops rows with missing target salary but retains missing age values so they can be imputed using the training data only.

## Setup

Use Python 3.10 or newer.

```bash
git clone <https://github.com/muhammadhammad22026-dot/Employee_Salary_Prediction>
cd Employee-Salary-Prediction

python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your authorized dataset at:

```text
data/raw/Messy_Employee_dataset.csv
```

## Run the project

Run commands from the repository root.

### 1. Clean the data

```bash
python src/Cleaning.py
```

Output:

```text
data/processed/Cleaned_data.csv
```

### 2. Generate visualizations

```bash
python visualizations/visualize_project.py
```

The script saves these charts in `visualizations/`:

- `missing_values.png`
- `salary_distribution.png`
- `salary_by_department.png`
- `salary_by_performance.png`

### 3. Train and evaluate the model

```bash
python src/Model.py
```

The model uses a scikit-learn `Pipeline` with:
- Median imputation for numeric features
- Most-frequent imputation for categorical features
- One-hot encoding for categorical features
- `RandomForestRegressor` for salary prediction

It reports:
- **R² score:** goodness of fit on the held-out test set
- **MAE:** average absolute prediction error in the salary units

The trained model is saved to:

```text
models/employee_salary_model.joblib
```

### 4. Open the explanatory notebook

```bash
jupyter notebook notebooks/Employee_Salary_Analysis.ipynb
```

The notebook explains the dataset, cleaning decisions, feature engineering, preprocessing pipeline, model training, evaluation, and project visualizations.

## Validation run (example, not a guaranteed result)

The project scripts were executed against the supplied CSV during setup. The first run produced an R² score of approximately **-0.0504** and an MAE of approximately **17,079.85** (in the dataset salary units). This is a weak result: on that split, the model did not outperform a simple test-set-mean baseline by R². Investigate feature relationships, target quality, and data-generation patterns before presenting the model as useful. Results can change with data or environment.

## Modeling notes and limitations

- This is a regression task because `Salary` is a continuous numeric target.
- The model predicts the observed `Salary` column. It does **not** create a synthetic salary target.
- `Age` is imputed inside the training pipeline, preventing the imputer from learning from the test split.
- `Join_Date` is transformed into `Tenure_year`; the original date is not used directly as a model feature.
- The train/test split is random (80% train, 20% test) with `random_state=42`.
- R² can be negative when predictions are worse than predicting the test-set mean. Results should be interpreted using the actual evaluation output, not assumed in advance.
- The dataset may be synthetic or may not represent a real hiring/payroll population. Model performance does not establish fairness, causality, or suitability for real compensation decisions.
- Salary prediction can encode historical inequities. Do not use this model to make employment, hiring, promotion, or compensation decisions about real people without rigorous validation, governance, and human review.

## Technologies

Python, pandas, scikit-learn, NumPy, Matplotlib, Seaborn, Jupyter Notebook, and joblib.

## Author

**Muhammad Hammad**  
Aspiring Data Scientist | BS Data Science Student

- LinkedIn: https://www.linkedin.com/in/muhammad-hammad-ml2026/
- GitHub: https://github.com/muhammadhammad22026-dot

## License

This project is licensed under the MIT License. See `LICENSE` for details.
