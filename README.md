# 💰 Predicting Loan Payback

## 🎯 Objective
The goal of this project is to predict whether a borrower will fully repay a loan (`loan_paid_back`) using structured financial and personal information. The predictions are evaluated using **ROC AUC** on Kaggle.

---
## 💯 Submission Score
Validation ROC AUC: **0.91904**

---
## 📊 Data
The dataset contains information about borrowers’ financial status, demographics, and loan details.

**Features include:**
- Numeric: `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`  
- Categorical: `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`  
- Engineered features:
  - `grade` → letter from `grade_subgrade`  :
    Represents the overall loan grade assigned by the lender (A, B, C, …).
    Encodes general creditworthiness; higher grades (A, B) usually indicate lower risk of default.
  - `subgrade` → number from `grade_subgrade`:
    Provides finer granularity within the loan grade (e.g., C1, C2, …).
    Captures subtle differences in borrower risk that the grade alone may not reflect. 
  - `loan_to_income` = `loan_amount / annual_income`:
Measures the relative size of the loan compared to the borrower’s income.
Higher ratios indicate a heavier financial burden, which can increase default risk.  
  - `debt_credit_interaction` = `debt_to_income_ratio * credit_score`:
    Combines existing debt burden (debt_to_income_ratio) with creditworthiness (credit_score).
    Helps the model understand how a borrower’s debt level interacts with their ability to repay.

  - `interest_burden` = `interest_rate * loan_amount`:
    Represents the total interest cost of the loan.
    Higher interest payments may increase the probability of late payment or default.

✅ Summary: These engineered features are designed to highlight financial stress, creditworthiness, and repayment risk in ways that raw features alone may not capture. They help the model make more accurate predictions about whether a borrower will repay the loan.

---

## 🏆 Competition
[Kaggle Competition: Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)  

- Evaluation metric: **Area Under the ROC Curve (AUC)**  
- Submission format: CSV with columns `id` and `loan_paid_back`  

---

## 🤔 Prediction Function

A function `predict_loan_paid_back` is provided to preprocess the test data, apply the trained XGBoost model, and generate the submission file:

```python
def predict_loan_paid_back(df, bst, label_mappings, submission_file="submission.csv"):
  
    df = df.copy()
    
    # Separing grade & subgrade
    df["grade"] = df["grade_subgrade"].str[0]
    df["subgrade"] = df["grade_subgrade"].str[1:].astype(int)
    df = df.drop("grade_subgrade", axis=1)

    # Encoding cat_cols with same train's LabelEncoders 
    for col, mapping in label_mappings.items():
      df[col] = df[col].map(mapping)

    # Feature engineering
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["debt_credit_interaction"] = df["debt_to_income_ratio"] * df["credit_score"]
    df["interest_burden"] = df["interest_rate"] * df["loan_amount"]
    
    X = df.drop(['id'], axis=1)

    # Reorder columns like train columns
    train_columns_order = [
        'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
        'gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose',
        'grade', 'subgrade', 'loan_to_income', 'debt_credit_interaction', 'interest_burden'
    ]
    X = X[train_columns_order]

    # Predictions
    y_pred = bst.predict_proba(X)[:,1]
    
    # Create Submission
    submission = pd.DataFrame({
        'id': df['id'],
        'loan_paid_back': y_pred
    })
    submission.to_csv(submission_file, index=False)
    print(f"Fichier {submission_file} créé ✅")
    
    return submission
```
## 🧠 How to Use

1️⃣ Clone the repository:
```bash
git clone https://github.com/MohamedAli1937/Loan-Payback-Prediction.git
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Run the notebook

4️⃣ Generate predictions on the test set:
```python
from scripts.predict_submission import predict_loan_paid_back
submission = predict_loan_paid_back(test_df, bst, label_mappings)

```
5️⃣ Submit the generated submission.csv to Kaggle. ✅️
