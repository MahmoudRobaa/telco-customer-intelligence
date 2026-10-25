# Data assessment

## 0. Source decision
Primary source: the Kaggle CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, 7,043 x 21). The IBM xlsx is not used, and the two files were not reconciled.
Cost of that choice: no `Churn Reason` (why customers leave) and no `CLTV`. Revenue at risk will use `MonthlyCharges`.

## A. Structure, types, missing values
- **Shape:** 7,043 rows x 21 columns. 17 text columns, 3 numeric (`SeniorCitizen`, `tenure`, `MonthlyCharges`), plus `TotalCharges`.
- **Wrong type:** `TotalCharges` loads as text although it is a dollar amount. It contains blank strings (" "), and one non-numeric value is enough to make pandas read the whole column as text.
- **Missing values:** `isna()` reports 0, which is misleading. The blanks are not NaN, so they are invisible until you convert with `pd.to_numeric(..., errors="coerce")`. That finds **11 hidden missing values**.
- **Who they are:** all 11 have `tenure == 0` (and all 11 customers with tenure 0 are in this set). All are still active (`Churn = No`), and 10 of 11 are on two-year contracts. They are customers who signed up this period and have not received a first bill yet. `TotalCharges` is not missing at random: it is undefined because no bill exists.
- **Handling:** do not impute a typical value. Set it to 0 (nothing billed yet) or drop the 11 rows, and record the choice. Dropping costs 0.16% of rows and no churners.
- **Duplicates:** `customerID` is unique. 22 rows are identical once the ID is ignored. With 19 mostly categorical columns, coincidental twins are expected, and distinct IDs mean they are different customers. Keep them.
- **Encoding:** `SeniorCitizen` is 0/1. Every other binary column is "Yes"/"No". `Churn` is "Yes"/"No" and will need a 0/1 target.
- **Redundant levels:** `MultipleLines` has "No phone service" and the six internet add-on columns have "No internet service". Both are information already held in `PhoneService` and `InternetService`.
