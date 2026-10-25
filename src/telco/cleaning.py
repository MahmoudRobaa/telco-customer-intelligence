import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco churn table.

    - TotalCharges: text -> float. Blank means "no bill yet", which only
      happens for tenure == 0 customers, so it becomes 0.0. A blank anywhere
      else is a real data problem and raises.
    - Churn: "Yes"/"No" -> 1/0.
    """
    out = df.copy()

    total = pd.to_numeric(out["TotalCharges"], errors="coerce")
    unexplained = total.isna() & (out["tenure"] > 0)
    if unexplained.any():
        raise ValueError(f"{unexplained.sum()} blank TotalCharges with tenure > 0")
    out["TotalCharges"] = total.fillna(0.0)

    churn = out["Churn"].map({"Yes": 1, "No": 0})
    if churn.isna().any():
        raise ValueError("Churn has values other than Yes/No")
    out["Churn"] = churn.astype(int)

    return out
