# src/03_feature_engineering/user_features.py

import pandas as pd


def generate_user_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Generate user-level 03_feature_engineering:
    1. Age group (categorical: 18-30, 30-50, 50-65, 65+)
    2. Account Type (categorical: keep original account type values)
    """

    # --- Age group bucketing ---
    def categorize_age(age: int) -> str:
        if pd.isnull(age):
            return "unknown"
        if 18 <= age < 30:
            return "18-30"
        elif 30 <= age < 50:
            return "30-50"
        elif 50 <= age < 65:
            return "50-65"
        elif age >= 65:
            return "65+"
        else:
            return "unknown"

    transactions["Age Group"] = transactions["Member Age"].apply(categorize_age)

    # --- Keep Account Type as-is (strip spaces for consistency) ---
    transactions["Account_Type_Clean"] = transactions["Account Type"].astype(str).str.strip()

    # --- Aggregate by user (Member ID is the unique user identifier) ---
    user_features = transactions.groupby("Member ID").agg(
        Age_Group=("Age Group", "first"),
        Account_Type=("Account_Type_Clean", "first"),
    ).reset_index()

    return user_features
