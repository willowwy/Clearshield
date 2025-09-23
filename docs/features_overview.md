# Features Overview

This document provides an overview of the features used in the **Clearshield** fraud detection system.  
Features are grouped into categories such as **User**, **Transaction Amount**, **Time**, **Channel**, **Merchant**, **Behavior**, and **Fraud labels**.

---

## Feature Table

| Category               | Feature Name             | Type        | Example    | Description                                                                    | Potential Use in Fraud Detection                                                                                                           |
| ---------------------- | ------------------------ | ----------- | ---------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **User**               | Age group                | categorical | 30–50      | Age bucket of the account holder                                               | Different age groups have different fraud rates. <br> **18–30:** 0.110% <br> **30–50:** 0.046% <br> **50–65:** 0.067% <br> **65+:** 0.032% |
|                        | Account Type             | categorical | nonprofit  | Whether the account is nonprofit, employee, corporation, etc.                  | Certain account types (e.g., nonprofit) may be more prone to fraud                                                                         |
| **Transaction Amount** | Transaction amount       | float       | 55         | Raw transaction value in USD                                                   | Base feature for modeling                                                                                                                  |
|                        | Fraud-prone amount range | categorical | 50 or 500  | Bucketed ranges: whether amount is at risk-prone values                        | Specific values (e.g., 50, 500) can be frequently exploited by fraudsters                                                                  |
|                        | High-value flag          | boolean     | FALSE      | TRUE if `amount > mean + 3×std` for the account                                | Detects anomalies relative to user’s history                                                                                               |
|                        | Is higher than neighbors | boolean     | FALSE      | TRUE if transaction is much higher than surrounding transactions               | Useful anomaly signal                                                                                                                      |
|                        | Is consecutive           | boolean     | FALSE      | TRUE if many similar transactions occur in short time                          | Indicates “smurfing” (split small-amount frauds)                                                                                           |
| **Time**               | Night transaction        | boolean     | TRUE       | TRUE if transaction time ∈ [0:00–5:00]                                         | Night transactions are often riskier                                                                                                       |
| **Channel**            | Source type              | categorical | POS        | Channel used (POS, Online Banking, ATM, etc.)                                  | Fraud may correlate with unusual channels                                                                                                  |
|                        | Action type              | categorical | Withdrawal | Type of transaction action                                                     | Rapid withdrawals are commonly linked to fraud                                                                                             |
| **Merchant**           | Merchant category        | categorical | E-commerce | Derived from merchant description (e.g., Apple → E-commerce, Uber → Transport) | Fraud risk differs across merchant categories                                                                                              |
| **Behavior**           | Transaction count (7d)   | int         | 12         | Number of transactions in past 7 days                                          | Spikes in activity may indicate compromised account                                                                                        |
|                        | Unique merchants (30d)   | int         | 25         | Number of distinct merchants in past 30 days                                   | Fraudsters often target many merchants quickly                                                                                             |
| **Fraud Label**        | Fraud flag (exists)      | boolean     | TRUE       | Indicator if transaction is confirmed fraud                                    | Ground truth for supervised training                                                                                                       |

---

## Notes

- Features in **User** category are aggregated per-user and stored in `data/processed/features/`.
- Features in **Transaction Amount, Time, Channel, Merchant, Behavior** are primarily per-transaction and may also be aggregated for sequence models.
- Fraud label (`Fraud flag`) is used only for training/evaluation, not as a model input.
