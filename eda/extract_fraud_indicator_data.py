import pandas as pd

df = pd.read_csv('../transaction_data.csv')
fraud_data = df[df['Fraud Adjustment Indicator'].notna()]
fraud_data.to_csv('./fraud_data.csv', index=False)