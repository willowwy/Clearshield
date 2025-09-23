import pandas as pd
import os

# Create output directory
output_dir = 'fraud_victim_histories'
os.makedirs(output_dir, exist_ok=True)

# Load transaction data
df = pd.read_csv('../transaction_data.csv')

# Create fraud indicator
df['Is_Fraud'] = df['Fraud Adjustment Indicator'].notna() & (df['Fraud Adjustment Indicator'].str.strip() != '')

# Get unique fraud victims
fraud_victims = df[df['Is_Fraud']]['Member ID'].unique()

# Extract all transaction history for each fraud victim
for member_id in fraud_victims:
    member_data = df[df['Member ID'] == member_id]
    filename = f'{output_dir}/fraud_victim_{member_id}_history.csv'
    member_data.to_csv(filename, index=False)

print(f"Created {len(fraud_victims)} CSV files in '{output_dir}' folder")