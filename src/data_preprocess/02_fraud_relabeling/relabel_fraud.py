import pandas as pd
import os
import re
from datetime import timedelta


def extract_fraud_date_from_description(description, adjustment_date):
    """Extract date from fraud adjustment description"""
    if pd.isna(description):
        return None

    match = re.search(r'(\d{1,2}/\d{1,2})', str(description))
    if match:
        try:
            month, day = map(int, match.group(1).split('/'))
            return pd.Timestamp(adjustment_date.year, month, day)
        except:
            pass
    return None


def match_fraud_adjustment_to_transaction(fraud_row, df, used_indices):
    """Find matching original transaction for fraud adjustment"""
    fraud_amount = abs(fraud_row['Amount'])
    fraud_date = fraud_row['PostDate']

    # Try to extract specific date from description
    target_date = extract_fraud_date_from_description(fraud_row['Fraud Adjustment Indicator'], fraud_date)

    if target_date:
        # Exact date matching
        candidates = df[
            (df['PostDate'] == target_date) &
            (abs(df['Amount']) == fraud_amount) &
            (df['Fraud Adjustment Indicator'].isna()) &
            (~df.index.isin(used_indices))
            ]
        if 'Post Time' in df.columns:
            candidates = candidates.sort_values('Post Time')
    else:
        # 30-day window matching
        start_date = fraud_date - timedelta(days=30)
        candidates = df[
            (df['PostDate'] >= start_date) &
            (df['PostDate'] <= fraud_date) &
            (abs(df['Amount']) == fraud_amount) &
            (df['Fraud Adjustment Indicator'].isna()) &
            (~df.index.isin(used_indices))
            ]

        # Same day transactions must be before adjustment
        if not candidates.empty and 'Post Time' in df.columns:
            same_day = candidates[candidates['PostDate'] == fraud_date]
            if not same_day.empty:
                same_day = same_day[same_day['Post Time'] < fraud_row['Post Time']]
                other_days = candidates[candidates['PostDate'] < fraud_date]
                candidates = pd.concat([other_days, same_day])

    if candidates.empty:
        return -1

    # Sort by time and prefer mobile deposits
    sort_cols = ['PostDate', 'Post Time'] if 'Post Time' in df.columns else ['PostDate']
    candidates = candidates.sort_values(sort_cols)

    if 'Transaction Description' in df.columns:
        mobile_deposits = candidates[
            candidates['Transaction Description'].str.contains('Mobile Deposit', na=False)
        ]
        if not mobile_deposits.empty:
            return mobile_deposits.index[0]

    return candidates.index[0]


def process_single_member_fraud(member_data):
    """Process fraud matching for a single member"""
    df = member_data.copy()
    df['Post Date'] = pd.to_datetime(df['PostDate'])
    df['Is_Fraud'] = False

    # Sort by time
    sort_cols = ['PostDate', 'Post Time'] if 'Post Time' in df.columns else ['PostDate']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Find fraud adjustments
    fraud_adjustments = df[
        df['Fraud Adjustment Indicator'].notna() &
        (df['Fraud Adjustment Indicator'].str.strip() != '')
        ]

    if fraud_adjustments.empty:
        return df, 0, 0

    # Match adjustments to transactions
    used_indices = set()
    matched_count = 0
    adjustments_to_remove = []

    for _, fraud_adj in fraud_adjustments.iterrows():
        fraud_amount = abs(fraud_adj['Amount'])

        # Check if amount exists in OTHER transactions (excluding the adjustment itself)
        other_transactions = df[df.index != fraud_adj.name]
        amount_exists = (abs(other_transactions['Amount']) == fraud_amount).any()

        if not amount_exists:
            adjustments_to_remove.append(fraud_adj.name)
            print(f"  Removing adjustment with amount {fraud_amount} - no matching amount found")
            continue

        # Try to match
        match_idx = match_fraud_adjustment_to_transaction(fraud_adj, df, used_indices)

        if match_idx >= 0:
            df.loc[match_idx, 'Is_Fraud'] = True
            used_indices.add(match_idx)
            matched_count += 1

    # Remove invalid adjustments
    if adjustments_to_remove:
        df = df.drop(adjustments_to_remove)

    # Recalculate total adjustments
    remaining_adjustments = df[
        df['Fraud Adjustment Indicator'].notna() &
        (df['Fraud Adjustment Indicator'].str.strip() != '')
        ]
    total_adjustments = len(remaining_adjustments)

    return df, matched_count, total_adjustments


def normalize_output_columns(df, remove_adjustment_indicator=True):
    """Normalize columns for output"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Create Fraud column from Is_Fraud
    df['Fraud'] = df.get('Is_Fraud', 0).astype(int)

    # Add Fraud Description if missing
    if 'Fraud Description' not in df.columns:
        df['Fraud Description'] = ""

    # Fill descriptions for fraud transactions
    if (df['Fraud'] == 1).any():
        df.loc[df['Fraud'] == 1, 'Fraud Description'] = 'Fraud transaction identified'

    # Remove auxiliary columns
    cols_to_remove = ['Is_Fraud', 'Amount_missing']
    if remove_adjustment_indicator:
        cols_to_remove.append('Fraud Adjustment Indicator')

    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def process_fraud_data_complete(input_csv_path, output_dir='processed_data'):
    """Main fraud processing pipeline"""

    # Create output directories (no need for no_fraud directory)
    dirs = ['matched', 'unmatched', 'problematic']
    dir_paths = {d: os.path.join(output_dir, d) for d in dirs}
    for path in dir_paths.values():
        os.makedirs(path, exist_ok=True)

    # Load data
    print("Loading transaction data...")
    df = pd.read_csv(input_csv_path, low_memory=False)
    print(f"Loaded {len(df)} transactions")

    # Identify member categories
    fraud_victims = df[
        df['Fraud Adjustment Indicator'].notna() &
        (df['Fraud Adjustment Indicator'].str.strip() != '')
        ]['Member ID'].unique()

    all_members = df['Member ID'].unique()
    no_fraud_members = set(all_members) - set(fraud_victims)

    print(f"Found {len(fraud_victims)} fraud victims")
    print(f"Found {len(no_fraud_members)} members with no fraud")

    # Initialize counters (including no_fraud for statistics)
    counts = {'matched': 0, 'unmatched': 0, 'problematic': 0, 'no_fraud': 0}
    summary_results = []

    # Process fraud victims
    print("\nProcessing fraud victims...")
    for member_id in fraud_victims:
        try:
            member_data = df[df['Member ID'] == member_id]

            # Check for problematic data (more comprehensive check)
            adjustments = member_data[
                member_data['Fraud Adjustment Indicator'].notna() &
                (member_data['Fraud Adjustment Indicator'].str.strip() != '')
                ]

            has_problems = False
            for _, adj in adjustments.iterrows():
                fraud_amount = abs(adj['Amount'])
                # Check if ANY other transaction (excluding the adjustment itself) has this amount
                other_transactions = member_data[member_data.index != adj.name]
                amount_exists = (abs(other_transactions['Amount']) == fraud_amount).any()
                if not amount_exists:
                    has_problems = True
                    print(f"  Member {member_id}: No matching amount {fraud_amount} found for adjustment")
                    break

            # Process matching
            processed_data, matches_found, total_adjustments = process_single_member_fraud(member_data)

            # Categorize and save
            if has_problems:
                category = 'problematic'
                filename = f"member_{member_id}.csv"
                output_data = normalize_output_columns(processed_data)
            elif matches_found == total_adjustments and matches_found > 0:
                category = 'matched'
                filename = f"member_{member_id}.csv"
                # For matched cases, remove adjustment rows first, then normalize
                matched_data = processed_data[processed_data['Fraud Adjustment Indicator'].isna()].copy()
                output_data = normalize_output_columns(matched_data)
            else:
                category = 'unmatched'
                filename = f"member_{member_id}.csv"
                output_data = normalize_output_columns(processed_data, remove_adjustment_indicator=False)

            output_data.to_csv(os.path.join(dir_paths[category], filename), index=False)
            counts[category] += 1

            # Record summary
            summary_results.append({
                'Member_ID': member_id,
                'Category': category,
                'Fraud_Found': matches_found,
                'Total_Adjustments': total_adjustments,
                'Match_Rate': f"{matches_found}/{total_adjustments}",
                'All_Matched': matches_found == total_adjustments and matches_found > 0,
                'Has_Problems': has_problems
            })

            print(f"Member {member_id}: {matches_found}/{total_adjustments} -> {category}")

        except Exception as e:
            print(f"Error processing Member {member_id}: {e}")
            counts['unmatched'] += 1

    # Count no-fraud members (no file output needed)
    counts['no_fraud'] = len(no_fraud_members)
    print(f"\nCounted {counts['no_fraud']} no-fraud members (no files generated)")

    # Save summary
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(output_dir, 'processing_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # Print results
    print(f"\n{'=' * 50}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total members: {len(all_members)}")
    print(f"Matched: {counts['matched']}")
    print(f"Unmatched: {counts['unmatched']}")
    print(f"Problematic: {counts['problematic']}")
    print(f"No fraud: {counts['no_fraud']}")
    print(f"\nResults saved to '{output_dir}'")
    print(f"Summary: '{summary_path}'")

    return len(all_members)


if __name__ == "__main__":
    input_file = "TransactionData 10-9-25.csv"
    output_folder = "processed_data"
    process_fraud_data_complete(input_file, output_folder)