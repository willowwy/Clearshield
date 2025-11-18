import pandas as pd
import os
import re
from glob import glob
from datetime import timedelta

# Configuration
INPUT_DIR = '../../../data/clustered_out'
OUTPUT_MEMBER_DIR = '../../../data/by_member'
OUTPUT_PROCESSED_DIR = '../../../data/processed'
CHUNKSIZE = 50000
MIN_HISTORY_LENGTH = 10  # Minimum number of transactions required


def reorganize_by_member(input_dir, output_dir, chunksize):
    """Reorganize transaction files by member"""
    os.makedirs(output_dir, exist_ok=True)
    all_files = sorted(glob(os.path.join(input_dir, '*.csv')))

    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return 0

    print(f"Found {len(all_files)} files")
    member_files_created = set()

    for i, file_path in enumerate(all_files, 1):
        print(f"Processing {i}/{len(all_files)}: {os.path.basename(file_path)}")
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
                grouped = chunk.groupby('Member ID')

                for member_id, member_data in grouped:
                    member_file = os.path.join(output_dir, f"member_{member_id}.csv")
                    file_exists = os.path.exists(member_file)

                    if member_file in member_files_created or file_exists:
                        member_data.to_csv(member_file, mode='a', header=False, index=False)
                    else:
                        member_data.to_csv(member_file, mode='w', header=True, index=False)
                        member_files_created.add(member_file)
        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"Created {len(member_files_created)} member files")
    print("Sorting files...")

    for i, member_file in enumerate(list(member_files_created), 1):
        if i % 1000 == 0:
            print(f"  Sorted {i}/{len(member_files_created)} files")
        try:
            df = pd.read_csv(member_file, low_memory=False)
            df['Post Date'] = pd.to_datetime(df['Post Date'])
            sort_cols = ['Post Date', 'Post Time'] if 'Post Time' in df.columns else ['Post Date']
            df = df.sort_values(sort_cols)
            df.to_csv(member_file, index=False)
        except Exception as e:
            print(f"Error sorting {os.path.basename(member_file)}: {e}")
            continue

    return len(member_files_created)


def extract_fraud_date_from_description(description, adjustment_date):
    """Extract date from fraud adjustment description

    Supports formats (in priority order):
    1. M/D/YYYY or MM/DD/YYYY (e.g., 2/3/2025, 12/31/2024)
    2. M/D or MM/DD (e.g., 2/3, 12/31) - uses adjustment year
    3. M.D.YYYY or MM.DD.YYYY (e.g., 2.3.2025, 12.31.2024)
    4. M.D or MM.DD (e.g., 2.3, 12.31) - uses adjustment year
    """
    if pd.isna(description):
        return None

    description_str = str(description)

    # Priority 1: M/D/YYYY or MM/DD/YYYY format
    match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', description_str)
    if match:
        try:
            month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return pd.Timestamp(year, month, day)
        except:
            pass

    # Priority 2: M/D or MM/DD format (no year, use adjustment year)
    match = re.search(r'(\d{1,2})/(\d{1,2})(?!/)', description_str)
    if match:
        try:
            month, day = int(match.group(1)), int(match.group(2))
            return pd.Timestamp(adjustment_date.year, month, day)
        except:
            pass

    # Priority 3: M.D.YYYY or MM.DD.YYYY format
    match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', description_str)
    if match:
        try:
            month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return pd.Timestamp(year, month, day)
        except:
            pass

    # Priority 4: M.D or MM.DD format (no year, use adjustment year)
    match = re.search(r'(\d{1,2})\.(\d{1,2})(?!\.)', description_str)
    if match:
        try:
            month, day = int(match.group(1)), int(match.group(2))
            return pd.Timestamp(adjustment_date.year, month, day)
        except:
            pass

    return None


def match_fraud_adjustment_to_transaction(fraud_row, df, used_indices):
    """Find matching original transaction for fraud adjustment"""
    fraud_amount = abs(fraud_row['Amount'])
    fraud_date = fraud_row['Post Date']
    target_date = extract_fraud_date_from_description(fraud_row['Fraud Adjustment Indicator'], fraud_date)

    # Exclude: already used transactions + all refund records
    fraud_col = df['Fraud Adjustment Indicator'].astype(str)
    exclude_mask = (
            df.index.isin(used_indices) |
            ((fraud_col != 'nan') & (fraud_col.str.strip() != ''))
    )

    # STEP 1: Try exact date match if date extracted
    if target_date:
        candidates = df[
            (df['Post Date'] == target_date) &
            (abs(df['Amount']) == fraud_amount) &
            (~exclude_mask)
            ]

        if not candidates.empty:
            if 'Post Time' in df.columns:
                candidates = candidates.sort_values('Post Time')

            # Prioritize Mobile Deposit
            if 'Transaction Description' in df.columns:
                mobile_deposits = candidates[
                    candidates['Transaction Description'].str.contains('Mobile Deposit', na=False)]
                if not mobile_deposits.empty:
                    return mobile_deposits.index[0]

            return candidates.index[0]

    # STEP 2: Exact date match failed, use 30-day range
    start_date = fraud_date - timedelta(days=30)
    candidates = df[
        (df['Post Date'] >= start_date) &
        (df['Post Date'] <= fraud_date) &
        (abs(df['Amount']) == fraud_amount) &
        (~exclude_mask)
        ]

    if candidates.empty:
        return -1

    # Prioritize same-day transactions with earlier time
    if 'Post Time' in df.columns:
        same_day = candidates[candidates['Post Date'] == fraud_date]
        if not same_day.empty:
            same_day = same_day[same_day['Post Time'] < fraud_row['Post Time']]
            other_days = candidates[candidates['Post Date'] < fraud_date]
            candidates = pd.concat([other_days, same_day])

            # Check if filtering removed all candidates
            if candidates.empty:
                return -1

        candidates = candidates.sort_values(['Post Date', 'Post Time'])
    else:
        candidates = candidates.sort_values('Post Date')

    # Final check before accessing index
    if candidates.empty:
        return -1

    # Prioritize Mobile Deposit
    if 'Transaction Description' in df.columns:
        mobile_deposits = candidates[candidates['Transaction Description'].str.contains('Mobile Deposit', na=False)]
        if not mobile_deposits.empty:
            return mobile_deposits.index[0]

    return candidates.index[0]


def process_single_member_fraud(member_data):
    """Process fraud matching for a single member - mark Fraud column and remove matched adjustment records"""
    df = member_data.copy()
    df['Post Date'] = pd.to_datetime(df['Post Date'])
    df['Fraud'] = 0

    sort_cols = ['Post Date', 'Post Time'] if 'Post Time' in df.columns else ['Post Date']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    fraud_col = df['Fraud Adjustment Indicator'].astype(str)
    fraud_adjustments = df[(fraud_col != 'nan') & (fraud_col.str.strip() != '')]

    if fraud_adjustments.empty:
        return df, 0, 0

    used_indices = set()
    matched_count = 0
    adjustment_indices_to_remove = []  # Track adjustment records to delete

    for idx, fraud_adj in fraud_adjustments.iterrows():
        match_idx = match_fraud_adjustment_to_transaction(fraud_adj, df, used_indices)

        if match_idx >= 0:
            # Mark the original transaction as fraud
            df.loc[match_idx, 'Fraud'] = 1
            used_indices.add(match_idx)
            matched_count += 1

            # Mark the adjustment record for deletion
            adjustment_indices_to_remove.append(idx)

    total_adjustments = len(fraud_adjustments)

    # Remove successfully matched adjustment records
    if adjustment_indices_to_remove:
        df = df.drop(adjustment_indices_to_remove).reset_index(drop=True)

    return df, matched_count, total_adjustments


def process_member_files_for_fraud(member_dir, output_dir, min_history_length=None):
    """Process member files for fraud detection - keep all records intact

    Args:
        member_dir: Directory containing member files
        output_dir: Output directory for processed files
        min_history_length: Minimum transaction count required (None = use global MIN_HISTORY_LENGTH)
    """
    if min_history_length is None:
        min_history_length = MIN_HISTORY_LENGTH

    for subdir in ['matched', 'unmatched', 'no_fraud']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    member_files = sorted(glob(os.path.join(member_dir, 'member_*.csv')))
    print(f"Found {len(member_files)} member files")

    if min_history_length > 0:
        print(f"Filtering: only processing members with >= {min_history_length} transactions")

    stats = {'matched': 0, 'unmatched': 0, 'no_fraud': 0, 'total': 0, 'skipped': 0}
    summary_records = []

    for i, member_file in enumerate(member_files, 1):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(member_files)} members")
        try:
            df = pd.read_csv(member_file, low_memory=False)
            member_id = os.path.basename(member_file).replace('member_', '').replace('.csv', '')

            # Filter by minimum history length
            if len(df) < min_history_length:
                stats['skipped'] += 1
                continue

            stats['total'] += 1

            fraud_col = df['Fraud Adjustment Indicator'].astype(str)
            has_fraud = (fraud_col != 'nan') & (fraud_col.str.strip() != '')
            fraud_count = has_fraud.sum()

            if fraud_count > 0:
                # Process fraud matching - only marks Fraud column, keeps all records
                processed_data, matches_found, total_adjustments = process_single_member_fraud(df)

                if matches_found == total_adjustments and matches_found > 0:
                    category = 'matched'
                    stats['matched'] += 1
                else:
                    category = 'unmatched'
                    stats['unmatched'] += 1

                output_data = processed_data

                summary_records.append({
                    'Member_ID': member_id,
                    'Total_Transactions': len(df),
                    'Fraud_Adjustments': total_adjustments,
                    'Matched': matches_found,
                    'Category': category
                })
            else:
                category = 'no_fraud'
                stats['no_fraud'] += 1
                df['Fraud'] = 0
                output_data = df

                summary_records.append({
                    'Member_ID': member_id,
                    'Total_Transactions': len(df),
                    'Fraud_Adjustments': 0,
                    'Matched': 0,
                    'Category': category
                })

            output_path = os.path.join(output_dir, category, f"member_{member_id}.csv")
            output_data.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error processing {os.path.basename(member_file)}: {e}")
            continue

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, 'member_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    return stats


def run_stage1():
    """Run Stage 1: Data Reorganization"""
    print("=" * 60)
    print("STAGE 1: DATA REORGANIZATION")
    print("=" * 60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_MEMBER_DIR}\n")

    num_members = reorganize_by_member(INPUT_DIR, OUTPUT_MEMBER_DIR, CHUNKSIZE)
    print(f"\n{num_members} member files created\n")

    return num_members


def run_stage2(min_history_length=None):
    """Run Stage 2: Fraud Detection

    Args:
        min_history_length: Minimum transaction count required (None = use global MIN_HISTORY_LENGTH)
    """
    if min_history_length is None:
        min_history_length = MIN_HISTORY_LENGTH

    print("=" * 60)
    print("STAGE 2: FRAUD DETECTION")
    print("=" * 60)
    print(f"Input: {OUTPUT_MEMBER_DIR}")
    print(f"Output: {OUTPUT_PROCESSED_DIR}")
    print(f"Min History Length: {min_history_length}\n")

    stats = process_member_files_for_fraud(OUTPUT_MEMBER_DIR, OUTPUT_PROCESSED_DIR, min_history_length)

    print(f"\nProcessing Summary:")
    print(f"  Total Processed: {stats['total']}")
    print(f"  Skipped (< {min_history_length} txns): {stats['skipped']}")
    print(f"  No Fraud: {stats['no_fraud']}")
    print(f"  Matched: {stats['matched']}")
    print(f"  Unmatched: {stats['unmatched']}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    return stats


def main():
    """Main execution function"""
    print("=" * 60)
    print("STAGE 1: DATA REORGANIZATION")
    print("=" * 60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_MEMBER_DIR}\n")

    num_members = reorganize_by_member(INPUT_DIR, OUTPUT_MEMBER_DIR, CHUNKSIZE)
    print(f"\n{num_members} member files created\n")

    print("=" * 60)
    print("STAGE 2: FRAUD DETECTION")
    print("=" * 60)
    print(f"Input: {OUTPUT_MEMBER_DIR}")
    print(f"Output: {OUTPUT_PROCESSED_DIR}\n")

    stats = process_member_files_for_fraud(OUTPUT_MEMBER_DIR, OUTPUT_PROCESSED_DIR)

    print(f"\nProcessing Summary:")
    print(f"  Total: {stats['total']}")
    print(f"  No Fraud: {stats['no_fraud']}")
    print(f"  Matched: {stats['matched']}")
    print(f"  Unmatched: {stats['unmatched']}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()