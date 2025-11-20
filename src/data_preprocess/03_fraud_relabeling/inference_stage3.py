#!/usr/bin/env python3
"""
Stage 3-1: Fraud relabeling (inference)

Goal:
  - Read clustered transaction CSV files (already cleaned in Stage 1).
  - Add a 'Fraud' column filled with 0 (prediction mode).
  - Drop 'Fraud Adjustment Indicator'.
  - Reorganize data by member: one CSV per Member ID.
  - Sort each member file by Post Date and Post Time.

Note: Date and Product ID formatting is done in Stage 1 (data cleaning).
      This stage only maintains the cleaned format.
"""

import os
from glob import glob

import pandas as pd

# ===== Global configuration (will be overridden by driver script) =====
INPUT_DIR = '../../../data/pred/clustered_out'
OUTPUT_MEMBER_DIR = '../../../data/pred/by_member'
CHUNKSIZE = 50000


def reorganize_with_fraud_by_member(
    input_dir: str,
    output_dir: str,
    chunksize: int = 50000,
    verbose: bool = True
) -> int:
    """
    Reorganize transaction files by member, add Fraud column, and sort by date/time.

    Steps:
      1) Scan all CSV files in input_dir (already cleaned in Stage 1).
      2) Read each file in chunks:
         - Drop 'Fraud Adjustment Indicator' if present.
         - Add 'Fraud' column filled with 0.
         - Group by 'Member ID' and append to member_{member_id}.csv.
      3) Sort each member file by 'Post Date' and 'Post Time' (if available).

    Returns:
        Number of member files created.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files in input_dir
    all_files = sorted(glob(os.path.join(input_dir, '*.csv')))

    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return 0

    if verbose:
        print("=" * 60)
        print("STAGE 3-1: FRAUD RELABELING & REORGANIZE BY MEMBER")
        print("=" * 60)
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(all_files)} CSV file(s)")
        print()

    member_files_modified = set()  # Track all modified files (created + appended)

    # ---------- First pass: read clustered files and write per-member files ----------
    for i, file_path in enumerate(all_files, 1):
        file_name = os.path.basename(file_path)
        if verbose:
            print(f"[{i}/{len(all_files)}] Processing {file_name}")

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False, dtype={'Product ID': str}):
                # Ensure Member ID exists
                if 'Member ID' not in chunk.columns:
                    # If no Member ID, skip this chunk
                    continue

                # Drop Fraud Adjustment Indicator if present
                if 'Fraud Adjustment Indicator' in chunk.columns:
                    chunk = chunk.drop(columns=['Fraud Adjustment Indicator'])

                # Add Fraud column (all 0 for inference)
                chunk['Fraud'] = 0

                # Group by Member ID
                grouped = chunk.groupby('Member ID')

                for member_id, member_data in grouped:
                    member_file = os.path.join(output_dir, f"member_{member_id}.csv")
                    file_exists = os.path.exists(member_file)

                    # Append without header if file already exists
                    if member_file in member_files_modified or file_exists:
                        member_data.to_csv(member_file, mode='a', header=False, index=False)
                    else:
                        member_data.to_csv(member_file, mode='w', header=True, index=False)

                    # Track this file as modified (whether created or appended)
                    member_files_modified.add(member_file)

        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            continue

    if verbose:
        print()
        print(f"Modified {len(member_files_modified)} member file(s) this run")
        print("Sorting modified member files by Post Date / Post Time...")

    # ---------- Second pass: sort ONLY modified member files ----------
    member_files_to_sort = sorted(list(member_files_modified))

    for i, member_file in enumerate(member_files_to_sort, 1):
        try:
            df = pd.read_csv(member_file, low_memory=False, dtype={'Product ID': str})

            if 'Post Date' in df.columns:
                # Parse Post Date as datetime for sorting
                df['Post Date'] = pd.to_datetime(df['Post Date'], errors='coerce')

                # Build sort columns
                sort_cols = ['Post Date']
                if 'Post Time' in df.columns:
                    # Convert Post Time to numeric for proper sorting
                    df['Post Time'] = pd.to_numeric(df['Post Time'], errors='coerce')
                    sort_cols.append('Post Time')

                df = df.sort_values(sort_cols)

                # Convert back to MM/DD/YYYY to maintain cleaned format
                df['Post Date'] = df['Post Date'].dt.strftime('%m/%d/%Y')

            # Save back to file
            df.to_csv(member_file, index=False)

            if verbose and (i % 1000 == 0 or i == len(member_files_to_sort)):
                print(f"  Sorted {i}/{len(member_files_to_sort)} member file(s)")

        except Exception as e:
            print(f"  Error sorting {os.path.basename(member_file)}: {e}")
            continue

    if verbose:
        print()
        print("=" * 60)
        print("STAGE 3-1 COMPLETE")
        print("=" * 60)
        print(f"Member files modified & sorted: {len(member_files_modified)}")
        print(f"Output location: {output_dir}")

    return len(member_files_modified)


def main(verbose: bool = True) -> int:
    """
    Entry point for notebook / pipeline.

    Uses global:
      - INPUT_DIR
      - OUTPUT_MEMBER_DIR
      - CHUNKSIZE

    Returns:
        Number of member files created.
    """
    return reorganize_with_fraud_by_member(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_MEMBER_DIR,
        chunksize=CHUNKSIZE,
        verbose=verbose
    )


if __name__ == "__main__":
    # When run directly as a script, use default globals.
    num_members = main(verbose=True)
    print(f"Total member files: {num_members}")
