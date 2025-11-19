#!/usr/bin/env python3
"""
CSV Data Cleaning Script
Cleans and standardizes transaction CSV files with automatic field correction and renaming.
"""

import os
import csv
import re
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# Configuration
ENABLE_RENAMING = True
RAW_DIR = '../../data/train/raw'
CLEANED_DIR = '../../data/train/cleaned'

STANDARD_HEADERS = [
    'Account ID', 'Member ID', 'Account Type', 'Account Open Date',
    'Member Age', 'Product ID', 'Post Date', 'Post Time', 'Amount',
    'Action Type', 'Source Type', 'Transaction Description',
    'Fraud Adjustment Indicator'
]


def normalize_header(header_fields):
    """Standardize header field names."""
    normalized = []
    for field in header_fields:
        field_clean = field.strip().replace(' ', '')
        matched = False
        for std in STANDARD_HEADERS:
            if field_clean.lower() == std.replace(' ', '').lower():
                normalized.append(std)
                matched = True
                break
        if not matched:
            normalized.append(field.strip())
    return normalized


def get_unique_filename(output_dir, filename):
    """Generate unique filename if file already exists."""
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(output_path):
        return filename

    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{name}_v{counter}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        if not os.path.exists(new_path):
            return new_filename
        counter += 1


def clean_field(field_value):
    """Clean a single field: remove newlines, normalize whitespace."""
    if not field_value:
        return field_value

    # Remove all types of newlines
    cleaned = field_value.replace('\r\n', ' ')
    cleaned = cleaned.replace('\n', ' ')
    cleaned = cleaned.replace('\r', ' ')

    # Normalize multiple spaces to single space
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def process_csv_file(file_path, output_dir):
    """Process a single CSV file with cleaning and standardization."""
    # Read CSV file (csv.reader handles quoted newlines automatically)
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        return None, "EMPTY"

    # Normalize headers
    normalized_header = normalize_header(all_rows[0])

    # Get column indices
    try:
        idx_map = {h: normalized_header.index(h) for h in [
            'Amount', 'Product ID', 'Action Type', 'Source Type',
            'Transaction Description', 'Account Type', 'Post Date'
        ]}
    except ValueError:
        idx_map = {}

    cleaned_rows = [normalized_header]
    fixed_comma = fixed_missing = fixed_amount = fixed_newlines = 0
    first_date = last_date = None

    # Process data rows
    for row in all_rows[1:]:
        fields = list(row)

        # Remove empty trailing fields
        while len(fields) > 13 and not fields[-1].strip():
            fields.pop()

        # Merge extra fields (from Transaction Description commas)
        while len(fields) > 13:
            fields[-3] = fields[-3] + ' ' + fields[-2]
            fields.pop(-2)
            fixed_comma += 1

        # Pad if needed
        while len(fields) < 13:
            fields.append('')

        # CRITICAL: Clean ALL fields to remove newlines
        for i in range(len(fields)):
            original = fields[i]
            cleaned = clean_field(original)

            # Track if we fixed a newline
            if '\n' in original or '\r' in original:
                fixed_newlines += 1

            fields[i] = cleaned

        # Clean amount field (remove dollar signs and commas)
        if 'Amount' in idx_map:
            amount_idx = idx_map['Amount']
            if 0 <= amount_idx < len(fields):
                amount_val = fields[amount_idx]
                cleaned_amount = amount_val.replace('$', '').replace(',', '').strip()
                if cleaned_amount != amount_val:
                    fields[amount_idx] = cleaned_amount
                    fixed_amount += 1

        # Track date range
        if 'Post Date' in idx_map:
            try:
                date_str = fields[idx_map['Post Date']].strip()
                if date_str:
                    date_val = pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
                    if pd.notna(date_val):
                        if first_date is None or date_val < first_date:
                            first_date = date_val
                        if last_date is None or date_val > last_date:
                            last_date = date_val
            except:
                pass

        # Fill missing values
        fill_rules = [
            ('Amount', '0'), ('Product ID', '0'), ('Action Type', 'Unknown'),
            ('Source Type', 'Unknown'), ('Transaction Description', 'Unknown'),
            ('Account Type', 'Unknown')
        ]

        for col, fill_val in fill_rules:
            if col in idx_map:
                idx = idx_map[col]
                if 0 <= idx < len(fields):
                    val = fields[idx].strip()
                    if not val or val.lower() == 'null':
                        fields[idx] = fill_val
                        fixed_missing += 1

        cleaned_rows.append(fields)

    # Merge Transaction Description into Fraud Adjustment Indicator
    fraud_idx = None
    desc_idx = None

    try:
        fraud_idx = normalized_header.index('Fraud Adjustment Indicator')
        desc_idx = normalized_header.index('Transaction Description')
    except ValueError:
        pass

    merged_count = 0
    if fraud_idx is not None and desc_idx is not None:
        for row in cleaned_rows[1:]:
            if len(row) > max(fraud_idx, desc_idx):
                fraud_val = row[fraud_idx].strip()
                desc_val = row[desc_idx].strip()

                if fraud_val and desc_val and desc_val.lower() != 'unknown':
                    row[fraud_idx] = f"{desc_val} {fraud_val}"
                    row[desc_idx] = 'Unknown'
                    merged_count += 1

    # Determine output filename
    output_filename = os.path.basename(file_path)
    if ENABLE_RENAMING and first_date and last_date:
        first_str = first_date.strftime('%m-%d-%Y')
        last_str = last_date.strftime('%m-%d-%Y')
        output_filename = f"{first_str}_to_{last_str}.csv"

    # Get unique filename if needed
    output_filename = get_unique_filename(output_dir, output_filename)

    # Save cleaned file
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_rows)

    # Compile summary
    stats = {
        'fixed_comma': fixed_comma,
        'fixed_amount': fixed_amount,
        'fixed_missing': fixed_missing,
        'fixed_newlines': fixed_newlines,
        'merged_fraud': merged_count,
        'renamed': output_filename != os.path.basename(file_path),
        'new_name': output_filename if ENABLE_RENAMING else None
    }

    return stats, None


def main():
    """Main execution function."""
    global ENABLE_RENAMING, RAW_DIR, CLEANED_DIR

    # Create cleaned directory
    os.makedirs(CLEANED_DIR, exist_ok=True)

    print(f"Raw: {RAW_DIR}")
    print(f"Cleaned: {CLEANED_DIR}\n")

    # Check if directory exists
    if not os.path.exists(RAW_DIR):
        print(f"Error: Directory {RAW_DIR} does not exist")
        return

    # Get all CSV files
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {RAW_DIR}")
    print("\nCSV Files List ⬇️")
    for i, filename in enumerate(csv_files, 1):
        file_size = os.path.getsize(os.path.join(RAW_DIR, filename)) / (1024 * 1024)
        print(f"  {i}. {filename} ({file_size:.2f} MB)")

    print("\n" + "=" * 60)
    print("Processing Files...")
    print("=" * 60 + "\n")

    # Process each file
    for i, filename in enumerate(csv_files, 1):
        file_path = os.path.join(RAW_DIR, filename)
        print(f"[{i}/{len(csv_files)}] {filename}...", end=' ')

        stats, error = process_csv_file(file_path, CLEANED_DIR)

        if error:
            print(error)
            continue

        # Output summary
        msg = []
        if stats['fixed_comma'] > 0:
            msg.append(f"Fields:{stats['fixed_comma']}")
        if stats['fixed_amount'] > 0:
            msg.append(f"Amount:{stats['fixed_amount']}")
        if stats['fixed_missing'] > 0:
            msg.append(f"Missing:{stats['fixed_missing']}")
        if stats['fixed_newlines'] > 0:
            msg.append(f"Newlines:{stats['fixed_newlines']}")
        if stats['merged_fraud'] > 0:
            msg.append(f"Fraud_Merged:{stats['merged_fraud']}")
        if stats['renamed']:
            msg.append(f"→{stats['new_name']}")

        print(', '.join(msg) if msg else "OK")

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()