#!/usr/bin/env python3
"""
Stage 4 Inference: Feature Encoding
Encode categorical features in member CSV files for inference.
"""

import pandas as pd
import json
import glob
import os

# Configuration
PROCESSED_DIR = '../../../data/pred/processed'
OUTPUT_DIR = '../../../data/pred/final'
CONFIG_PATH = '../../../config/tokenize_dict.json'
CAT_FEATURES = ["Account Type", "Action Type", "Source Type", "Product ID"]


def parse_post_time(x):
    """Convert post time string to time object (e.g., '105856' -> 10:58:56)"""
    if pd.isna(x) or x == '':
        return None

    try:
        s = str(int(float(x))).zfill(6)
        hh, mm, ss = int(s[0:2]), int(s[2:4]), int(s[4:6])
        return pd.to_datetime(f"{hh}:{mm}:{ss}", format="%H:%M:%S").time()
    except:
        return None


def encode_features(processed_dir=None, output_dir=None, config_path=None):
    """Encode categorical features in processed member files

    Args:
        processed_dir: Directory containing member CSV files
        output_dir: Directory to save encoded files
        config_path: Path to tokenize_dict.json

    Returns:
        Number of files processed
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if config_path is None:
        config_path = CONFIG_PATH

    print("=" * 60)
    print("STAGE 4: FEATURE ENCODING (INFERENCE)")
    print("=" * 60)
    print(f"Input Dir: {processed_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Config Path: {config_path}\n")

    # Load encoding dictionary
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return 0

    with open(config_path, "r") as f:
        uni_vals = json.load(f)
    print(f"Loaded encoding dictionary with {len(uni_vals)} features\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all member CSV files directly in processed_dir
    csv_files = glob.glob(os.path.join(processed_dir, "member_*.csv"))

    if not csv_files:
        print(f"No member files found in {processed_dir}")
        return 0

    print(f"Found {len(csv_files)} member file(s)")
    total_processed = 0

    for i, file in enumerate(csv_files, 1):
        if i % 100 == 0 or i == len(csv_files):
            print(f"  Processed {i}/{len(csv_files)} files")

        try:
            df = pd.read_csv(file, dtype={'Product ID': str})

            # Delete ID columns if they exist
            for col in ["Account ID", "Member ID"]:
                if col in df.columns:
                    del df[col]

            # Encode categorical features
            for col in CAT_FEATURES:
                if col not in df.columns:
                    continue

                # Handle NaN values by converting to string
                df[col] = df[col].fillna("NaN").astype(str)
                df[col + '_enc'] = df[col].map(uni_vals[col])
                del df[col]

            # Parse time features
            if "Post Time" in df.columns:
                df["Post Time Parsed"] = df["Post Time"].apply(parse_post_time)
                del df["Post Time"]

            # Delete text columns if they exist
            for col in ["Transaction Description", "Fraud Adjustment Indicator"]:
                if col in df.columns:
                    del df[col]

            # Fill missing encoded values with default values
            df.fillna({
                "Action Type_enc": 3,
                "Source Type_enc": 2,
                "Product ID_enc": 33,
                "Account Type_enc": 0
            }, inplace=True)

            # Save processed file to output directory
            output_file = os.path.join(output_dir, os.path.basename(file))
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            total_processed += 1

        except Exception as e:
            print(f"  Error processing {os.path.basename(file)}: {e}")
            continue

    print()
    print("=" * 60)
    print("STAGE 4 COMPLETE (INFERENCE)")
    print("=" * 60)
    print(f"Total files processed: {total_processed}/{len(csv_files)}")
    print(f"Output location: {output_dir}")

    return total_processed


def main():
    """Main execution function"""
    encode_features(PROCESSED_DIR, OUTPUT_DIR, CONFIG_PATH)


if __name__ == "__main__":
    main()
