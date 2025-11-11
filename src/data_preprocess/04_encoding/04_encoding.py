import pandas as pd
import json
import glob
import os

# Configuration
PROCESSED_DIR = '../../../data/processed'
OUTPUT_DIR = '../../../data/final'
CONFIG_PATH = '../../../config/tokenize_dict.json' #json cnofig
CAT_FEATURES = ["Account Type", "Action Type", "Source Type", "Product ID"]


def parse_post_time(time_str):
    """Convert post time string to decimal hours (e.g., '105856' -> 10.98)"""
    if pd.isna(time_str) or time_str == '':
        return 0.0

    try:
        time_str = str(int(float(time_str))).zfill(6)  # Ensure 6 digits
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        return hour + minute / 60.0 + second / 3600.0
    except:
        return 0.0


def encode_features(processed_dir=None, output_dir=None, config_path=None):
    """Encode categorical features in processed member files

    Args:
        processed_dir: Directory containing matched/unmatched/no_fraud folders
        output_dir: Directory to save encoded files (default: data/final)
        config_path: Path to tokenize_dict.json
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if config_path is None:
        config_path = CONFIG_PATH

    print("=" * 60)
    print("FEATURE ENCODING")
    print("=" * 60)
    print(f"Input Dir: {processed_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Config Path: {config_path}\n")

    # Load encoding dictionary
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    uni_vals = json.load(open(config_path, "r"))
    print(f"Loaded encoding dictionary with {len(uni_vals)} features")

    # Process each subfolder
    subfolders = ['matched', 'unmatched', 'no_fraud']
    total_files = 0
    total_processed = 0

    for subfolder in subfolders:
        subfolder_path = os.path.join(processed_dir, subfolder)
        output_subfolder_path = os.path.join(output_dir, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"\nWarning: {subfolder_path} does not exist, skipping...")
            continue

        # Create output directory if it doesn't exist
        os.makedirs(output_subfolder_path, exist_ok=True)

        csv_files = glob.glob(os.path.join(subfolder_path, "member_*.csv"))

        if not csv_files:
            print(f"\n{subfolder}: No member files found")
            continue

        print(f"\n{subfolder}: Found {len(csv_files)} files")
        total_files += len(csv_files)

        for i, file in enumerate(csv_files, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(csv_files)} files")

            try:
                df = pd.read_csv(file)

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

                # Convert date columns
                if "Post Date" in df.columns:
                    df["Post Date"] = pd.to_datetime(df["Post Date"])
                if "Account Open Date" in df.columns:
                    df["Account Open Date"] = pd.to_datetime(df["Account Open Date"])

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
                output_file = os.path.join(output_subfolder_path, os.path.basename(file))
                df.to_csv(output_file, index=False, encoding="utf-8-sig")
                total_processed += 1

            except Exception as e:
                print(f"  Error processing {os.path.basename(file)}: {e}")
                continue

        print(f"  {subfolder}: Encoded {i}/{len(csv_files)} files")

    print("\n" + "=" * 60)
    print(f"Encoding Complete!")
    print(f"Total files found: {total_files}")
    print(f"Total files processed: {total_processed}")
    print("=" * 60)

    return total_processed


def main():
    """Main execution function"""
    encode_features(PROCESSED_DIR, OUTPUT_DIR, CONFIG_PATH)


if __name__ == "__main__":
    main()
