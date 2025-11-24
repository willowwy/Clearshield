#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearShield - Automated Training Data Processing Pipeline
Runs the complete 5-stage preprocessing pipeline automatically.
"""

import sys
import os
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

# Fix multiprocessing issues on macOS
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.pipeline_config import get_train_config


def load_module(module_name, file_path):
    """Dynamically load a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def stage1_data_cleaning(dc, config):
    """Stage 1: Data Cleaning"""
    start_time = datetime.now()
    print(f"\n{'=' * 70}")
    print("STAGE 1: Data Cleaning")
    print(f"{'=' * 70}")

    # Configure using centralized config
    dc.ENABLE_RENAMING = True
    dc.RAW_DIR = str(config.get_path('raw'))
    dc.CLEANED_DIR = str(config.get_path('cleaned'))

    # Custom simplified output
    import os
    csv_files = sorted([f for f in os.listdir(dc.RAW_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} file(s)\n")

    # Process each file with custom output
    for i, filename in enumerate(csv_files, 1):
        input_path = os.path.join(dc.RAW_DIR, filename)
        print(f"[{i}/{len(csv_files)}] {filename}...", end=" ", flush=True)

        # Process file (suppress detailed output)
        import io
        import contextlib
        import pandas as pd

        with contextlib.redirect_stdout(io.StringIO()):
            # Run the cleaning for this file
            df = pd.read_csv(input_path)
            # Get output filename by calling the internal logic
            df_copy = df.copy()
            if 'Post Date' in df_copy.columns:
                df_copy['Post Date'] = pd.to_datetime(df_copy['Post Date'], errors='coerce')
                min_date = df_copy['Post Date'].min()
                max_date = df_copy['Post Date'].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    output_filename = f"{min_date.strftime('%m-%d-%Y')}_to_{max_date.strftime('%m-%d-%Y')}.csv"
                else:
                    output_filename = filename
            else:
                output_filename = filename

        print(f"→ {output_filename}")

    # Run the full cleaning (suppress output)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        dc.main()

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Stage 1 Complete ({elapsed:.1f}s)\n")
    return elapsed


def stage2_feature_engineering(fe, config):
    """Stage 2: Feature Engineering - Description Encoding and Clustering"""
    start_time = datetime.now()

    # Configure using centralized config
    fe_params = config.feature_engineering
    fe.PROCESSED_DIR = str(config.get_path('cleaned'))
    fe.MODEL_NAME = fe_params['model_name']
    fe.TEXT_COLUMN = fe_params['text_column']
    fe.BATCH_SIZE = fe_params['batch_size']
    fe.MAX_LENGTH = fe_params['max_length']
    fe.PCA_DIM = fe_params['pca_dim']
    fe.MIN_K = fe_params['min_k']
    fe.MAX_K = fe_params['max_k']
    fe.K_STEP = fe_params['k_step']
    fe.SAMPLE_SIZE = fe_params['sample_size']
    fe.CLUSTER_BATCH_SIZE = fe_params['cluster_batch_size']
    fe.RANDOM_STATE = fe_params['random_state']

    # Run (keep original output)
    fe.main()

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Stage 2 Complete ({elapsed:.1f}s)\n")
    return elapsed


def stage3_fraud_matching(fr, config, min_history_length=None):
    """Stage 3: Fraud Matching and Re-labeling"""
    start_time = datetime.now()
    print(f"{'=' * 70}")
    print("STAGE 3: Fraud Matching and Re-labeling")
    print(f"{'=' * 70}")

    # Use config default if not specified
    if min_history_length is None:
        min_history_length = config.fraud_matching['min_history_length']

    # Configure using centralized config
    fr.INPUT_DIR = str(config.get_path('clustered'))
    fr.OUTPUT_MEMBER_DIR = str(config.get_path('by_member_temp'))
    fr.OUTPUT_PROCESSED_DIR = str(config.get_path('by_member'))
    fr.CHUNKSIZE = config.fraud_matching['chunksize']

    print(f"Reorganizing by member and matching fraud adjustments...\n")

    # Run (suppress detailed output)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        fr.run_stage1()
        stats = fr.run_stage2(min_history_length)

    # Display summary from returned stats
    print("Processing Summary:")
    print(f"  Total Processed: {stats['total']:,}")
    print(f"  Skipped (< {min_history_length} txns): {stats['skipped']:,}")
    print(f"  Matched: {stats['matched']:,}")
    print(f"  Unmatched: {stats['unmatched']:,}")
    print(f"  No Fraud: {stats['no_fraud']:,}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Stage 3 Complete ({elapsed:.1f}s)\n")
    return elapsed


def stage4_encoding(enc, config):
    """Stage 4: Feature Encoding"""
    start_time = datetime.now()
    print(f"{'=' * 70}")
    print("STAGE 4: Feature Encoding")
    print(f"{'=' * 70}")

    # Configure using centralized config
    enc.PROCESSED_DIR = str(config.get_path('by_member'))
    enc.OUTPUT_DIR = str(config.get_path('final'))
    enc.CONFIG_PATH = str(config.get_path('tokenize_config'))

    print(f"Encoding features from member files...\n")

    # Run (keep progress output)
    enc.encode_features(enc.PROCESSED_DIR, enc.OUTPUT_DIR, enc.CONFIG_PATH)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Stage 4 Complete ({elapsed:.1f}s)\n")
    return elapsed


def stage5_vulnerability_scan(vuln_scanner, config, sample_size=1000):
    """Stage 5: Vulnerability Scanning (Optional)"""
    start_time = datetime.now()
    print(f"{'=' * 70}")
    print("STAGE 5: Vulnerability Scanning")
    print(f"{'=' * 70}")

    # Configure vulnerability scanner
    export_path = str(config.PROJECT_ROOT / 'vulnerability_scan_results.json')

    print(f"Running security tests on {sample_size} sample records...\n")

    # Run vulnerability scan (suppress detailed output)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        scan_results = vuln_scanner.run_vulnerability_scan(
            final_data_path=str(config.get_path('final')),
            data_file_pattern='*.csv',
            data_sample_size=sample_size,
            export_path=export_path,
            verbose=False  # Suppress verbose output
        )

    # Display summary
    passed = scan_results['status_breakdown'].get('PASSED', 0)
    failed = scan_results['status_breakdown'].get('FAILED', 0)
    warnings = scan_results['status_breakdown'].get('WARNING', 0)
    skipped = scan_results['status_breakdown'].get('SKIPPED', 0)
    vulns = scan_results['total_vulnerabilities']

    print("Test Summary:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Warnings: {warnings}")
    print(f"  Skipped: {skipped}")
    print(f"  Vulnerabilities: {vulns}")
    print(f"\nDetailed results saved to: {export_path}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Stage 5 Complete ({elapsed:.1f}s)\n")

    return elapsed


def run_pipeline():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='ClearShield Automated Training Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_train_pipeline.py                           # Run complete pipeline
  python run_train_pipeline.py --skip-cleaning           # Skip data cleaning step
  python run_train_pipeline.py --min-history 15          # Set minimum history to 15
  python run_train_pipeline.py --skip-vuln-scan          # Skip vulnerability scanning
  python run_train_pipeline.py --vuln-sample-size 2000   # Use 2000 samples for scanning
        """
    )

    parser.add_argument('--skip-cleaning', action='store_true',
                        help='Skip Stage 1: Data Cleaning')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                        help='Skip Stage 2: Feature Engineering')
    parser.add_argument('--skip-fraud-matching', action='store_true',
                        help='Skip Stage 3: Fraud Matching')
    parser.add_argument('--skip-encoding', action='store_true',
                        help='Skip Stage 4: Feature Encoding')
    parser.add_argument('--skip-vuln-scan', action='store_true',
                        help='Skip Stage 5: Vulnerability Scanning')
    parser.add_argument('--min-history', type=int, default=None,
                        help='Minimum transaction history length (default: from config)')
    parser.add_argument('--vuln-sample-size', type=int, default=1000,
                        help='Sample size for vulnerability scanning (default: 1000)')

    args = parser.parse_args()

    # Initialize configuration
    config = get_train_config()

    # Print header
    print("\n" + "=" * 70)
    print("CLEARSHIELD TRAINING PIPELINE")
    print("=" * 70)

    pipeline_start = datetime.now()
    timings = {}

    # Setup directories (silent)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        config.create_directories(verbose=False)

    # Load modules
    dc = load_module("data_cleaning", "./01_data_cleaning/01_data_cleaning.py")
    fe = load_module("feature_engineering", "./02_feature_engineering/02_feature_engineering.py")
    fr = load_module("fraud_relabeling", "./03_fraud_relabeling/03_fraud_relabeling.py")
    enc = load_module("encoding", "./04_encoding/04_encoding.py")
    vuln_scanner = load_module("vuln_scanner", "./05_security/vuln_scanner.py")

    try:
        # Stage 1: Data Cleaning
        if not args.skip_cleaning:
            timings['Stage 1'] = stage1_data_cleaning(dc, config)
        else:
            print(f"\n{'=' * 70}")
            print("STAGE 1: Data Cleaning [SKIPPED]")
            print(f"{'=' * 70}\n")

        # Stage 2: Feature Engineering
        if not args.skip_feature_engineering:
            timings['Stage 2'] = stage2_feature_engineering(fe, config)
        else:
            print(f"{'=' * 70}")
            print("STAGE 2: Feature Engineering [SKIPPED]")
            print(f"{'=' * 70}\n")

        # Stage 3: Fraud Matching
        if not args.skip_fraud_matching:
            timings['Stage 3'] = stage3_fraud_matching(fr, config, args.min_history)
        else:
            print(f"{'=' * 70}")
            print("STAGE 3: Fraud Matching [SKIPPED]")
            print(f"{'=' * 70}\n")

        # Stage 4: Encoding
        if not args.skip_encoding:
            timings['Stage 4'] = stage4_encoding(enc, config)
        else:
            print(f"{'=' * 70}")
            print("STAGE 4: Feature Encoding [SKIPPED]")
            print(f"{'=' * 70}\n")

        # Stage 5: Vulnerability Scanning (Optional)
        if not args.skip_vuln_scan:
            timings['Stage 5'] = stage5_vulnerability_scan(
                vuln_scanner, config, args.vuln_sample_size
            )
        else:
            print(f"{'=' * 70}")
            print("STAGE 5: Vulnerability Scanning [SKIPPED]")
            print(f"{'=' * 70}\n")

        # Print summary
        total_elapsed = (datetime.now() - pipeline_start).total_seconds()

        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
        print(f"\nFinal Output: {config.get_path('final')}")
        print("  ├── matched/")
        print("  ├── unmatched/")
        print("  └── no_fraud/")
        print("=" * 70 + "\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Pipeline failed!")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        print("\nPlease check the error message and try again.")
        print("=" * 70 + "\n")
        sys.exit(1)


def main():
    """Entry point with multiprocessing protection"""
    # Disable scikit-learn parallel processing to avoid fork issues on macOS
    try:
        from sklearn.utils import parallel_backend
        with parallel_backend('threading', n_jobs=1):
            run_pipeline()
    except ImportError:
        run_pipeline()


if __name__ == "__main__":
    main()
