#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearShield - Automated Data Processing Pipeline
Runs the complete 4-stage preprocessing pipeline automatically.
"""

import sys
import os
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

# Disable MPS (Apple Silicon GPU) to avoid segmentation faults with transformers
# This forces PyTorch to use CPU for better stability
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def load_module(module_name, file_path):
    """Dynamically load a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def print_stage_header(stage_num, stage_name):
    """Print formatted stage header"""
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num}: {stage_name}")
    print("=" * 70)


def print_stage_footer(stage_num, stage_name, elapsed):
    """Print formatted stage footer with timing"""
    print("=" * 70)
    print(f"STAGE {stage_num} COMPLETE: {stage_name}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print("=" * 70 + "\n")


def setup_directories():
    """Create necessary directories"""
    directories = [
        '../../data/raw',
        '../../data/cleaned',
        '../../data/clustered_out',
        '../../data/by_member',
        '../../data/processed/matched',
        '../../data/processed/unmatched',
        '../../data/processed/no_fraud',
        '../../data/final/matched',
        '../../data/final/unmatched',
        '../../data/final/no_fraud',
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def stage1_data_cleaning(dc, verbose=True):
    """Stage 1: Data Cleaning"""
    start_time = datetime.now()
    print_stage_header(1, "Data Cleaning")

    # Configure
    dc.ENABLE_RENAMING = True
    dc.RAW_DIR = '../../data/raw'
    dc.CLEANED_DIR = '../../data/cleaned'

    if verbose:
        print(f"Input:  {dc.RAW_DIR}")
        print(f"Output: {dc.CLEANED_DIR}\n")

    # Run
    dc.main()

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(1, "Data Cleaning", elapsed)
    return elapsed


def stage2_feature_engineering(fe, verbose=True):
    """Stage 2: Feature Engineering - Description Encoding and Clustering"""
    start_time = datetime.now()
    print_stage_header(2, "Feature Engineering")

    # Configure only the parameters that are actually used
    fe.PROCESSED_DIR = '../../data/cleaned'
    fe.MODEL_NAME = 'prajjwal1/bert-tiny'
    fe.TEXT_COLUMN = 'Transaction Description'
    fe.BATCH_SIZE = 64
    fe.MAX_LENGTH = 64
    fe.PCA_DIM = 20
    fe.CLUSTER_BATCH_SIZE = 4096
    fe.RANDOM_STATE = 42

    if verbose:
        print(f"Input:  {fe.PROCESSED_DIR}")
        print(f"Output: ../../data/clustered_out/")
        print(f"Model:  {fe.MODEL_NAME}")
        print(f"PCA Dimensions: {fe.PCA_DIM}")
        print("Device: CPU (MPS disabled for stability)\n")

    # Run
    outputs = fe.main()

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(2, "Feature Engineering", elapsed)
    return elapsed


def stage3_fraud_matching(fr, min_history_length=10, verbose=True):
    """Stage 3: Fraud Matching and Re-labeling"""
    start_time = datetime.now()
    print_stage_header(3, "Fraud Matching and Re-labeling")

    # Configure
    fr.INPUT_DIR = '../../data/clustered_out'
    fr.OUTPUT_MEMBER_DIR = '../../data/by_member'
    fr.OUTPUT_PROCESSED_DIR = '../../data/processed'
    fr.CHUNKSIZE = 50000

    if verbose:
        print(f"Input:  {fr.INPUT_DIR}")
        print(f"Output: {fr.OUTPUT_PROCESSED_DIR}")
        print(f"Min History Length: {min_history_length}\n")

    # Run Stage 1: Reorganize by member
    print("Stage 3.1: Reorganizing transactions by member...\n")
    num_members = fr.run_stage1()

    # Run Stage 2: Fraud detection
    print("\nStage 3.2: Fraud detection and matching...\n")
    stats = fr.run_stage2(min_history_length)

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(3, "Fraud Matching", elapsed)
    return elapsed


def stage4_encoding(enc, verbose=True):
    """Stage 4: Feature Encoding"""
    start_time = datetime.now()
    print_stage_header(4, "Feature Encoding")

    # Configure
    enc.PROCESSED_DIR = '../../data/processed'
    enc.OUTPUT_DIR = '../../data/final'
    enc.CONFIG_PATH = '../../config/tokenize_dict.json'

    if verbose:
        print(f"Input:  {enc.PROCESSED_DIR}")
        print(f"Output: {enc.OUTPUT_DIR}")
        print(f"Config: {enc.CONFIG_PATH}\n")

    # Run
    total_processed = enc.encode_features(enc.PROCESSED_DIR, enc.OUTPUT_DIR, enc.CONFIG_PATH)

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(4, "Feature Encoding", elapsed)
    return elapsed


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='ClearShield Automated Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run complete pipeline
  python run_pipeline.py --skip-cleaning    # Skip data cleaning step
  python run_pipeline.py --min-history 15   # Set minimum history to 15
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
    parser.add_argument('--min-history', type=int, default=10,
                        help='Minimum transaction history length (default: 10)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print("ClearShield - Automated Data Processing Pipeline")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    pipeline_start = datetime.now()
    timings = {}

    # Setup directories
    print("\nSetting up directories...")
    setup_directories()
    print("Directories ready\n")

    # Load modules
    print("Loading modules...")

    # Monkey patch torch to disable MPS before loading feature engineering module
    try:
        import torch
        if hasattr(torch.backends, 'mps'):
            original_is_available = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False
            print("Note: MPS (Apple GPU) disabled, using CPU for stability")
    except ImportError:
        pass

    dc = load_module("data_cleaning", "./01_data_cleaning/01_data_cleaning.py")
    fe = load_module("feature_engineering", "./02_feature_engineering/02_feature_engineering.py")
    fr = load_module("fraud_relabeling", "./03_fraud_relabeling/03_fraud_relabeling.py")
    enc = load_module("encoding", "./04_encoding/04_encoding.py")
    print("All modules loaded\n")

    verbose = not args.quiet

    try:
        # Stage 1: Data Cleaning
        if not args.skip_cleaning:
            timings['Stage 1'] = stage1_data_cleaning(dc, verbose)
        else:
            print("\nSkipping Stage 1: Data Cleaning\n")

        # Stage 2: Feature Engineering
        if not args.skip_feature_engineering:
            timings['Stage 2'] = stage2_feature_engineering(fe, verbose)
        else:
            print("\nSkipping Stage 2: Feature Engineering\n")

        # Stage 3: Fraud Matching
        if not args.skip_fraud_matching:
            timings['Stage 3'] = stage3_fraud_matching(fr, args.min_history, verbose)
        else:
            print("\nSkipping Stage 3: Fraud Matching\n")

        # Stage 4: Encoding
        if not args.skip_encoding:
            timings['Stage 4'] = stage4_encoding(enc, verbose)
        else:
            print("\nSkipping Stage 4: Feature Encoding\n")

        # Print summary
        total_elapsed = (datetime.now() - pipeline_start).total_seconds()

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTotal Time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")

        if timings:
            print("\nStage Timings:")
            for stage, elapsed in timings.items():
                print(f"  {stage}: {elapsed:.2f}s ({elapsed / 60:.2f}m)")

        print("\nFinal Output: ../../data/final/")
        print("  - matched/")
        print("  - unmatched/")
        print("  - no_fraud/")
        print("\nAll preprocessing complete! Data is ready for model training.")
        print("=" * 70 + "\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Pipeline failed!")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        print("\nPlease check the error message and try again.")
        print("=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
