#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearShield - Automated Training Data Processing Pipeline
Runs the complete 4-stage preprocessing pipeline automatically.
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


def stage1_data_cleaning(dc, config, verbose=True):
    """Stage 1: Data Cleaning"""
    start_time = datetime.now()
    print_stage_header(1, "Data Cleaning")

    # Configure using centralized config
    dc.ENABLE_RENAMING = True
    dc.RAW_DIR = str(config.get_path('raw'))
    dc.CLEANED_DIR = str(config.get_path('cleaned'))

    if verbose:
        print(f"Input:  {dc.RAW_DIR}")
        print(f"Output: {dc.CLEANED_DIR}\n")

    # Run
    dc.main()

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(1, "Data Cleaning", elapsed)
    return elapsed


def stage2_feature_engineering(fe, config, verbose=True):
    """Stage 2: Feature Engineering - Description Encoding and Clustering"""
    start_time = datetime.now()
    print_stage_header(2, "Feature Engineering")

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

    if verbose:
        print(f"Input:  {fe.PROCESSED_DIR}")
        print(f"Model:  {fe.MODEL_NAME}")
        print(f"PCA Dim: {fe.PCA_DIM}, Clusters: {fe.MIN_K}-{fe.MAX_K}\n")

    # Run
    outputs = fe.main()

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(2, "Feature Engineering", elapsed)
    return elapsed


def stage3_fraud_matching(fr, config, min_history_length=None, verbose=True):
    """Stage 3: Fraud Matching and Re-labeling"""
    start_time = datetime.now()
    print_stage_header(3, "Fraud Matching and Re-labeling")

    # Use config default if not specified
    if min_history_length is None:
        min_history_length = config.fraud_matching['min_history_length']

    # Configure using centralized config
    fr.INPUT_DIR = str(config.get_path('clustered'))
    fr.OUTPUT_MEMBER_DIR = str(config.get_path('by_member_temp'))
    fr.OUTPUT_PROCESSED_DIR = str(config.get_path('by_member'))
    fr.CHUNKSIZE = config.fraud_matching['chunksize']

    if verbose:
        print(f"Input:  {fr.INPUT_DIR}")
        print(f"Temp:   {fr.OUTPUT_MEMBER_DIR}")
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


def stage4_encoding(enc, config, verbose=True):
    """Stage 4: Feature Encoding"""
    start_time = datetime.now()
    print_stage_header(4, "Feature Encoding")

    # Configure using centralized config
    enc.PROCESSED_DIR = str(config.get_path('by_member'))
    enc.OUTPUT_DIR = str(config.get_path('final'))
    enc.CONFIG_PATH = str(config.get_path('tokenize_config'))

    if verbose:
        print(f"Input:  {enc.PROCESSED_DIR}")
        print(f"Output: {enc.OUTPUT_DIR}")
        print(f"Config: {enc.CONFIG_PATH}\n")

    # Run
    total_processed = enc.encode_features(enc.PROCESSED_DIR, enc.OUTPUT_DIR, enc.CONFIG_PATH)

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(4, "Feature Encoding", elapsed)
    return elapsed


def run_pipeline():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='ClearShield Automated Training Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_train_pipeline.py                    # Run complete pipeline
  python run_train_pipeline.py --skip-cleaning    # Skip data cleaning step
  python run_train_pipeline.py --min-history 15   # Set minimum history to 15
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
    parser.add_argument('--min-history', type=int, default=None,
                        help='Minimum transaction history length (default: from config)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Initialize configuration
    config = get_train_config()

    # Print header
    print("\n" + "=" * 70)
    print("ClearShield - Automated Training Data Processing Pipeline")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not args.quiet:
        print("\nConfiguration:")
        print(f"  Mode: {config.mode}")
        print(f"  Project Root: {config.PROJECT_ROOT}")
        print(f"  Min History: {args.min_history if args.min_history else config.fraud_matching['min_history_length']}")

    pipeline_start = datetime.now()
    timings = {}

    # Setup directories
    print("\nSetting up directories...")
    config.create_directories(verbose=not args.quiet)
    print("Directories ready\n")

    # Load modules
    print("Loading modules...")
    dc = load_module("data_cleaning", "./01_data_cleaning/01_data_cleaning.py")
    fe = load_module("feature_engineering", "./02_feature_engineering/02_feature_engineering.py")
    fr = load_module("fraud_relabeling", "./03_fraud_relabeling/03_fraud_relabeling.py")
    enc = load_module("encoding", "./04_encoding/04_encoding.py")
    print("All modules loaded\n")

    verbose = not args.quiet

    try:
        # Stage 1: Data Cleaning
        if not args.skip_cleaning:
            timings['Stage 1'] = stage1_data_cleaning(dc, config, verbose)
        else:
            print("\nSkipping Stage 1: Data Cleaning\n")

        # Stage 2: Feature Engineering
        if not args.skip_feature_engineering:
            timings['Stage 2'] = stage2_feature_engineering(fe, config, verbose)
        else:
            print("\nSkipping Stage 2: Feature Engineering\n")

        # Stage 3: Fraud Matching
        if not args.skip_fraud_matching:
            timings['Stage 3'] = stage3_fraud_matching(fr, config, args.min_history, verbose)
        else:
            print("\nSkipping Stage 3: Fraud Matching\n")

        # Stage 4: Encoding
        if not args.skip_encoding:
            timings['Stage 4'] = stage4_encoding(enc, config, verbose)
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

        print(f"\nIntermediate Output: {config.get_path('by_member')}")
        print("  - matched/")
        print("  - unmatched/")
        print("  - no_fraud/")
        print(f"\nFinal Output: {config.get_path('final')}")
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
