#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearShield - Automated Prediction Data Processing Pipeline
Runs the complete 4-stage preprocessing pipeline for prediction/inference data.
Uses pre-trained models and simplified fraud labeling.
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
from config.pipeline_config import get_pred_config


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


def stage2_clustering_inference(infer_stage2, config, verbose=True):
    """Stage 2: Description Clustering (INFERENCE - using pre-trained model)"""
    start_time = datetime.now()
    print_stage_header(2, "Description Clustering (Inference)")

    # Get paths from config
    stage2_io = config.get_stage_io(2)
    model_path = str(config.get_path('cluster_model'))

    if verbose:
        print(f"Input:  {stage2_io['input']}")
        print(f"Output: {stage2_io['output']}")
        print(f"Model:  {model_path}")
        print(f"Mode:   INFERENCE ONLY (no training)\n")

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found: {model_path}")

    # Apply pre-trained clustering model
    output_files = infer_stage2.infer_clusters_for_directory(
        input_dir=str(stage2_io['input']),
        output_dir=str(stage2_io['output']),
        model_path=model_path,
        text_column=config.feature_engineering['text_column'],
        batch_size=config.feature_engineering['batch_size'],
        verbose=verbose
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(2, "Description Clustering (Inference)", elapsed)
    return elapsed


def stage3_reorganize_inference(infer_stage3, config, verbose=True):
    """Stage 3: Reorganize by Member (INFERENCE - add Fraud=0)"""
    start_time = datetime.now()
    print_stage_header(3, "Reorganize by Member (Inference)")

    # Get paths from config
    stage3_io = config.get_stage_io(3)

    # Configure
    infer_stage3.INPUT_DIR = str(stage3_io['input'])
    infer_stage3.OUTPUT_MEMBER_DIR = str(stage3_io['output'])
    infer_stage3.CHUNKSIZE = 50000

    if verbose:
        print(f"Input:  {infer_stage3.INPUT_DIR}")
        print(f"Output: {infer_stage3.OUTPUT_MEMBER_DIR}")
        print(f"Mode:   INFERENCE (Fraud=0 for all)\n")

    # Run
    num_members = infer_stage3.main(verbose=verbose)

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(3, "Reorganize by Member (Inference)", elapsed)
    return elapsed


def stage4_encoding_inference(infer_stage4, config, verbose=True):
    """Stage 4: Feature Encoding (INFERENCE)"""
    start_time = datetime.now()
    print_stage_header(4, "Feature Encoding (Inference)")

    # Get paths from config
    stage4_io = config.get_stage_io(4)

    # Configure
    infer_stage4.PROCESSED_DIR = str(stage4_io['input'])
    infer_stage4.OUTPUT_DIR = str(stage4_io['output'])
    infer_stage4.CONFIG_PATH = str(config.get_path('tokenize_config'))

    if verbose:
        print(f"Input:  {infer_stage4.PROCESSED_DIR}")
        print(f"Output: {infer_stage4.OUTPUT_DIR}")
        print(f"Config: {infer_stage4.CONFIG_PATH}\n")

    # Run
    total_processed = infer_stage4.encode_features(
        infer_stage4.PROCESSED_DIR,
        infer_stage4.OUTPUT_DIR,
        infer_stage4.CONFIG_PATH
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print_stage_footer(4, "Feature Encoding (Inference)", elapsed)
    return elapsed


def run_pipeline():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='ClearShield Prediction Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pred_pipeline.py                    # Run complete prediction pipeline
  python run_pred_pipeline.py --skip-cleaning    # Skip data cleaning step
        """
    )

    parser.add_argument('--skip-cleaning', action='store_true',
                        help='Skip Stage 1: Data Cleaning')
    parser.add_argument('--skip-clustering', action='store_true',
                        help='Skip Stage 2: Description Clustering')
    parser.add_argument('--skip-reorganize', action='store_true',
                        help='Skip Stage 3: Reorganize by Member')
    parser.add_argument('--skip-encoding', action='store_true',
                        help='Skip Stage 4: Feature Encoding')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Initialize configuration
    config = get_pred_config()

    # Print header
    print("\n" + "=" * 70)
    print("ClearShield - Prediction Data Processing Pipeline")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Mode: INFERENCE (using pre-trained models)")
    print("=" * 70)

    if not args.quiet:
        print("\nConfiguration:")
        print(f"  Mode: {config.mode}")
        print(f"  Project Root: {config.PROJECT_ROOT}")
        print(f"  Cluster Model: {config.get_path('cluster_model')}")

    pipeline_start = datetime.now()
    timings = {}

    # Setup directories
    print("\nSetting up directories...")
    config.create_directories(verbose=not args.quiet)
    print("Directories ready\n")

    # Load modules
    print("Loading modules...")
    dc = load_module("data_cleaning", "./01_data_cleaning/01_data_cleaning.py")
    infer_stage2 = load_module("inference_stage2", "./02_feature_engineering/inference_stage2.py")
    infer_stage3 = load_module("inference_stage3", "./03_fraud_relabeling/inference_stage3.py")
    infer_stage4 = load_module("inference_stage4", "./04_encoding/inference_stage4.py")
    print("All modules loaded\n")

    verbose = not args.quiet

    try:
        # Stage 1: Data Cleaning
        if not args.skip_cleaning:
            timings['Stage 1'] = stage1_data_cleaning(dc, config, verbose)
        else:
            print("\nSkipping Stage 1: Data Cleaning\n")

        # Stage 2: Clustering Inference
        if not args.skip_clustering:
            timings['Stage 2'] = stage2_clustering_inference(infer_stage2, config, verbose)
        else:
            print("\nSkipping Stage 2: Description Clustering\n")

        # Stage 3: Reorganize by Member
        if not args.skip_reorganize:
            timings['Stage 3'] = stage3_reorganize_inference(infer_stage3, config, verbose)
        else:
            print("\nSkipping Stage 3: Reorganize by Member\n")

        # Stage 4: Encoding
        if not args.skip_encoding:
            timings['Stage 4'] = stage4_encoding_inference(infer_stage4, config, verbose)
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

        print(f"\nFinal Output: {config.get_path('final')}")
        print("  - member_*.csv files ready for prediction")
        print("\nAll preprocessing complete! Data is ready for model inference.")
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
