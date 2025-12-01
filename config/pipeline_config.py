#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClearShield - Pipeline Configuration
Centralized configuration for all data preprocessing paths and parameters.
"""

from pathlib import Path
import os


class PipelineConfig:
    """Centralized configuration for data preprocessing pipeline"""

    def __init__(self, mode='train'):
        """
        Initialize pipeline configuration

        Args:
            mode (str): 'train' or 'pred' - determines which data paths to use
        """
        if mode not in ['train', 'pred']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'pred'")

        self.mode = mode

        # Auto-detect project root (config/ is at project root)
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_ROOT = self.PROJECT_ROOT / 'data'
        self.CONFIG_ROOT = self.PROJECT_ROOT / 'config'
        self.SRC_ROOT = self.PROJECT_ROOT / 'src'

        # Initialize paths based on mode
        self._init_paths()

        # Initialize model parameters
        self._init_model_params()

    def _init_paths(self):
        """Initialize all data paths based on mode"""
        mode_root = self.DATA_ROOT / self.mode

        # Common paths for both train and pred
        self.paths = {
            'raw': mode_root / 'raw',
            'cleaned': mode_root / 'cleaned',
            'clustered': mode_root / 'clustered_out',
            'by_member': mode_root / 'by_member',
            'final': mode_root / 'final',
        }

        # Mode-specific paths (build sub-paths based on parent paths)
        if self.mode == 'train':
            # by_member subdirectories
            self.paths['by_member_temp'] = self.paths['by_member'] / 'temp'
            self.paths['by_member_matched'] = self.paths['by_member'] / 'matched'
            self.paths['by_member_unmatched'] = self.paths['by_member'] / 'unmatched'
            self.paths['by_member_no_fraud'] = self.paths['by_member'] / 'no_fraud'
            # final subdirectories
            self.paths['final_matched'] = self.paths['final'] / 'matched'
            self.paths['final_unmatched'] = self.paths['final'] / 'unmatched'
            self.paths['final_no_fraud'] = self.paths['final'] / 'no_fraud'

        # Model and config paths (shared between train and pred)
        self.paths['model_dir'] = (
            self.SRC_ROOT / 'data_preprocess' / '02_feature_engineering' /
            '02b_description_encoding'
        )
        self.paths['cluster_model'] = self.paths['model_dir'] / 'global_cluster_model.pkl'
        self.paths['tokenize_config'] = self.CONFIG_ROOT / 'tokenize_dict.json'

    def _init_model_params(self):
        """Initialize model and processing parameters"""
        # Stage 2: Feature Engineering
        self.feature_engineering = {
            'model_name': 'prajjwal1/bert-tiny',
            'text_column': 'Transaction Description',
            'batch_size': 64,
            'max_length': 64,
            'pca_dim': 20,
            'min_k': 10,
            'max_k': 60,
            'k_step': 10,
            'sample_size': 10000,
            'cluster_batch_size': 4096,
            'random_state': 42,
        }

        # Stage 3: Fraud Matching (train mode only)
        self.fraud_matching = {
            'chunksize': 50000,
            'min_history_length': 10,
        }

    def get_path(self, key):
        """
        Get a specific path

        Args:
            key (str): Path key (e.g., 'raw', 'cleaned', 'final')

        Returns:
            Path: The requested path object
        """
        if key not in self.paths:
            raise KeyError(f"Path '{key}' not found. Available paths: {list(self.paths.keys())}")
        return self.paths[key]

    def get_all_paths(self):
        """Get all paths as a dictionary"""
        return self.paths.copy()

    def create_directories(self, verbose=True):
        """
        Create all necessary directories for the current mode

        Args:
            verbose (bool): Print created directories
        """
        # Determine which directories to create based on mode
        if self.mode == 'train':
            dirs_to_create = [
                'raw', 'cleaned', 'clustered',
                'by_member_temp', 'by_member_matched',
                'by_member_unmatched', 'by_member_no_fraud',
                'final_matched', 'final_unmatched', 'final_no_fraud',
                'model_dir'
            ]
        else:  # pred
            dirs_to_create = [
                'raw', 'cleaned', 'clustered',
                'by_member', 'final', 'model_dir'
            ]

        created = []
        for key in dirs_to_create:
            path = self.paths[key]
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(str(path))

        if verbose and created:
            print(f"Created {len(created)} directories:")
            for d in created:
                print(f"  ✓ {d}")
        elif verbose:
            print("All directories already exist")

    def get_stage_io(self, stage):
        """
        Get input/output paths for a specific pipeline stage

        Args:
            stage (int): Stage number (1, 2, 3, or 4)

        Returns:
            dict: Dictionary with 'input' and 'output' paths
        """
        stage_map = {
            1: {'input': 'raw', 'output': 'cleaned'},
            2: {'input': 'cleaned', 'output': 'clustered'},
            3: {
                'input': 'clustered',
                'temp': 'by_member_temp' if self.mode == 'train' else 'by_member',
                'output': 'by_member'
            },
            4: {'input': 'by_member', 'output': 'final'},
        }

        if stage not in stage_map:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, 3, or 4")

        io_keys = stage_map[stage]
        return {k: self.paths[v] for k, v in io_keys.items()}

    def to_dict(self):
        """Export configuration as dictionary (for debugging/logging)"""
        return {
            'mode': self.mode,
            'project_root': str(self.PROJECT_ROOT),
            'paths': {k: str(v) for k, v in self.paths.items()},
            'feature_engineering': self.feature_engineering,
            'fraud_matching': self.fraud_matching,
        }

    def print_config(self):
        """Print current configuration in a readable format"""
        print("=" * 70)
        print(f"ClearShield Pipeline Configuration - Mode: {self.mode.upper()}")
        print("=" * 70)
        print(f"\nProject Root: {self.PROJECT_ROOT}")
        print(f"\nData Paths:")
        for key, path in self.paths.items():
            print(f"  {key:20s}: {path}")
        print(f"\nFeature Engineering Parameters:")
        for key, value in self.feature_engineering.items():
            print(f"  {key:20s}: {value}")
        if self.mode == 'train':
            print(f"\nFraud Matching Parameters:")
            for key, value in self.fraud_matching.items():
                print(f"  {key:20s}: {value}")
        print("=" * 70)


# Convenience functions for quick access
def get_config(mode='train'):
    """
    Get a pipeline configuration object

    Args:
        mode (str): 'train' or 'pred'

    Returns:
        PipelineConfig: Configuration object
    """
    return PipelineConfig(mode=mode)


def get_train_config():
    """Get training pipeline configuration"""
    return PipelineConfig(mode='train')


def get_pred_config():
    """Get prediction pipeline configuration"""
    return PipelineConfig(mode='pred')


# Example usage
if __name__ == "__main__":
    # Test train mode
    print("Testing TRAIN mode:")
    train_config = get_train_config()
    train_config.print_config()

    print("\n" * 2)

    # Test pred mode
    print("Testing PRED mode:")
    pred_config = get_pred_config()
    pred_config.print_config()

    # Test stage I/O
    print("\n" * 2)
    print("Stage I/O paths (train mode):")
    for stage in [1, 2, 3, 4]:
        print(f"\nStage {stage}:")
        io = train_config.get_stage_io(stage)
        for k, v in io.items():
            print(f"  {k}: {v}")
