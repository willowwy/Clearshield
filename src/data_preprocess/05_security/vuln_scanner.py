"""
ClearShield Elder Fraud Detection - Comprehensive Testing Framework
Updated to use a Transformer Encoder model
Updated with automatic data loading from ../data/final folder
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
from datetime import datetime
import json
from pathlib import Path
import glob
import os

# ============================================================================
# DATA LOADER FOR FINAL FOLDER
# ============================================================================


class FinalDataLoader:
    """
    Loads processed data from the final folder for vulnerability scanning.
    Handles data from ../data/final/ directory structure.
    """

    def __init__(self, final_data_path: str = "../data/final"):
        """
        Initialize the data loader.

        Args:
            final_data_path: Path to the final data directory
        """
        self.final_data_path = Path(final_data_path)
        self.matched_path = self.final_data_path / "matched"
        self.unmatched_path = self.final_data_path / "unmatched"
        self.no_fraud_path = self.final_data_path / "no_fraud"

    def load_all_data(self,
                      file_pattern: str = "*.csv",
                      sample_size: Optional[int] = None,
                      include_matched: bool = True,
                      include_unmatched: bool = True,
                      include_no_fraud: bool = True) -> Tuple[pd.DataFrame, torch.Tensor]:
        """
        Load all data from the final folder.

        Args:
            file_pattern: Pattern to match files (e.g., "*.csv")
            sample_size: If provided, randomly sample this many rows
            include_matched: Whether to include matched fraud data
            include_unmatched: Whether to include unmatched fraud data
            include_no_fraud: Whether to include no fraud data

        Returns:
            Tuple of (DataFrame, Tensor) for testing
        """
        print(f"Loading data from: {self.final_data_path}")

        all_dataframes = []

        # Load from each subdirectory
        if include_matched and self.matched_path.exists():
            matched_df = self._load_from_directory(
                self.matched_path, file_pattern)
            if matched_df is not None:
                matched_df['fraud_category'] = 'matched'
                all_dataframes.append(matched_df)
                print(f"  ✓ Loaded {len(matched_df)} rows from matched/")

        if include_unmatched and self.unmatched_path.exists():
            unmatched_df = self._load_from_directory(
                self.unmatched_path, file_pattern)
            if unmatched_df is not None:
                unmatched_df['fraud_category'] = 'unmatched'
                all_dataframes.append(unmatched_df)
                print(f"  ✓ Loaded {len(unmatched_df)} rows from unmatched/")

        if include_no_fraud and self.no_fraud_path.exists():
            no_fraud_df = self._load_from_directory(
                self.no_fraud_path, file_pattern)
            if no_fraud_df is not None:
                no_fraud_df['fraud_category'] = 'no_fraud'
                all_dataframes.append(no_fraud_df)
                print(f"  ✓ Loaded {len(no_fraud_df)} rows from no_fraud/")

        if not all_dataframes:
            raise ValueError(
                f"No data found in {self.final_data_path}. Check path and file pattern.")

        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"  ✓ Total rows loaded: {len(combined_df)}")

        # Sample if requested
        if sample_size and sample_size < len(combined_df):
            combined_df = combined_df.sample(n=sample_size, random_state=42)
            print(f"  ✓ Sampled to {sample_size} rows")

        # Convert to tensor for model testing
        tensor_data = self._dataframe_to_tensor(combined_df)
        print(f"  ✓ Converted to tensor shape: {tensor_data.shape}")

        return combined_df, tensor_data

    def _load_from_directory(self, directory: Path, pattern: str) -> Optional[pd.DataFrame]:
        """Load all files matching pattern from a directory."""
        if not directory.exists():
            return None

        files = list(directory.glob(pattern))
        if not files:
            return None

        dataframes = []
        for file in files:
            df = self._load_file(file)
            if df is not None:
                dataframes.append(df)

        if not dataframes:
            return None

        return pd.concat(dataframes, ignore_index=True)

    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single file based on its extension."""
        try:
            if file_path.suffix == '.csv':
                return pd.read_csv(file_path)
            else:
                print(f"  ⚠ Unsupported file format: {file_path.suffix}")
                return None
        except Exception as e:
            print(f"  ⚠ Error loading {file_path}: {str(e)}")
            return None

    def _dataframe_to_tensor(self, df: pd.DataFrame,
                             sequence_length: int = 10,
                             feature_columns: Optional[List[str]] = None) -> torch.Tensor:
        """
        Convert DataFrame to PyTorch tensor for sequence model testing.

        Args:
            df: Input DataFrame
            sequence_length: Length of sequences for the model
            feature_columns: Specific columns to use as features (if None, uses numeric columns)

        Returns:
            Tensor of shape (batch_size, sequence_length, num_features)
        """
        # Select feature columns
        if feature_columns is None:
            # Use all numeric columns except labels / IDs / timestamps
            exclude_cols = ['is_fraud', 'fraud_category', 'user_id', 'member_id',
                            'transaction_id', 'timestamp']
            feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                               if col not in exclude_cols]

        if not feature_columns:
            # If no numeric columns found, create dummy features
            print("  No numeric feature columns found, using dummy data")
            num_samples = min(len(df), 100)
            return torch.randn(num_samples, sequence_length, 20)

        # Extract features
        features = df[feature_columns].values

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        # Create sequences
        num_features = len(feature_columns)
        num_samples = len(features) // sequence_length

        if num_samples == 0:
            # Not enough data for even one sequence
            num_samples = 1
            # Pad or truncate to sequence_length
            if len(features) < sequence_length:
                padding = np.zeros(
                    (sequence_length - len(features), num_features))
                features = np.vstack([features, padding])
            else:
                features = features[:sequence_length]

        # Reshape to (num_samples, sequence_length, num_features)
        try:
            tensor_data = features[:num_samples * sequence_length].reshape(
                num_samples, sequence_length, num_features
            )
        except Exception:
            # Fallback: just use available data
            tensor_data = features.reshape(1, -1, num_features)
            if tensor_data.shape[1] < sequence_length:
                # Pad to sequence_length
                padding = np.zeros(
                    (1, sequence_length - tensor_data.shape[1], num_features))
                tensor_data = np.concatenate([tensor_data, padding], axis=1)
            elif tensor_data.shape[1] > sequence_length:
                # Truncate to sequence_length
                tensor_data = tensor_data[:, :sequence_length, :]

        return torch.FloatTensor(tensor_data)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing tests"""
    required_columns: List[str] = field(
        default_factory=lambda: ['transaction_id', 'user_id', 'amount', 'timestamp'])
    key_columns_for_duplicates: List[str] = field(
        default_factory=lambda: ['transaction_id'])
    user_id_column: str = 'user_id'
    fraud_column: str = 'is_fraud'
    n_preceding_transactions: int = 10
    min_transactions_per_user: int = 5


@dataclass
class ClusteringConfig:
    """Configuration for clustering module tests"""
    default_n_clusters: int = 5
    k_range_min: int = 2
    k_range_max: int = 11
    random_state: int = 42
    reproducibility_test_runs: int = 5
    davies_bouldin_good_threshold: float = 1.0
    davies_bouldin_moderate_threshold: float = 2.0
    silhouette_min_acceptable: float = 0.3


@dataclass
class TransformerModelConfig:
    """
    Configuration for model tests (now used for Transformer Encoder).
    Name kept for compatibility.
    """
    expected_input_size: int = 20
    expected_output_size: int = 1
    expected_sequence_length: int = 10
    recall_target_min: float = 0.70
    recall_target_max: float = 0.75
    convergence_threshold: float = 0.01
    convergence_window_epochs: int = 5
    max_false_positive_rate: float = 0.15


@dataclass
class PerformanceConfig:
    """Configuration for performance and integration tests"""
    latency_test_iterations: int = 100
    latency_warmup_runs: int = 10
    p99_threshold_ms: float = 250.0
    p50_threshold_ms: float = 50.0
    p95_threshold_ms: float = 150.0
    end_to_end_threshold_ms: float = 1000.0
    concurrent_requests_light: int = 50
    concurrent_requests_medium: int = 100
    concurrent_requests_heavy: int = 500
    model_save_path: str = "/tmp/test_model.pt"
    min_throughput_qps: float = 10.0


@dataclass
class SecurityConfig:
    """Configuration for security and compliance tests"""
    # PII detection
    prohibited_pii_fields: List[str] = field(default_factory=lambda: [
        'ssn', 'social_security', 'social_security_number',
        'full_name', 'first_name', 'last_name',
        'driver_license', 'drivers_license',
        'passport', 'passport_number',
        'phone_number', 'phone', 'mobile',
        'email', 'email_address',
        'address', 'street_address', 'home_address',
        'date_of_birth', 'dob', 'birth_date'
    ])

    # Adversarial attack parameters
    fgsm_epsilon: float = 0.1
    fgsm_high_sensitivity_threshold: float = 0.3
    fgsm_moderate_sensitivity_threshold: float = 0.1
    adversarial_test_epsilons: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.15])

    # Input validation
    extreme_value_multiplier: float = 1e6

    # Model poisoning detection
    parameter_sparsity_threshold: float = 0.9
    parameter_extreme_value_threshold: float = 100.0
    parameter_uniqueness_threshold: float = 0.1
    min_parameter_size_for_check: int = 100

    # Privacy leakage
    privacy_noise_level: float = 0.01
    membership_inference_high_risk_threshold: float = 0.5
    membership_inference_moderate_risk_threshold: float = 0.2

    # Model extraction
    query_throughput_threshold_qps: float = 1000.0


@dataclass
class TestSuiteConfig:
    """Master configuration for entire test suite"""
    preprocessing: DataPreprocessingConfig = field(
        default_factory=DataPreprocessingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    Transformer: TransformerModelConfig = field(
        default_factory=TransformerModelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Global settings
    enable_preprocessing_tests: bool = True
    enable_clustering_tests: bool = True
    # Name kept; now controls Transformer tests
    enable_Transformer_tests: bool = True
    enable_performance_tests: bool = True
    enable_security_tests: bool = True
    verbose: bool = True

    # Data loading settings
    final_data_path: str = "../data/final"
    data_file_pattern: str = "*.csv"
    data_sample_size: Optional[int] = None

    @classmethod
    def from_json(cls, json_path: str) -> 'TestSuiteConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        return cls(
            preprocessing=DataPreprocessingConfig(
                **config_dict.get('preprocessing', {})),
            clustering=ClusteringConfig(**config_dict.get('clustering', {})),
            Transformer=TransformerModelConfig(
                **config_dict.get('Transformer', {})),
            performance=PerformanceConfig(
                **config_dict.get('performance', {})),
            security=SecurityConfig(**config_dict.get('security', {})),
            **{k: v for k, v in config_dict.items() if k not in
               ['preprocessing', 'clustering', 'Transformer', 'performance', 'security']}
        )

    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'preprocessing': self.preprocessing.__dict__,
            'clustering': self.clustering.__dict__,
            'Transformer': self.Transformer.__dict__,
            'performance': self.performance.__dict__,
            'security': self.security.__dict__,
            'enable_preprocessing_tests': self.enable_preprocessing_tests,
            'enable_clustering_tests': self.enable_clustering_tests,
            'enable_Transformer_tests': self.enable_Transformer_tests,
            'enable_performance_tests': self.enable_performance_tests,
            'enable_security_tests': self.enable_security_tests,
            'verbose': self.verbose,
            'final_data_path': self.final_data_path,
            'data_file_pattern': self.data_file_pattern,
            'data_sample_size': self.data_sample_size
        }

        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


# ============================================================================
# TEST RESULT DATA STRUCTURES
# ============================================================================

class TestStatus(Enum):
    """Status of individual test cases"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


class VulnerabilityLevel(Enum):
    """Severity levels for security vulnerabilities"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    module: str
    status: TestStatus
    message: str
    execution_time: float
    details: Dict = None


@dataclass
class Vulnerability:
    """Security vulnerability finding"""
    name: str
    level: VulnerabilityLevel
    description: str
    recommendation: str
    details: Dict


# ============================================================================
# DATA PREPROCESSING TESTING
# ============================================================================

class DataPreprocessingTests:
    """Tests for data cleaning, preprocessing, and feature engineering"""

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.results: List[TestResult] = []

    def test_missing_value_handling(self, df: pd.DataFrame,
                                    required_columns: Optional[List[str]] = None) -> TestResult:
        """Test that missing values are properly handled"""
        start_time = time.time()
        columns_to_check = required_columns or self.config.required_columns

        try:
            # Only check columns that exist in the dataframe
            existing_columns = [
                col for col in columns_to_check if col in df.columns]

            if not existing_columns:
                status = TestStatus.SKIPPED
                message = f"None of the required columns {columns_to_check} found in dataframe"
                details = {"available_columns": df.columns.tolist()}
            else:
                missing_counts = df[existing_columns].isnull().sum()

                if missing_counts.sum() > 0:
                    status = TestStatus.FAILED
                    message = f"Found {missing_counts.sum()} missing values in required columns"
                    details = {"missing_by_column": missing_counts.to_dict()}
                else:
                    status = TestStatus.PASSED
                    message = "No missing values in required columns"
                    details = {"total_rows": len(
                        df), "columns_checked": existing_columns}

            execution_time = time.time() - start_time
            result = TestResult("Missing Value Handling", "Data Preprocessing",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Missing Value Handling", "Data Preprocessing",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_duplicate_detection(self, df: pd.DataFrame,
                                 key_columns: Optional[List[str]] = None) -> TestResult:
        """Test duplicate transaction detection and removal"""
        start_time = time.time()
        columns_to_check = key_columns or self.config.key_columns_for_duplicates

        try:
            existing_columns = [
                col for col in columns_to_check if col in df.columns]

            if not existing_columns:
                status = TestStatus.SKIPPED
                message = f"Key columns {columns_to_check} not found in dataframe"
                details = {"available_columns": df.columns.tolist()}
            else:
                duplicates = df.duplicated(subset=existing_columns, keep=False)
                duplicate_count = duplicates.sum()

                if duplicate_count > 0:
                    status = TestStatus.WARNING
                    message = f"Found {duplicate_count} duplicate transactions"
                    details = {"duplicate_count": duplicate_count,
                               "total_rows": len(df)}
                else:
                    status = TestStatus.PASSED
                    message = "No duplicate transactions found"
                    details = {"total_rows": len(df)}

            execution_time = time.time() - start_time
            result = TestResult("Duplicate Detection", "Data Preprocessing",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Duplicate Detection", "Data Preprocessing",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_feature_extraction(self, df: pd.DataFrame,
                                user_id_col: Optional[str] = None,
                                n_preceding: Optional[int] = None) -> TestResult:
        """Test feature extraction for n preceding transactions per user"""
        start_time = time.time()
        user_col = user_id_col or self.config.user_id_column
        n_trans = n_preceding or self.config.n_preceding_transactions

        try:
            if user_col not in df.columns:
                status = TestStatus.SKIPPED
                message = f"User ID column '{user_col}' not found in dataframe"
                details = {"available_columns": df.columns.tolist()}
            else:
                user_groups = df.groupby(user_col)
                users_with_insufficient_data = []

                for user_id, group in user_groups:
                    if len(group) < n_trans:
                        users_with_insufficient_data.append(user_id)

                if len(users_with_insufficient_data) > 0:
                    insufficient_pct = (
                        len(users_with_insufficient_data) / len(user_groups)) * 100
                    status = TestStatus.WARNING
                    message = f"{len(users_with_insufficient_data)} users ({insufficient_pct:.1f}%) have < {n_trans} transactions"
                    details = {
                        "insufficient_data_users": len(users_with_insufficient_data),
                        "n_preceding": n_trans,
                        "total_users": len(user_groups),
                        "percentage": insufficient_pct,
                        "sample_users": list(users_with_insufficient_data[:5])
                    }
                else:
                    status = TestStatus.PASSED
                    message = f"All users have sufficient transaction history (>= {n_trans})"
                    details = {"total_users": len(
                        user_groups), "n_preceding": n_trans}

            execution_time = time.time() - start_time
            result = TestResult("Feature Extraction Validation", "Data Preprocessing",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Feature Extraction Validation", "Data Preprocessing",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_fraud_flag_conversion(self, df: pd.DataFrame,
                                   fraud_col: Optional[str] = None) -> TestResult:
        """Test that fraud indicator is properly converted to binary flag"""
        start_time = time.time()
        fraud_column = fraud_col or self.config.fraud_column

        try:
            if fraud_column not in df.columns:
                status = TestStatus.SKIPPED
                message = f"Fraud column '{fraud_column}' not found in dataframe"
                details = {"available_columns": df.columns.tolist()}
            else:
                unique_values = df[fraud_column].unique()

                # Check if values are binary (0 and 1)
                if set(unique_values).issubset({0, 1}):
                    status = TestStatus.PASSED
                    message = "Fraud flag properly converted to binary"
                    details = {
                        "unique_values": list(unique_values),
                        "distribution": df[fraud_column].value_counts().to_dict()
                    }
                else:
                    status = TestStatus.FAILED
                    message = f"Fraud flag has non-binary values: {unique_values}"
                    details = {"unique_values": list(unique_values)}

            execution_time = time.time() - start_time
            result = TestResult("Fraud Flag Conversion", "Data Preprocessing",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Fraud Flag Conversion", "Data Preprocessing",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result


# ============================================================================
# CLUSTERING MODULE TESTING
# ============================================================================

class ClusteringModuleTests:
    """Tests for K-Means clustering implementation"""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.results: List[TestResult] = []

    def test_kmeans_implementation(self, features: np.ndarray,
                                   n_clusters: Optional[int] = None) -> TestResult:
        """Test basic K-Means implementation"""
        start_time = time.time()
        n_clust = n_clusters or self.config.default_n_clusters

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clust,
                            random_state=self.config.random_state)
            labels = kmeans.fit_predict(features)

            unique_labels = len(np.unique(labels))

            if unique_labels == n_clust:
                status = TestStatus.PASSED
                message = f"K-Means successfully created {n_clust} clusters"
                details = {"n_clusters": n_clust,
                           "inertia": float(kmeans.inertia_)}
            else:
                status = TestStatus.WARNING
                message = f"Expected {n_clust} clusters but got {unique_labels}"
                details = {"expected": n_clust, "actual": unique_labels}

            execution_time = time.time() - start_time
            result = TestResult("K-Means Implementation", "Clustering Module",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("K-Means Implementation", "Clustering Module",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_optimal_k_validation(self, features: np.ndarray,
                                  k_range: Optional[range] = None) -> TestResult:
        """Test optimal K selection using silhouette score"""
        start_time = time.time()
        k_values = k_range or range(
            self.config.k_range_min, self.config.k_range_max)

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            silhouette_scores = {}

            for k in k_values:
                kmeans = KMeans(
                    n_clusters=k, random_state=self.config.random_state)
                labels = kmeans.fit_predict(features)
                score = silhouette_score(features, labels)
                silhouette_scores[k] = score

            optimal_k = max(silhouette_scores, key=silhouette_scores.get)
            optimal_score = silhouette_scores[optimal_k]

            if optimal_score >= self.config.silhouette_min_acceptable:
                status = TestStatus.PASSED
                message = f"Optimal K: {optimal_k} (silhouette: {optimal_score:.3f})"
            else:
                status = TestStatus.WARNING
                message = f"Optimal K: {optimal_k}, but low silhouette score: {optimal_score:.3f}"

            details = {"optimal_k": optimal_k,
                       "optimal_score": optimal_score, "all_scores": silhouette_scores}

            execution_time = time.time() - start_time
            result = TestResult("Optimal K Validation", "Clustering Module",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Optimal K Validation", "Clustering Module",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_cluster_reproducibility(self, features: np.ndarray,
                                     n_clusters: Optional[int] = None,
                                     n_runs: Optional[int] = None) -> TestResult:
        """Test that clustering is reproducible across multiple runs"""
        start_time = time.time()
        n_clust = n_clusters or self.config.default_n_clusters
        runs = n_runs or self.config.reproducibility_test_runs

        try:
            from sklearn.cluster import KMeans

            all_labels = []
            for run in range(runs):
                kmeans = KMeans(n_clusters=n_clust,
                                random_state=self.config.random_state)
                labels = kmeans.fit_predict(features)
                all_labels.append(labels)

            first_run = all_labels[0]
            all_identical = all(np.array_equal(first_run, labels)
                                for labels in all_labels)

            if all_identical:
                status = TestStatus.PASSED
                message = f"Clustering is reproducible across {runs} runs"
                details = {"n_runs": runs, "reproducible": True}
            else:
                status = TestStatus.FAILED
                message = "Clustering results vary across runs"
                details = {"n_runs": runs, "reproducible": False}

            execution_time = time.time() - start_time
            result = TestResult("Cluster Reproducibility", "Clustering Module",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Cluster Reproducibility", "Clustering Module",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_davies_bouldin_index(self, features: np.ndarray,
                                  n_clusters: Optional[int] = None) -> TestResult:
        """Test Davies-Bouldin Index for cluster quality"""
        start_time = time.time()
        n_clust = n_clusters or self.config.default_n_clusters

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import davies_bouldin_score

            kmeans = KMeans(n_clusters=n_clust,
                            random_state=self.config.random_state)
            labels = kmeans.fit_predict(features)
            db_score = davies_bouldin_score(features, labels)

            if db_score < self.config.davies_bouldin_good_threshold:
                status = TestStatus.PASSED
                message = f"Good cluster separation (DB Index: {db_score:.3f})"
            elif db_score < self.config.davies_bouldin_moderate_threshold:
                status = TestStatus.WARNING
                message = f"Moderate cluster separation (DB Index: {db_score:.3f})"
            else:
                status = TestStatus.FAILED
                message = f"Poor cluster separation (DB Index: {db_score:.3f})"

            details = {
                "davies_bouldin_index": db_score,
                "n_clusters": n_clust,
                "good_threshold": self.config.davies_bouldin_good_threshold,
                "moderate_threshold": self.config.davies_bouldin_moderate_threshold
            }

            execution_time = time.time() - start_time
            result = TestResult("Davies-Bouldin Index", "Clustering Module",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Davies-Bouldin Index", "Clustering Module",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result


# ============================================================================
# MODEL TESTING (NOW FOR TRANSFORMER ENCODER)
# ============================================================================

class TransformerModelTests:
    """
    Tests for model architecture, training, and prediction accuracy.
    (Class name kept for compatibility; logic updated for Transformer Encoder.)
    """

    def __init__(self, model: nn.Module, config: TransformerModelConfig):
        self.model = model
        self.config = config
        self.results: List[TestResult] = []

    def test_architecture_validation(self, expected_input_size: Optional[int] = None,
                                     expected_output_size: Optional[int] = None,
                                     expected_seq_length: Optional[int] = None) -> TestResult:
        """Validate Transformer architecture structure"""
        start_time = time.time()

        input_size = expected_input_size or self.config.expected_input_size
        output_size = expected_output_size or self.config.expected_output_size
        seq_length = expected_seq_length or self.config.expected_sequence_length

        try:
            has_transformer = any(isinstance(m, nn.TransformerEncoder)
                                  for m in self.model.modules())

            if not has_transformer:
                status = TestStatus.FAILED
                message = "Model does not contain TransformerEncoder layers"
                details = {"has_transformer": False}
            else:
                self.model.eval()
                # Use configuration parameters for dummy input
                dummy_input = torch.randn(1, seq_length, input_size)

                with torch.no_grad():
                    output = self.model(dummy_input)

                if output.shape[-1] == output_size:
                    status = TestStatus.PASSED
                    message = "Transformer architecture validated successfully"
                    details = {
                        "has_transformer": True,
                        "expected_input_size": input_size,
                        "expected_output_size": output_size,
                        "actual_output_size": output.shape[-1],
                        "sequence_length": seq_length
                    }
                else:
                    status = TestStatus.FAILED
                    message = f"Output size mismatch: expected {output_size}, got {output.shape[-1]}"
                    details = {"expected": output_size,
                               "actual": output.shape[-1]}

            execution_time = time.time() - start_time
            result = TestResult("Architecture Validation", "Model (Transformer)",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Architecture Validation", "Model (Transformer)",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_training_convergence(self, train_losses: List[float],
                                  convergence_threshold: Optional[float] = None,
                                  window_size: Optional[int] = None) -> TestResult:
        """Test that model training converges properly"""
        start_time = time.time()

        threshold = convergence_threshold or self.config.convergence_threshold
        window = window_size or self.config.convergence_window_epochs

        try:
            if len(train_losses) < 2:
                status = TestStatus.SKIPPED
                message = "Insufficient training history"
                details = {"epochs": len(train_losses)}
            else:
                initial_loss = train_losses[0]
                final_loss = train_losses[-1]
                loss_reduction = (initial_loss - final_loss) / \
                    initial_loss if initial_loss != 0 else 0

                # Check convergence in recent epochs
                recent_losses = train_losses[-window:]
                loss_variance = np.var(recent_losses)

                if loss_reduction > 0 and loss_variance < threshold:
                    status = TestStatus.PASSED
                    message = f"Model converged (loss reduced by {loss_reduction*100:.1f}%)"
                    details = {
                        "initial_loss": initial_loss,
                        "final_loss": final_loss,
                        "loss_reduction": loss_reduction,
                        "recent_variance": loss_variance,
                        "threshold": threshold,
                        "window_size": window
                    }
                elif loss_reduction > 0:
                    status = TestStatus.WARNING
                    message = "Loss decreasing but not yet stabilized"
                    details = {"loss_reduction": loss_reduction,
                               "variance": loss_variance, "threshold": threshold}
                else:
                    status = TestStatus.FAILED
                    message = "Training loss not decreasing"
                    details = {"initial_loss": initial_loss,
                               "final_loss": final_loss}

            execution_time = time.time() - start_time
            result = TestResult("Training Convergence", "Model (Transformer)",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Training Convergence", "Model (Transformer)",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_fraud_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      recall_target: Optional[float] = None,
                                      max_fpr: Optional[float] = None) -> TestResult:
        """Test fraud detection accuracy metrics (REQ-F1)"""
        start_time = time.time()

        recall_min = recall_target or self.config.recall_target_min
        fpr_max = max_fpr or self.config.max_false_positive_rate

        try:
            from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            # Check requirements
            meets_recall = recall >= recall_min
            meets_fpr = fpr <= fpr_max

            if meets_recall and meets_fpr:
                status = TestStatus.PASSED
                message = f"Meets requirements: Recall={recall*100:.1f}%, FPR={fpr*100:.1f}%"
            elif meets_recall:
                status = TestStatus.WARNING
                message = f"Recall OK ({recall*100:.1f}%) but FPR too high ({fpr*100:.1f}%)"
            else:
                status = TestStatus.FAILED
                message = f"Recall below target: {recall*100:.1f}% (target: {recall_min*100:.0f}%)"

            details = {
                "recall": recall,
                "recall_target": recall_min,
                "precision": precision,
                "f1_score": f1,
                "false_positive_rate": fpr,
                "fpr_max_allowed": fpr_max,
                "confusion_matrix": cm.tolist(),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            }

            execution_time = time.time() - start_time
            result = TestResult("Fraud Detection Accuracy (REQ-F1)", "Model (Transformer)",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Fraud Detection Accuracy (REQ-F1)", "Model (Transformer)",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_sequence_input_formatting(self, sample_sequences: torch.Tensor,
                                       expected_seq_length: Optional[int] = None) -> TestResult:
        """Test that sequence inputs are properly formatted for the Transformer"""
        start_time = time.time()
        seq_length = expected_seq_length or self.config.expected_sequence_length

        try:
            if sample_sequences.dim() != 3:
                status = TestStatus.FAILED
                message = f"Invalid sequence dimensions: expected 3D, got {sample_sequences.dim()}D"
                details = {"shape": list(sample_sequences.shape)}
            elif sample_sequences.shape[1] != seq_length:
                status = TestStatus.WARNING
                message = f"Sequence length mismatch: expected {seq_length}, got {sample_sequences.shape[1]}"
                details = {"expected": seq_length,
                           "actual": sample_sequences.shape[1]}
            else:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(sample_sequences)

                status = TestStatus.PASSED
                message = "Sequence input formatting validated"
                details = {
                    "input_shape": list(sample_sequences.shape),
                    "output_shape": list(output.shape),
                    "expected_seq_length": seq_length
                }

            execution_time = time.time() - start_time
            result = TestResult("Sequence Input Formatting", "Model (Transformer)",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Sequence Input Formatting", "Model (Transformer)",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result


# ============================================================================
# PERFORMANCE & INTEGRATION TESTING
# ============================================================================

class PerformanceIntegrationTests:
    """Tests for latency, throughput, and end-to-end integration"""

    def __init__(self, model: nn.Module, config: PerformanceConfig):
        self.model = model
        self.config = config
        self.results: List[TestResult] = []

    def test_inference_latency(self, sample_input: torch.Tensor,
                               n_iterations: Optional[int] = None,
                               warmup_runs: Optional[int] = None,
                               p99_threshold_ms: Optional[float] = None) -> TestResult:
        """Test P99 inference latency (REQ-P1)"""
        start_time = time.time()

        iterations = n_iterations or self.config.latency_test_iterations
        warmup = warmup_runs or self.config.latency_warmup_runs
        p99_limit = p99_threshold_ms or self.config.p99_threshold_ms

        try:
            self.model.eval()
            latencies = []

            # Warm-up runs
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.model(sample_input)

            # Actual measurement
            for _ in range(iterations):
                iter_start = time.time()
                with torch.no_grad():
                    _ = self.model(sample_input)
                iter_end = time.time()
                latencies.append((iter_end - iter_start) * 1000)

            # Calculate percentiles
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            mean_latency = np.mean(latencies)

            if p99 <= p99_limit:
                status = TestStatus.PASSED
                message = f"P99 latency within threshold: {p99:.2f}ms (limit: {p99_limit}ms)"
                print(
                    f"   ✓ Performance OK: P99={p99:.2f}ms, Mean={mean_latency:.2f}ms")
            else:
                status = TestStatus.FAILED
                message = f"P99 latency exceeds threshold: {p99:.2f}ms (limit: {p99_limit}ms)"

                # Detailed failure explanation
                print(f"\n   ❌ PERFORMANCE FAILURE - LATENCY TOO HIGH!")
                print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"   Current P99 Latency: {p99:.2f}ms")
                print(f"   Required Threshold:  {p99_limit:.2f}ms")
                print(
                    f"   Exceeded by:         {p99 - p99_limit:.2f}ms ({((p99/p99_limit - 1)*100):.1f}% over limit)")
                print(f"   ")
                print(f"   Latency Breakdown:")
                print(f"   - P50 (median):  {p50:.2f}ms")
                print(f"   - P95:           {p95:.2f}ms")
                print(f"   - P99:           {p99:.2f}ms (FAILED)")
                print(f"   - Mean:          {mean_latency:.2f}ms")
                print(f"   ")
                print(f"   Why This Matters:")
                print(f"   For real-time fraud detection, transactions must be")
                print(f"   analyzed in <250ms. Current speed is too slow to stop")
                print(f"   fraudulent transactions before they complete.")
                print(f"   ")
                print(f"   How to Fix:")
                print(
                    f"   1. Reduce model complexity (fewer layers/smaller hidden size)")
                print(f"   2. Use model quantization or pruning")
                print(f"   3. Move model to GPU if available")
                print(f"   4. Optimize preprocessing pipeline")
                print(f"   5. Consider model distillation (train smaller model)")
                print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

            details = {
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "mean_ms": mean_latency,
                "threshold_ms": p99_limit,
                "n_iterations": iterations,
                "warmup_runs": warmup
            }

            execution_time = time.time() - start_time
            result = TestResult("Inference Latency P99 (REQ-P1)", "Performance & Integration",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Inference Latency P99 (REQ-P1)", "Performance & Integration",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_end_to_end_decision_time(self, preprocessing_time_ms: float,
                                      inference_time_ms: float,
                                      postprocessing_time_ms: float,
                                      threshold_ms: Optional[float] = None) -> TestResult:
        """Test end-to-end decision time (REQ-P2)"""
        start_time = time.time()
        limit_ms = threshold_ms or self.config.end_to_end_threshold_ms

        try:
            total_time_ms = preprocessing_time_ms + \
                inference_time_ms + postprocessing_time_ms

            if total_time_ms <= limit_ms:
                status = TestStatus.PASSED
                message = f"End-to-end decision time within limit: {total_time_ms:.2f}ms"
            else:
                status = TestStatus.FAILED
                message = f"End-to-end decision time exceeds limit: {total_time_ms:.2f}ms (limit: {limit_ms}ms)"

            details = {
                "preprocessing_ms": preprocessing_time_ms,
                "inference_ms": inference_time_ms,
                "postprocessing_ms": postprocessing_time_ms,
                "total_ms": total_time_ms,
                "threshold_ms": limit_ms,
                "breakdown_pct": {
                    "preprocessing": (preprocessing_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0,
                    "inference": (inference_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0,
                    "postprocessing": (postprocessing_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
                }
            }

            execution_time = time.time() - start_time
            result = TestResult("End-to-End Decision Time (REQ-P2)", "Performance & Integration",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("End-to-End Decision Time (REQ-P2)", "Performance & Integration",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_concurrent_load(self, sample_input: torch.Tensor,
                             n_concurrent_requests: Optional[int] = None,
                             load_level: str = "medium") -> TestResult:
        """Test system behavior under concurrent load"""
        start_time = time.time()

        # Select load level from config
        if load_level == "light":
            n_requests = self.config.concurrent_requests_light
        elif load_level == "heavy":
            n_requests = self.config.concurrent_requests_heavy
        else:
            n_requests = self.config.concurrent_requests_medium

        if n_concurrent_requests is not None:
            n_requests = n_concurrent_requests

        try:
            self.model.eval()

            # Simulate concurrent requests by batching
            batch_input = sample_input.repeat(n_requests, 1, 1)

            batch_start = time.time()
            with torch.no_grad():
                outputs = self.model(batch_input)
            batch_end = time.time()

            total_time_ms = (batch_end - batch_start) * 1000
            avg_time_per_request_ms = total_time_ms / n_requests
            throughput_qps = n_requests / (batch_end - batch_start)

            if avg_time_per_request_ms <= self.config.p99_threshold_ms:
                status = TestStatus.PASSED
                message = f"Handled {n_requests} concurrent requests successfully"
            else:
                status = TestStatus.WARNING
                message = f"Average latency {avg_time_per_request_ms:.2f}ms under load"

            details = {
                "n_concurrent_requests": n_requests,
                "load_level": load_level,
                "total_time_ms": total_time_ms,
                "avg_time_per_request_ms": avg_time_per_request_ms,
                "throughput_qps": throughput_qps,
                "min_throughput_qps": self.config.min_throughput_qps
            }

            execution_time = time.time() - start_time
            result = TestResult("Concurrent Load Testing", "Performance & Integration",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Concurrent Load Testing", "Performance & Integration",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_model_serialization(self, save_path: Optional[str] = None) -> TestResult:
        """Test model can be saved and loaded correctly"""
        start_time = time.time()
        path = save_path or self.config.model_save_path

        try:
            # Save model
            torch.save(self.model.state_dict(), path)

            # Create new model instance and load
            new_model = self.model.__class__(*self._get_model_init_args())
            new_model.load_state_dict(torch.load(path))
            new_model.eval()

            # Compare outputs
            test_input = torch.randn(1, 10, self._infer_input_size())
            with torch.no_grad():
                original_output = self.model(test_input)
                loaded_output = new_model(test_input)

            if torch.allclose(original_output, loaded_output, rtol=1e-5):
                status = TestStatus.PASSED
                message = "Model serialization/deserialization successful"
                details = {"save_path": path, "outputs_match": True}
            else:
                status = TestStatus.FAILED
                message = "Loaded model outputs differ from original"
                details = {"save_path": path, "outputs_match": False}

            execution_time = time.time() - start_time
            result = TestResult("Model Serialization", "Performance & Integration",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Model Serialization", "Performance & Integration",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def _get_model_init_args(self):
        """Helper to infer model initialization arguments"""
        # Using default arguments of the model's __init__
        return ()

    def _infer_input_size(self):
        """Helper to infer input size from model (for Transformer)"""
        # If model has an input projection layer, use it
        if hasattr(self.model, "input_proj") and isinstance(self.model.input_proj, nn.Linear):
            return self.model.input_proj.in_features
        return 20


# ============================================================================
# SECURITY & COMPLIANCE TESTING
# ============================================================================

class SecurityComplianceTests:
    """Security and compliance tests for fraud detection system"""

    def __init__(self, model: nn.Module, config: SecurityConfig):
        self.model = model
        self.config = config
        self.results: List[TestResult] = []
        self.vulnerabilities: List[Vulnerability] = []

    def test_pii_compliance(self, data_columns: List[str],
                            prohibited_fields: Optional[List[str]] = None) -> TestResult:
        """Test PII compliance (REQ-F5)"""
        start_time = time.time()
        prohibited = prohibited_fields or self.config.prohibited_pii_fields

        try:
            found_pii = [col for col in data_columns
                         if any(pii in col.lower() for pii in prohibited)]

            if len(found_pii) > 0:
                status = TestStatus.FAILED
                message = f"Found {len(found_pii)} columns with potential PII: {found_pii}"
                details = {"pii_columns": found_pii,
                           "total_columns": len(data_columns)}

                self.vulnerabilities.append(Vulnerability(
                    name="PII Data Exposure",
                    level=VulnerabilityLevel.CRITICAL,
                    description=f"Dataset contains PII columns: {found_pii}",
                    recommendation="Remove or hash PII fields. Use only de-identified transaction signals per NCUA/FinCEN requirements.",
                    details={"pii_columns": found_pii}
                ))
            else:
                status = TestStatus.PASSED
                message = "No direct PII columns detected"
                details = {"columns_checked": len(
                    data_columns), "prohibited_fields_checked": len(prohibited)}

            execution_time = time.time() - start_time
            result = TestResult("PII Compliance (REQ-F5)", "Security & Compliance",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("PII Compliance (REQ-F5)", "Security & Compliance",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_adversarial_robustness(self, sample_data: torch.Tensor,
                                    epsilon: Optional[float] = None) -> TestResult:
        """Test model robustness against adversarial attacks"""
        start_time = time.time()
        eps = epsilon or self.config.fgsm_epsilon

        try:
            self.model.eval()
            data = sample_data.clone().requires_grad_(True)

            output = self.model(data)
            if output.dim() == 1:
                output = output.unsqueeze(0)

            loss = output.mean()
            loss.backward()

            data_grad = data.grad.data
            perturbed_data = data + eps * data_grad.sign()

            with torch.no_grad():
                original_pred = self.model(data)
                perturbed_pred = self.model(perturbed_data)

            pred_change = torch.abs(
                original_pred - perturbed_pred).mean().item()

            if pred_change > self.config.fgsm_high_sensitivity_threshold:
                status = TestStatus.FAILED
                message = f"High adversarial sensitivity: {pred_change:.3f}"

                self.vulnerabilities.append(Vulnerability(
                    name="High Adversarial Sensitivity",
                    level=VulnerabilityLevel.HIGH,
                    description=f"Model predictions change significantly ({pred_change:.3f}) under FGSM attack with epsilon={eps}",
                    recommendation="Implement adversarial training with elder fraud attack patterns",
                    details={"prediction_change": pred_change, "epsilon": eps}
                ))
            elif pred_change > self.config.fgsm_moderate_sensitivity_threshold:
                status = TestStatus.WARNING
                message = f"Moderate adversarial sensitivity: {pred_change:.3f}"
            else:
                status = TestStatus.PASSED
                message = f"Good adversarial robustness: {pred_change:.3f}"

            details = {
                "prediction_change": pred_change,
                "epsilon": eps,
                "high_threshold": self.config.fgsm_high_sensitivity_threshold,
                "moderate_threshold": self.config.fgsm_moderate_sensitivity_threshold
            }

            execution_time = time.time() - start_time
            result = TestResult("Adversarial Robustness", "Security & Compliance",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Adversarial Robustness", "Security & Compliance",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_input_validation(self, sample_data: torch.Tensor,
                              extreme_multiplier: Optional[float] = None) -> TestResult:
        """Test model handles malformed inputs gracefully"""
        start_time = time.time()
        multiplier = extreme_multiplier or self.config.extreme_value_multiplier

        try:
            self.model.eval()
            test_cases = []

            # Test 1: Extreme values
            try:
                extreme_data = torch.ones_like(sample_data) * multiplier
                with torch.no_grad():
                    extreme_output = self.model(extreme_data)

                if torch.isnan(extreme_output).any() or torch.isinf(extreme_output).any():
                    test_cases.append(
                        ("extreme_values", "FAIL", "NaN/Inf output"))
                else:
                    test_cases.append(
                        ("extreme_values", "PASS", "Handled correctly"))
            except Exception as e:
                test_cases.append(("extreme_values", "FAIL", str(e)))

            # Test 2: Zero values
            try:
                zero_data = torch.zeros_like(sample_data)
                with torch.no_grad():
                    zero_output = self.model(zero_data)
                test_cases.append(("zero_values", "PASS", "Handled correctly"))
            except Exception as e:
                test_cases.append(("zero_values", "FAIL", str(e)))

            # Test 3: Negative values
            try:
                negative_data = -torch.abs(sample_data)
                with torch.no_grad():
                    negative_output = self.model(negative_data)
                test_cases.append(
                    ("negative_values", "PASS", "Handled correctly"))
            except Exception as e:
                test_cases.append(("negative_values", "FAIL", str(e)))

            failures = [tc for tc in test_cases if tc[1] == "FAIL"]

            if len(failures) > 0:
                status = TestStatus.FAILED
                message = f"{len(failures)}/{len(test_cases)} input validation tests failed"

                self.vulnerabilities.append(Vulnerability(
                    name="Input Validation Missing",
                    level=VulnerabilityLevel.CRITICAL,
                    description="Model crashes or produces invalid outputs on edge case inputs",
                    recommendation="Add input validation, normalization, and error handling before model inference",
                    details={"failed_tests": failures}
                ))
            else:
                status = TestStatus.PASSED
                message = f"All {len(test_cases)} input validation tests passed"

            details = {"test_cases": test_cases,
                       "extreme_multiplier": multiplier}

            execution_time = time.time() - start_time
            result = TestResult("Input Validation", "Security & Compliance",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Input Validation", "Security & Compliance",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_model_poisoning_indicators(self) -> TestResult:
        """Check for indicators of model poisoning attacks"""
        start_time = time.time()

        try:
            suspicious_patterns = []

            for name, param in self.model.named_parameters():
                if 'weight' in name and param.numel() > self.config.min_parameter_size_for_check:
                    param_data = param.data.flatten()

                    # Check 1: Suspicious sparsity
                    zeros_ratio = (param_data == 0).float().mean().item()
                    if zeros_ratio > self.config.parameter_sparsity_threshold:
                        suspicious_patterns.append({
                            "parameter": name,
                            "issue": "high_sparsity",
                            "zeros_ratio": zeros_ratio,
                            "threshold": self.config.parameter_sparsity_threshold
                        })

                    # Check 2: Extreme values
                    max_abs_val = param_data.abs().max().item()
                    if max_abs_val > self.config.parameter_extreme_value_threshold:
                        suspicious_patterns.append({
                            "parameter": name,
                            "issue": "extreme_values",
                            "max_abs_value": max_abs_val,
                            "threshold": self.config.parameter_extreme_value_threshold
                        })

                    # Check 3: Unusual distributions
                    unique_ratio = len(torch.unique(
                        param_data)) / param.numel()
                    if unique_ratio < self.config.parameter_uniqueness_threshold:
                        suspicious_patterns.append({
                            "parameter": name,
                            "issue": "low_uniqueness",
                            "unique_ratio": unique_ratio,
                            "threshold": self.config.parameter_uniqueness_threshold
                        })

            if len(suspicious_patterns) > 0:
                status = TestStatus.WARNING
                message = f"Found {len(suspicious_patterns)} suspicious parameter patterns"

                self.vulnerabilities.append(Vulnerability(
                    name="Suspicious Model Parameters",
                    level=VulnerabilityLevel.MEDIUM,
                    description="Model parameters show unusual patterns that may indicate poisoning",
                    recommendation="Review model training provenance and validate against known good checkpoints",
                    details={"patterns": suspicious_patterns[:5]}
                ))
            else:
                status = TestStatus.PASSED
                message = "No model poisoning indicators detected"

            details = {
                "suspicious_patterns": suspicious_patterns[:10],
                "total_patterns": len(suspicious_patterns),
                "parameters_checked": sum(1 for _, p in self.model.named_parameters()
                                          if 'weight' in _ and p.numel() > self.config.min_parameter_size_for_check)
            }

            execution_time = time.time() - start_time
            result = TestResult("Model Poisoning Indicators", "Security & Compliance",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Model Poisoning Indicators", "Security & Compliance",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result

    def test_privacy_leakage(self, sample_data: torch.Tensor,
                             noise_level: Optional[float] = None,
                             high_risk_threshold: Optional[float] = None) -> TestResult:
        """Test for membership inference vulnerability"""
        start_time = time.time()
        noise = noise_level or self.config.privacy_noise_level
        threshold = high_risk_threshold or self.config.membership_inference_high_risk_threshold

        try:
            self.model.eval()

            with torch.no_grad():
                output1 = self.model(sample_data)

                perturbation = torch.randn_like(sample_data) * noise
                modified_data = sample_data + perturbation
                output2 = self.model(modified_data)

            confidence_diff = torch.abs(output1 - output2).mean().item()

            if confidence_diff > threshold:
                status = TestStatus.WARNING
                message = f"High membership inference risk: {confidence_diff:.3f}"

                self.vulnerabilities.append(Vulnerability(
                    name="High Membership Inference Risk",
                    level=VulnerabilityLevel.HIGH,
                    description=f"Large confidence variation ({confidence_diff:.3f}) for similar inputs",
                    recommendation="Implement differential privacy techniques or confidence smoothing to protect elder financial data",
                    details={"confidence_difference": confidence_diff,
                             "threshold": threshold}
                ))
            elif confidence_diff > self.config.membership_inference_moderate_risk_threshold:
                status = TestStatus.WARNING
                message = f"Moderate membership inference risk: {confidence_diff:.3f}"
            else:
                status = TestStatus.PASSED
                message = f"Low membership inference risk: {confidence_diff:.3f}"

            details = {
                "confidence_difference": confidence_diff,
                "noise_level": noise,
                "high_risk_threshold": threshold,
                "moderate_risk_threshold": self.config.membership_inference_moderate_risk_threshold
            }

            execution_time = time.time() - start_time
            result = TestResult("Privacy Leakage Risk", "Security & Compliance",
                                status, message, execution_time, details)
            self.results.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult("Privacy Leakage Risk", "Security & Compliance",
                                TestStatus.FAILED, f"Error: {str(e)}", execution_time)
            self.results.append(result)
            return result


# ============================================================================
# COMPREHENSIVE TEST SUITE ORCHESTRATOR (UPDATED WITH DATA LOADER)
# ============================================================================

class ClearShieldTestSuite:
    """Main test orchestrator for ClearShield Elder Fraud Detection System"""

    def __init__(self, model: nn.Module,
                 config: Optional[TestSuiteConfig] = None,
                 sample_data: Optional[torch.Tensor] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 auto_load_data: bool = False):
        """
        Initialize the test suite.

        Args:
            model: Sequence model (Transformer Encoder) for fraud detection
            config: Test configuration (uses defaults if None)
            sample_data: Sample tensor data for testing (optional if auto_load_data=True)
            dataframe: DataFrame with transaction data (optional if auto_load_data=True)
            auto_load_data: If True, automatically load data from final folder
        """
        self.model = model
        self.config = config or TestSuiteConfig()

        # Load data automatically if requested
        if auto_load_data:
            print("\n" + "="*70)
            print("LOADING DATA FROM FINAL FOLDER")
            print("="*70)
            data_loader = FinalDataLoader(self.config.final_data_path)
            self.dataframe, self.sample_data = data_loader.load_all_data(
                file_pattern=self.config.data_file_pattern,
                sample_size=self.config.data_sample_size
            )
            print("="*70 + "\n")
        else:
            self.sample_data = sample_data
            self.dataframe = dataframe

        # Initialize test modules based on configuration
        self.preprocessing_tests = DataPreprocessingTests(
            self.config.preprocessing)
        self.clustering_tests = ClusteringModuleTests(self.config.clustering)
        self.Transformer_tests = TransformerModelTests(
            model, self.config.Transformer)
        self.performance_tests = PerformanceIntegrationTests(
            model, self.config.performance)
        self.security_tests = SecurityComplianceTests(
            model, self.config.security)

        self.all_results: List[TestResult] = []
        self.all_vulnerabilities: List[Vulnerability] = []

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite across all modules"""
        print("="*70)
        print("CLEARSHIELD FRAUD DETECTION - COMPREHENSIVE TEST SUITE")
        print("="*70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Run tests based on configuration flags
        test_count = 0
        total_modules = sum([
            self.config.enable_preprocessing_tests,
            self.config.enable_clustering_tests,
            self.config.enable_Transformer_tests,
            self.config.enable_performance_tests,
            self.config.enable_security_tests
        ])

        if self.config.enable_preprocessing_tests and self.dataframe is not None:
            test_count += 1
            print(
                f"\n[{test_count}/{total_modules}] Running Data Preprocessing Tests...")
            print("-" * 70)
            self._run_preprocessing_tests()

        if self.config.enable_clustering_tests and self.sample_data is not None:
            test_count += 1
            print(
                f"\n[{test_count}/{total_modules}] Running Clustering Module Tests...")
            print("-" * 70)
            self._run_clustering_tests()

        if self.config.enable_Transformer_tests:
            test_count += 1
            print(
                f"\n[{test_count}/{total_modules}] Running Model Tests (Transformer Encoder)...")
            print("-" * 70)
            self._run_Transformer_tests()

        if self.config.enable_performance_tests and self.sample_data is not None:
            test_count += 1
            print(
                f"\n[{test_count}/{total_modules}] Running Performance & Integration Tests...")
            print("-" * 70)
            self._run_performance_tests()

        if self.config.enable_security_tests:
            test_count += 1
            print(
                f"\n[{test_count}/{total_modules}] Running Security & Compliance Tests...")
            print("-" * 70)
            self._run_security_tests()

        # Collect all results
        self._collect_results()

        # Generate and return summary
        return self.generate_summary()

    def _run_preprocessing_tests(self):
        """Run all data preprocessing tests"""
        if self.dataframe is not None:
            self.preprocessing_tests.test_missing_value_handling(
                self.dataframe)
            self.preprocessing_tests.test_duplicate_detection(self.dataframe)
            self.preprocessing_tests.test_feature_extraction(self.dataframe)
            self.preprocessing_tests.test_fraud_flag_conversion(self.dataframe)

    def _run_clustering_tests(self):
        """Run all clustering module tests"""
        if self.sample_data is not None:
            features = self.sample_data.numpy().reshape(
                self.sample_data.shape[0], -1)

            self.clustering_tests.test_kmeans_implementation(features)
            self.clustering_tests.test_optimal_k_validation(features)
            self.clustering_tests.test_cluster_reproducibility(features)
            self.clustering_tests.test_davies_bouldin_index(features)

    def _run_Transformer_tests(self):
        """Run all model tests (Transformer)"""
        self.Transformer_tests.test_architecture_validation()

        if self.sample_data is not None:
            self.Transformer_tests.test_sequence_input_formatting(
                self.sample_data)

    def _run_performance_tests(self):
        """Run all performance and integration tests"""
        if self.sample_data is not None:
            self.performance_tests.test_inference_latency(self.sample_data)
            self.performance_tests.test_concurrent_load(self.sample_data)
            self.performance_tests.test_model_serialization()

    def _run_security_tests(self):
        """Run all security and compliance tests"""
        if self.dataframe is not None:
            self.security_tests.test_pii_compliance(
                self.dataframe.columns.tolist())

        if self.sample_data is not None:
            self.security_tests.test_adversarial_robustness(self.sample_data)
            self.security_tests.test_input_validation(self.sample_data)
            self.security_tests.test_privacy_leakage(self.sample_data)

        self.security_tests.test_model_poisoning_indicators()

    def _collect_results(self):
        """Collect all test results and vulnerabilities"""
        self.all_results.extend(self.preprocessing_tests.results)
        self.all_results.extend(self.clustering_tests.results)
        self.all_results.extend(self.Transformer_tests.results)
        self.all_results.extend(self.performance_tests.results)
        self.all_results.extend(self.security_tests.results)

        self.all_vulnerabilities.extend(self.security_tests.vulnerabilities)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        status_counts = {status: 0 for status in TestStatus}
        for result in self.all_results:
            status_counts[result.status] += 1

        vuln_counts = {level: 0 for level in VulnerabilityLevel}
        for vuln in self.all_vulnerabilities:
            vuln_counts[vuln.level] += 1

        total_time = sum(r.execution_time for r in self.all_results)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "final_data_path": self.config.final_data_path,
                "data_file_pattern": self.config.data_file_pattern,
                "preprocessing": self.config.preprocessing.__dict__,
                "clustering": self.config.clustering.__dict__,
                "Transformer": self.config.Transformer.__dict__,
                "performance": self.config.performance.__dict__,
                "security": {k: v for k, v in self.config.security.__dict__.items()
                             if not k.startswith('prohibited_pii')}
            },
            "total_tests": len(self.all_results),
            "status_breakdown": {k.value: v for k, v in status_counts.items()},
            "total_vulnerabilities": len(self.all_vulnerabilities),
            "vulnerability_breakdown": {k.value: v for k, v in vuln_counts.items()},
            "total_execution_time_seconds": total_time,
            "test_results": [
                {
                    "test_name": r.test_name,
                    "module": r.module,
                    "status": r.status.value,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "details": r.details
                }
                for r in self.all_results
            ],
            "vulnerabilities": [
                {
                    "name": v.name,
                    "level": v.level.value,
                    "description": v.description,
                    "recommendation": v.recommendation,
                    "details": v.details
                }
                for v in self.all_vulnerabilities
            ]
        }

        return summary

    def print_report(self):
        """Print formatted test report to console"""
        summary = self.generate_summary()

        print("\n" + "="*70)
        print("TEST EXECUTION SUMMARY")
        print("="*70)
        print(f"Total Tests Run: {summary['total_tests']}")
        print(
            f"Total Execution Time: {summary['total_execution_time_seconds']:.2f}s\n")

        print("Test Results by Status:")
        for status, count in summary['status_breakdown'].items():
            if count > 0:
                print(f"  {status}: {count}")

        print(
            f"\nTotal Security Vulnerabilities: {summary['total_vulnerabilities']}")
        if summary['total_vulnerabilities'] > 0:
            print("Vulnerabilities by Severity:")
            for level, count in summary['vulnerability_breakdown'].items():
                if count > 0:
                    print(f"  {level}: {count}")

        print("\n" + "-"*70)
        print("DETAILED RESULTS BY MODULE")
        print("-"*70)

        # Group results by module
        by_module = {}
        for result in self.all_results:
            if result.module not in by_module:
                by_module[result.module] = []
            by_module[result.module].append(result)

        # Print each module's results
        for module, results in by_module.items():
            print(f"\n{module}:")
            for result in results:
                status_symbol = {"PASSED": "PASS", "FAILED": "FAIL",
                                 "WARNING": "WARN", "SKIPPED": "SKIP"}
                symbol = status_symbol.get(result.status.value, "INFO")
                print(f"  [{symbol}] {result.test_name}")
                print(f"        {result.message}")
                if result.status == TestStatus.FAILED and result.details and self.config.verbose:
                    print(f"        Details: {result.details}")

        # Print security vulnerabilities
        if len(self.all_vulnerabilities) > 0:
            print("\n" + "-"*70)
            print("SECURITY VULNERABILITIES")
            print("-"*70)

            sorted_vulns = sorted(self.all_vulnerabilities,
                                  key=lambda v: list(VulnerabilityLevel).index(v.level))

            for i, vuln in enumerate(sorted_vulns, 1):
                print(f"\n{i}. [{vuln.level.value}] {vuln.name}")
                print(f"   Description: {vuln.description}")
                print(f"   Recommendation: {vuln.recommendation}")
                if vuln.details and self.config.verbose:
                    print(f"   Details: {vuln.details}")

        print("\n" + "="*70)
        print("END OF REPORT")
        print("="*70)

    def export_results(self, json_path: str):
        """Export test results to JSON file"""
        summary = self.generate_summary()
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nTest results exported to: {json_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a Transformer Encoder model for fraud detection
    class FraudDetectionTransformer(nn.Module):
        def __init__(self, input_size=20, model_dim=64, num_heads=4, num_layers=2, dropout=0.1):
            super().__init__()
            # Project raw features into model dimension
            self.input_proj = nn.Linear(input_size, model_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

            self.fc = nn.Linear(model_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x: (batch, seq, features)
            x = self.input_proj(x)
            encoded = self.transformer(x)       # (batch, seq, model_dim)
            last_output = encoded[:, -1, :]     # use last time step
            output = self.fc(last_output)
            return self.sigmoid(output)

    print("=" * 70)
    print("EXAMPLE: Using Automatic Data Loading from ../data/final Folder")
    print("="*70)

    model = FraudDetectionTransformer(
        input_size=20, model_dim=64, num_heads=4, num_layers=2)

    # Configure to load from your final folder
    config = TestSuiteConfig(
        final_data_path="../data/final",   # Path to your final folder
        data_file_pattern="*.csv",         # Or "*.pkl", "*.parquet", etc.
        data_sample_size=1000,             # Optional: sample size for testing
        verbose=True
    )

    # Create test suite with automatic data loading
    test_suite = ClearShieldTestSuite(
        model,
        config=config,
        auto_load_data=True  # This will automatically load from ../data/final/
    )

    # Run all tests
    results = test_suite.run_all_tests()
    test_suite.print_report()
    test_suite.export_results('test_results.json')

    print("\n" + "="*70)
    print("TESTING COMPLETED")
    print("="*70)
