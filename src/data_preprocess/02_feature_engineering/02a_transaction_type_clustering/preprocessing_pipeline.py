# ============================================
# COMPLETE DATA PREPROCESSING PIPELINE
# File: preprocessing_pipeline.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


class TransactionPreprocessor:
    """
    Complete transaction_type_clustering pipeline for transaction 01_data_cleaning
    Includes: cleaning, encoding, clustering (feature reduction)
    """

    def __init__(self, n_clusters=4, sample_size=300000, random_state=42):
        self.n_clusters = n_clusters
        self.sample_size = sample_size
        self.random_state = random_state

        # Components to be fitted
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.kmeans = None
        self.categorical_features = ['Account Type', 'Action Type', 'Source Type']

        # Metadata
        self.is_fitted = False
        self.cluster_stats = None

    def fit(self, df, verbose=True):
        """
        Fit the transaction_type_clustering pipeline on training 01_data_cleaning

        Args:
            df: Training dataframe
            verbose: Print progress info
        """
        if verbose:
            print("=" * 60)
            print("FITTING PREPROCESSING PIPELINE")
            print("=" * 60)

        # -----------------------------------------
        # Step 1: Sample 01_data_cleaning for clustering
        # -----------------------------------------
        if len(df) > self.sample_size:
            if verbose:
                print(f"\n[1/5] Sampling {self.sample_size:,} from {len(df):,} transactions")

            # Stratified sampling
            df['_strata'] = (
                    df['Account Type'].astype(str) + '_' +
                    df['Action Type'].astype(str) + '_' +
                    df['Source Type'].astype(str)
            )

            sampling_fraction = self.sample_size / len(df)
            df_sample = df.groupby('_strata', group_keys=False).apply(
                lambda x: x.sample(frac=sampling_fraction, random_state=self.random_state)
            ).reset_index(drop=True)
            df_sample = df_sample.drop('_strata', axis=1)

            if '_strata' in df.columns:
                df = df.drop('_strata', axis=1)
        else:
            df_sample = df.copy()
            if verbose:
                print(f"\n[1/5] Using all {len(df):,} transactions")

        # -----------------------------------------
        # Step 2: Handle missing values
        # -----------------------------------------
        if verbose:
            print(f"\n[2/5] Handling missing values")

        for col in self.categorical_features:
            df_sample[col] = df_sample[col].fillna('Unknown')

        # -----------------------------------------
        # Step 3: Fit label encoders
        # -----------------------------------------
        if verbose:
            print(f"\n[3/5] Fitting label encoders")

        encoded_features = []
        for col in self.categorical_features:
            le = LabelEncoder()
            encoded_col = le.fit_transform(df_sample[col])
            encoded_features.append(encoded_col)
            self.label_encoders[col] = le

            if verbose:
                print(f"  {col}: {len(le.classes_)} categories")

        X_encoded = np.column_stack(encoded_features)

        # -----------------------------------------
        # Step 4: Fit scaler
        # -----------------------------------------
        if verbose:
            print(f"\n[4/5] Fitting scaler")

        X_scaled = self.scaler.fit_transform(X_encoded)

        # -----------------------------------------
        # Step 5: Fit KMeans clustering
        # -----------------------------------------
        if verbose:
            print(f"\n[5/5] Fitting KMeans clustering (k={self.n_clusters})")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,
            max_iter=500
        )

        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Calculate metrics
        sil_score = silhouette_score(X_scaled, cluster_labels)
        db_score = davies_bouldin_score(X_scaled, cluster_labels)

        if verbose:
            print(f"\n  Silhouette Score: {sil_score:.4f}")
            print(f"  Davies-Bouldin Index: {db_score:.4f}")

        # Store cluster statistics
        df_sample['Transaction_Pattern_Cluster'] = cluster_labels
        self.cluster_stats = self._calculate_cluster_stats(df_sample)

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print("✓ PIPELINE FITTED SUCCESSFULLY")
            print("=" * 60)
            self._print_cluster_summary()

        return self

    def transform(self, df, verbose=True):
        """
        Transform 01_data_cleaning using fitted pipeline

        Args:
            df: Dataframe to transform
            verbose: Print progress info

        Returns:
            Transformed dataframe with cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call .fit() first.")

        if verbose:
            print("=" * 60)
            print(f"TRANSFORMING {len(df):,} TRANSACTIONS")
            print("=" * 60)

        df_transformed = df.copy()

        # Handle missing values
        for col in self.categorical_features:
            df_transformed[col] = df_transformed[col].fillna('Unknown')

        # Encode 02_feature_engineering
        encoded_features = []
        for col in self.categorical_features:
            le = self.label_encoders[col]

            # Handle unseen categories
            df_transformed[col] = df_transformed[col].apply(
                lambda x: x if x in le.classes_ else 'Unknown'
            )

            encoded_col = le.transform(df_transformed[col])
            encoded_features.append(encoded_col)

        X_encoded = np.column_stack(encoded_features)

        # Scale 02_feature_engineering
        X_scaled = self.scaler.transform(X_encoded)

        # Predict clusters
        df_transformed['Transaction_Pattern_Cluster'] = self.kmeans.predict(X_scaled)

        if verbose:
            print(f"\n✓ Transformation complete")
            print(f"\nCluster distribution:")
            cluster_dist = df_transformed['Transaction_Pattern_Cluster'].value_counts().sort_index()
            for cluster_id, count in cluster_dist.items():
                print(f"  Cluster {cluster_id}: {count:,} ({count / len(df_transformed) * 100:.1f}%)")

        return df_transformed

    def fit_transform(self, df, verbose=True):
        """
        Fit pipeline and transform 01_data_cleaning in one step
        """
        self.fit(df, verbose=verbose)
        return self.transform(df, verbose=verbose)

    def save(self, save_dir):
        """
        Save fitted pipeline components
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        os.makedirs(save_dir, exist_ok=True)

        # Save components
        with open(os.path.join(save_dir, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)

        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(save_dir, 'kmeans.pkl'), 'wb') as f:
            pickle.dump(self.kmeans, f)

        # Save metadata
        metadata = {
            'n_clusters': self.n_clusters,
            'categorical_features': self.categorical_features,
            'cluster_stats': self.cluster_stats
        }

        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✓ Pipeline saved to: {save_dir}")

    @staticmethod
    def load(save_dir):
        """
        Load a fitted pipeline
        """
        preprocessor = TransactionPreprocessor()

        # Load components
        with open(os.path.join(save_dir, 'label_encoders.pkl'), 'rb') as f:
            preprocessor.label_encoders = pickle.load(f)

        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            preprocessor.scaler = pickle.load(f)

        with open(os.path.join(save_dir, 'kmeans.pkl'), 'rb') as f:
            preprocessor.kmeans = pickle.load(f)

        # Load metadata
        with open(os.path.join(save_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            preprocessor.n_clusters = metadata['n_clusters']
            preprocessor.categorical_features = metadata['categorical_features']
            preprocessor.cluster_stats = metadata['cluster_stats']

        preprocessor.is_fitted = True

        print(f"✓ Pipeline loaded from: {save_dir}")
        return preprocessor

    def _calculate_cluster_stats(self, df):
        """Calculate statistics for each cluster"""
        stats = []

        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['Transaction_Pattern_Cluster'] == cluster_id]

            stat = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }

            for col in self.categorical_features:
                mode_val = cluster_data[col].mode()[0]
                mode_pct = (cluster_data[col] == mode_val).sum() / len(cluster_data) * 100
                stat[f'{col}_mode'] = mode_val
                stat[f'{col}_mode_pct'] = mode_pct

            stats.append(stat)

        return pd.DataFrame(stats)

    def _print_cluster_summary(self):
        """Print cluster statistics"""
        print("\nCLUSTER PROFILES:")
        print("-" * 60)

        for _, row in self.cluster_stats.iterrows():
            print(f"\nCluster {int(row['cluster_id'])}: {row['size']:.0f} samples ({row['percentage']:.1f}%)")

            for col in self.categorical_features:
                mode_val = row[f'{col}_mode']
                mode_pct = row[f'{col}_mode_pct']
                print(f"  {col}: {mode_val} ({mode_pct:.1f}%)")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSACTION DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # -----------------------------------------
    # Step 1: Load raw 01_data_cleaning
    # -----------------------------------------
    print("\n[STEP 1] Loading raw 01_data_cleaning...")
    df_raw = pd.read_csv("../../../data/processed/transaction_data_cleaned.csv")
    print(f"Loaded {len(df_raw):,} transactions")

    # -----------------------------------------
    # Step 2: Initialize and fit pipeline
    # -----------------------------------------
    print("\n[STEP 2] Fitting transaction_type_clustering pipeline...")

    preprocessor = TransactionPreprocessor(
        n_clusters=4,  # Adjust based on your analysis
        sample_size=300000,  # For clustering training
        random_state=42
    )

    preprocessor.fit(df_raw)

    # -----------------------------------------
    # Step 3: Transform all 01_data_cleaning
    # -----------------------------------------
    print("\n[STEP 3] Transforming all 01_data_cleaning...")
    df_processed = preprocessor.transform(df_raw)

    # -----------------------------------------
    # Step 4: Save processed 01_data_cleaning for LSTM
    # -----------------------------------------
    print("\n[STEP 4] Saving processed 01_data_cleaning...")

    output_path = "../../../data/processed/transaction_data_for_lstm.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")

    # -----------------------------------------
    # Step 5: Save pipeline for future use
    # -----------------------------------------
    print("\n[STEP 5] Saving pipeline...")

    pipeline_dir = "../../../models/transaction_type_clustering"
    preprocessor.save(pipeline_dir)

    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Data: {output_path}")
    print(f"  - Pipeline: {pipeline_dir}")
    print(f"\nReady for LSTM training!")