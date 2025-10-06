# train_preprocessing.py

from preprocessing_pipeline import TransactionPreprocessor
import pandas as pd

# Load data
df = pd.read_csv("../../../data/processed/transaction_data_cleaned.csv")

# Fit pipeline
preprocessor = TransactionPreprocessor(n_clusters=4, sample_size=300000)
df_processed = preprocessor.fit_transform(df)

# Save processed data
df_processed.to_csv("../../../data/processed/transaction_data_for_lstm.csv", index=False)

# Save pipeline
preprocessor.save("../../../models/preprocessing")

print("✓ Ready for LSTM training!")