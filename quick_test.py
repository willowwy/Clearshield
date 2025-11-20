"""
Quick test of vuln_scanner.py
No external files needed - creates dummy data
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tests.vuln_scanner import ClearShieldTestSuite

# Create dummy model
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(20, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

# Create dummy data
model = SimpleLSTM()
sample_data = torch.randn(50, 10, 20)
df = pd.DataFrame({
    'transaction_id': range(200),
    'user_id': np.random.randint(1, 30, 200),
    'amount': np.random.uniform(10, 5000, 200),
    'timestamp': pd.date_range('2024-01-01', periods=200),
    'is_fraud': np.random.binomial(1, 0.1, 200)
})

# Run tests
print("Running vulnerability scanner...\n")
test_suite = ClearShieldTestSuite(model, sample_data=sample_data, dataframe=df)
results = test_suite.run_all_tests()
test_suite.print_report()
#test_suite.export_results('results.json')

print("\n✅ Done! Results saved to results.json")
