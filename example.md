# Detailed Statistical and Machine Learning Analysis

This document presents the complete source code used for the final project, organized into logical sections for clarity and professional review.

## 1. Environment Setup and Data Loading

This section ensures all required libraries are installed and imports them. It then loads and merges the initial datasets from the specified GitHub repository.

### 1.1 Dependencies Installation
```python
# Install necessary libraries
!pip install scipy
!pip install ISLP
!pip install torchinfo
```
### 1.2 Dependencies Installation
```python
# Import core data analysis and visualization libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Import Machine Learning libraries
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Import PyTorch and Deep Learning utilities
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import seed_everything # Used for reproducibility

# Set seed for reproducibility for PyTorch
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)
```
### 1.3 Data Merging
```python
# Define search pattern to find all CSV files recursively in subdirectories
search_pattern = os.path.join('**', '*.csv')
csv_files = glob.glob(search_pattern, recursive=True)

print("Found CSV files:")
print(csv_files)

# Merge the first four CSV files into a single DataFrame
df_merged = pd.DataFrame()
for i in range(4):
    try:
        current_df = pd.read_csv(csv_files[i], sep=";")
        df_merged = pd.concat([df_merged, current_df], ignore_index=True)
    except IndexError:
        print(f"Warning: Only found {len(csv_files)} files. Stopping at index {i}.")
        break
```
