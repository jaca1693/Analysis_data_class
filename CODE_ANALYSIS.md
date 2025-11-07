# Detailed Statistical and Machine Learning Analysis
**Authors:** Mateo H. Sanchez, Jose A. Calvetty, Milena R. de Sousa

**Dataset:** [Generated Datasets for Burst Detection in Water Distribution Systems](https://github.com/ArieleZanfei/generated-datasets-for-burst-detection-in-water-distribution-systems)

This document presents the complete source code used for the final project, organized into logical sections for clarity and professional review, designed to be executed in a Colab/Jupyter environment.

## 1. Environment Setup and Data Loading

### 1.1 Dependencies Installation and Core Imports
```python
!pip install scipy
# Import core data analysis and visualization libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import os
```
### 1.2 Library Imports and Configuration
All core, machine learning, and deep learning libraries are imported here.
```python
# Import Machine Learning libraries
import sklearn as skl
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR, 
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestRegressor as RF, 
                              GradientBoostingRegressor as GBR) 
from ISLP.bart import BART 
from sklearn.model_selection import train_test_split 

# Import PyTorch and Deep Learning utilities
import torch
from torch import nn
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything # Used for reproducibility

# Set seed for reproducibility for PyTorch and Scikit-learn
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)
```
### 1.3 Data Loading and Merging
The initial datasets are loaded and merged into a single DataFrame (df_merged).
```python
# Define search pattern to find all CSV files recursively in subdirectories
search_pattern = os.path.join('**', '*.csv')
csv_files = glob.glob(search_pattern, recursive=True)

print(csv_files)

# Merge the first four CSV files into a single DataFrame
df_merged = pd.DataFrame()
for i in range(4):
    current_df = pd.read_csv(csv_files[i],sep=";")
    df_merged = pd.concat([df_merged, current_df], ignore_index=True)
```
## 2. Exploratory Data Analysis (EDA) and Statistical Tests
This part performs descriptive statistics, visualizes distributions, and conducts various parametric and non-parametric hypothesis tests on the sensor data.
### 2.1 Descrtiptive Statistics
Calculation of central tendency, dispersion, and shape measures for all sensor columns.
```python
# Create a copy for statistical analysis, dropping target and index columns
df_analy = df_merged.copy()
df_analy = df_analy.drop(columns=['burst','index'])

# Calculate basic descriptive statistics
df_stats = df_analy.describe()

# Calculate specific statistics not included in .describe()
mode = stats.mode(df_analy, keepdims=True)
var = df_analy.var()
std = df_analy.std()
sem = df_analy.sem()
skew = df_analy.skew()
kurtosis = df_analy.kurtosis()

# Format statistics into DataFrames for concatenation
mode = pd.DataFrame(mode.mode.reshape(1, -1), columns=df_analy.columns, index = ['mode']) # Most frequent value
var = pd.DataFrame(var.values.reshape(1, -1), columns=df_analy.columns, index = ['variance']) #Variance
std = pd.DataFrame(std.values.reshape(1, -1), columns=df_analy.columns, index = ['std']) #Standard deviation
sem = pd.DataFrame(sem.values.reshape(1, -1), columns=df_analy.columns, index = ['sem']) #Standard error of the mean
skew = pd.DataFrame(skew.values.reshape(1, -1), columns=df_analy.columns, index = ['skewness']) #Asymmetry of a distribution
kurtosis = pd.DataFrame(kurtosis.values.reshape(1, -1), columns=df_analy.columns, index = ['kurtosis'])

# Concatenate all statistics into one summary table
df_stats = pd.concat([df_stats, mode, var, std, sem, skew, kurtosis]) #Use pd.concat to combine the dataframes
display(df_stats)
```
Descriptive Statistics Table 
|          |    flow_meter1 |    flow_meter2 |   flow_meter3 |    flow_meter4 |         3_press |        12_press |        36_press |         50_press |        60_press |        84_press |        93_press |       112_press |       138_press |       139_press |
|:---------|---------------:|---------------:|--------------:|---------------:|----------------:|----------------:|----------------:|-----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|
| count    | 140160         | 140160         | 140160        | 140160         | 140160          | 140160          | 140160          | 140160           | 140160          | 140160          | 140160          | 140160          | 140160          | 140160          |
| mean     |     31.4583    |     32.4345    |     89.5173   |     27.7136    |     32.6305     |     32.134      |     35.9364     |     39.0964      |     33.6549     |     32.5922     |     31.5185     |     34.9086     |     35.163      |     31.3232     |
| std      |      8.9716    |     12.2473    |     41.6546   |      8.62795   |      2.07409    |      1.8892     |      2.50653    |      0.055288    |      2.57513    |      2.55157    |      2.50647    |      2.21456    |      2.07334    |      1.52728    |
| min      |     16.89      |     -2.0516    |     13.616    |     13.434     |     19.892      |     20.329      |     19.878      |     38.759       |     17.371      |     16.507      |     16.293      |     21.331      |     22.119      |     21.863      |
| 25%      |     22.733     |     20.7995    |     51.678    |     19.263     |     31.236      |     30.87       |     34.291      |     39.06        |     31.95       |     30.908      |     29.8498     |     33.457      |     33.795      |     30.302      |
| 50%      |     32.994     |     34.573     |     99.085    |     29.0545    |     32.628      |     32.129      |     35.935      |     39.096       |     33.651      |     32.588      |     31.454      |     34.883      |     35.16       |     31.332      |
| 75%      |     38.008     |     41.5203    |    120.53     |     34.151     |     34.68       |     34.009      |     38.374      |     39.15        |     36.184      |     35.096      |     33.988      |     37.077      |     37.192      |     32.815      |
| max      |     75.621     |    241.22      |    231.02     |     61.91      |     35.489      |     34.719      |     39.262      |     39.169       |     37.054      |     35.944      |     34.985      |     37.777      |     37.858      |     33.404      |
| mode     |     19.026     |     15.194     |    110.31     |     16.446     |     35.218      |     34.5        |     38.992      |     39.164       |     36.78       |     35.772      |     34.588      |     37.625      |     37.769      |     33.214      |
| var      |     80.4895    |    149.997     |   1735.1      |     74.4415    |      4.30185    |      3.56907    |      6.2827     |      0.00305676  |      6.63128    |      6.51052    |      6.28238    |      4.90427    |      4.29872    |      2.3326     |
| std      |      8.9716    |     12.2473    |     41.6546   |      8.62795   |      2.07409    |      1.8892     |      2.50653    |      0.055288    |      2.57513    |      2.55157    |      2.50647    |      2.21456    |      2.07334    |      1.52728    |
| sem      |      0.0239639 |      0.0327137 |      0.111263 |      0.023046  |      0.00554008 |      0.00504621 |      0.00669516 |      0.000147679 |      0.00687839 |      0.00681547 |      0.00669499 |      0.00591527 |      0.00553806 |      0.00407951 |
| skew     |     -0.0117617 |     -0.0284494 |     -0.230735 |     -0.0343642 |     -0.456349   |     -0.487605   |     -0.525518   |     -0.489277    |     -0.511187   |     -0.516275   |     -0.452406   |     -0.508267   |     -0.512303   |     -0.48636    |
| kurtosis |     -0.96428   |     -0.299599  |     -1.05591  |     -1.00815   |     -0.377178   |     -0.247688   |     -0.190024   |     -0.288931    |     -0.264105   |     -0.251045   |     -0.348022   |     -0.249646   |     -0.254266   |     -0.313507   |

### 2.2 Parametric Tests (T-tests and ANOVA)
Performing independent t-tests for pairwise comparisons and ANOVA for multiple group comparisons.
```python
# Define lists of 'flow_meter' and 'press' column names
flow_meter_cols = [col for col in df_analy.columns if 'flow_meter' in col]
press_cols = [col for col in df_analy.columns if '_press' in col]

# Convert the groups to lists of NumPy arrays for ANOVA
flow_meter_groups = [df_analy[col].values for col in flow_meter_cols]
press_groups = [df_analy[col].values for col in press_cols]

print("\n--- Independent T-tests between pairs of 'flow_meter' columns ---")
for i in range(len(flow_meter_cols)):
    for j in range(i + 1, len(flow_meter_cols)):
        col1 = flow_meter_cols[i]
        col2 = flow_meter_cols[j]
        ttest_result = stats.ttest_ind(df_analy[col1], df_analy[col2])
        print(f"T-test between {col1} and {col2}: {ttest_result}")

print("\n--- ANOVA Tests ---")
# Perform one-way ANOVA test on 'flow_meter' columns
anova_flow_meters = stats.f_oneway(*flow_meter_groups)
print(f"ANOVA test for 'flow_meter' columns: F-statistic: {anova_flow_meters.statistic}, P-value: {anova_flow_meters.pvalue}")

# Perform one-way ANOVA test on 'press' columns
anova_presses = stats.f_oneway(*press_groups)
print(f"ANOVA test for 'press' columns: F-statistic: {anova_presses.statistic}, P-value: {anova_presses.pvalue}")
```
```text
Independent t-tests between pairs of 'flow_meter' columns:
T-test between flow_meter1 and flow_meter2: TtestResult(statistic=np.float64(-24.07262472891008), pvalue=np.float64(6.532477176455693e-128), df=np.float64(280318.0))
T-test between flow_meter1 and flow_meter3: TtestResult(statistic=np.float64(-510.12016586640374), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between flow_meter1 and flow_meter4: TtestResult(statistic=np.float64(112.63164267721473), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between flow_meter2 and flow_meter3: TtestResult(statistic=np.float64(-492.2097857928514), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between flow_meter2 and flow_meter4: TtestResult(statistic=np.float64(117.97431020888729), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between flow_meter3 and flow_meter4: TtestResult(statistic=np.float64(543.928746301255), pvalue=np.float64(0.0), df=np.float64(280318.0))

Independent t-tests between pairs of 'press' columns:
T-test between 3_press and 12_press: TtestResult(statistic=np.float64(66.24865369400028), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 3_press and 36_press: TtestResult(statistic=np.float64(-380.4261480169346), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 3_press and 50_press: TtestResult(statistic=np.float64(-1166.7031109605657), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 3_press and 60_press: TtestResult(statistic=np.float64(-115.98736241780686), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 3_press and 84_press: TtestResult(statistic=np.float64(4.356498827275517), pvalue=np.float64(1.3220625473487372e-05), df=np.float64(280318.0))
T-test between 3_press and 93_press: TtestResult(statistic=np.float64(127.95744315401873), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 3_press and 112_press: TtestResult(statistic=np.float64(-281.0982798225494), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 3_press and 138_press: TtestResult(statistic=np.float64(-323.30498237752863), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 3_press and 139_press: TtestResult(statistic=np.float64(190.0015537761829), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 36_press: TtestResult(statistic=np.float64(-453.53577856032075), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 50_press: TtestResult(statistic=np.float64(-1379.1325825501578), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 60_press: TtestResult(statistic=np.float64(-178.27588012925182), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 84_press: TtestResult(statistic=np.float64(-54.030020655292866), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 93_press: TtestResult(statistic=np.float64(73.41495591336735), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 112_press: TtestResult(statistic=np.float64(-356.8514053233418), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 138_press: TtestResult(statistic=np.float64(-404.2875628005277), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 12_press and 139_press: TtestResult(statistic=np.float64(124.94511729352874), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 36_press and 50_press: TtestResult(statistic=np.float64(-471.8649443258594), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 36_press and 60_press: TtestResult(statistic=np.float64(237.68891104587354), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 36_press and 84_press: TtestResult(statistic=np.float64(350.0373132524156), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 36_press and 93_press: TtestResult(statistic=np.float64(466.5986569591143), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 36_press and 112_press: TtestResult(statistic=np.float64(115.04149800700236), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 36_press and 138_press: TtestResult(statistic=np.float64(89.00558856804956), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 36_press and 139_press: TtestResult(statistic=np.float64(588.4028334794855), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 50_press and 60_press: TtestResult(statistic=np.float64(790.9211868798418), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 50_press and 84_press: TtestResult(statistic=np.float64(954.1021093720425), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 50_press and 93_press: TtestResult(statistic=np.float64(1131.5953348711228), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 50_press and 112_press: TtestResult(statistic=np.float64(707.7361522686425), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 50_press and 138_press: TtestResult(statistic=np.float64(709.9844475723189), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 50_press and 139_press: TtestResult(statistic=np.float64(1904.1623761842047), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 60_press and 84_press: TtestResult(statistic=np.float64(109.74421923750175), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 60_press and 93_press: TtestResult(statistic=np.float64(222.56627028063303), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 60_press and 112_press: TtestResult(statistic=np.float64(-138.19954001946604), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 60_press and 138_press: TtestResult(statistic=np.float64(-170.7873463564681), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 60_press and 139_press: TtestResult(statistic=np.float64(291.55588731955174), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 84_press and 93_press: TtestResult(statistic=np.float64(112.38354869593984), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 84_press and 112_press: TtestResult(statistic=np.float64(-256.6824811264855), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 84_press and 138_press: TtestResult(statistic=np.float64(-292.74593075995847), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 84_press and 139_press: TtestResult(statistic=np.float64(159.75510703485566), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 93_press and 112_press: TtestResult(statistic=np.float64(-379.4681629422728), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 93_press and 138_press: TtestResult(statistic=np.float64(-419.45765460856904), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 93_press and 139_press: TtestResult(statistic=np.float64(24.907203537628725), pvalue=np.float64(8.7674111254035e-137), df=np.float64(280318.0))
T-test between 112_press and 138_press: TtestResult(statistic=np.float64(-31.39831100119615), pvalue=np.float64(5.075241353653009e-216), df=np.float64(280318.0))
T-test between 112_press and 139_press: TtestResult(statistic=np.float64(498.9669160241004), pvalue=np.float64(0.0), df=np.float64(280318.0))
T-test between 138_press and 139_press: TtestResult(statistic=np.float64(558.2398853795611), pvalue=np.float64(0.0), df=np.float64(280318.0))
```
```text
ANOVA test for 'flow_meter' columns:
F-statistic: 240152.20061594024
P-value: 0.0

ANOVA test for 'press' columns:
F-statistic: 181474.47539419503
P-value: 0.0
```
### 2.3 Non-Parametric Tests (Normality, Homogeneity, Kruskal-Wallis)
Checking the assumptions for parametric tests and using Kruskal-Wallis as an alternative.
```python
print("\n--- Normality and Homogeneity Checks (Shapiro-Wilk and Levene) ---")
# Check normality for 'flow_meter' columns
print("Normality test (Shapiro-Wilk) for 'flow_meter' columns:")
for col in flow_meter_cols:
    shapiro_test = stats.shapiro(df_analy[col])
    print(f"Shapiro-Wilk test for {col}: {shapiro_test}")

# Check normality for 'press' columns
print("\nNormality test (Shapiro-Wilk) for 'press' columns:")
for col in press_cols:
    shapiro_test = stats.shapiro(df_analy[col])
    print(f"Shapiro-Wilk test for {col}: {shapiro_test}")

# Check homogeneity of variances for 'flow_meter' columns
print("\nHomogeneity of variances test (Levene) for 'flow_meter' columns:")
levene_flow_meters = stats.levene(*flow_meter_groups)
print(f"Levene test for 'flow_meter' columns: {levene_flow_meters}")

# Check homogeneity of variances for 'press' columns
print("\nHomogeneity of variances test (Levene) for 'press' columns:")
levene_presses = stats.levene(*press_groups)
print(f"Levene test for 'press' columns: {levene_presses}")

print("\n--- Kruskal-Wallis H-test ---")
# Perform Kruskal-Wallis test on 'flow_meter' columns
kruskal_flow_meters = stats.kruskal(*flow_meter_groups)
print(f"Kruskal-Wallis test for 'flow_meter' columns: Statistic: {kruskal_flow_meters.statistic}, P-value: {kruskal_flow_meters.pvalue}")

# Perform Kruskal-Wallis test on 'press' columns
kruskal_presses = stats.kruskal(*press_groups)
print(f"Kruskal-Wallis test for 'press' columns: Statistic: {kruskal_presses.statistic}, P-value: {kruskal_presses.pvalue}")
```
```text
Normality test (Shapiro-Wilk) for 'flow_meter' columns:
Shapiro-Wilk test for flow_meter1: ShapiroResult(statistic=np.float64(0.9493228975606264), pvalue=np.float64(4.9171084527515064e-101))
Shapiro-Wilk test for flow_meter2: ShapiroResult(statistic=np.float64(0.9478726513632699), pvalue=np.float64(8.99703303860943e-102))
Shapiro-Wilk test for flow_meter3: ShapiroResult(statistic=np.float64(0.9404495102073275), pvalue=np.float64(2.7342723606498036e-105))
Shapiro-Wilk test for flow_meter4: ShapiroResult(statistic=np.float64(0.9487620409763547), pvalue=np.float64(2.5369035572336675e-101))

Normality test (Shapiro-Wilk) for 'press' columns:
Shapiro-Wilk test for 3_press: ShapiroResult(statistic=np.float64(0.946357893014183), pvalue=np.float64(1.5939649288714802e-102))
Shapiro-Wilk test for 12_press: ShapiroResult(statistic=np.float64(0.9454502125805722), pvalue=np.float64(5.766148434067288e-103))
Shapiro-Wilk test for 36_press: ShapiroResult(statistic=np.float64(0.9425084201104278), pvalue=np.float64(2.3568159106545237e-104))
Shapiro-Wilk test for 50_press: ShapiroResult(statistic=np.float64(0.9418686068432865), pvalue=np.float64(1.1982244630685571e-104))
Shapiro-Wilk test for 60_press: ShapiroResult(statistic=np.float64(0.9402617296536535), pvalue=np.float64(2.253878248604768e-105))
Shapiro-Wilk test for 84_press: ShapiroResult(statistic=np.float64(0.9399330371865104), pvalue=np.float64(1.6091743734511091e-105))
Shapiro-Wilk test for 93_press: ShapiroResult(statistic=np.float64(0.9462561012594131), pvalue=np.float64(1.4211315099164976e-102))
Shapiro-Wilk test for 112_press: ShapiroResult(statistic=np.float64(0.9402854001259934), pvalue=np.float64(2.3093785778318252e-105))
Shapiro-Wilk test for 138_press: ShapiroResult(statistic=np.float64(0.9402604235224317), pvalue=np.float64(2.2508554313729875e-105))
Shapiro-Wilk test for 139_press: ShapiroResult(statistic=np.float64(0.944536100250562), pvalue=np.float64(2.1017266138332891e-103))

Homogeneity of variances test (Levene) for 'flow_meter' columns:
Levene test for 'flow_meter' columns: LeveneResult(statistic=np.float64(136660.87043572907), pvalue=np.float64(0.0))

Homogeneity of variances test (Levene) for 'press' columns:
Levene test for 'press' columns: LeveneResult(statistic=np.float64(39350.37973802515), pvalue=np.float64(0.0))
/usr/local/lib/python3.12/dist-packages/scipy/stats/_axis_nan_policy.py:579: UserWarning: scipy.stats.shapiro: For N > 5000, computed p-value may not be accurate. Current N is 140160.
  res = hypotest_fun_out(*samples, **kwds)
```
```text
Kruskal-Wallis test for 'flow_meter' columns:
Kruskal-Wallis statistic: 183729.10612872834
P-value: 0.0

Kruskal-Wallis test for 'press' columns:
Kruskal-Wallis statistic: 714818.3031763351
P-value: 0.0
```
### 2.4 Non-parametric Tests (Mann-Whitney U)
Using Mann-Whitney U tests for pairwise comparisons when Kruskal-Wallis is significant.
```python
# Perform Mann-Whitney U tests between pairs of 'flow_meter' columns
print("\nMann-Whitney U tests between pairs of 'flow_meter' columns:")
for i in range(len(flow_meter_cols)):
    for j in range(i + 1, len(flow_meter_cols)):
        col1 = flow_meter_cols[i]
        col2 = flow_meter_cols[j]
        mannwhitneyu_result = stats.mannwhitneyu(df_analy[col1], df_analy[col2])
        print(f"Mann-Whitney U test between {col1} and {col2}: {mannwhitneyu_result}")

# Perform Mann-Whitney U tests between pairs of 'press' columns
print("\nMann-Whitney U tests between pairs of 'press' columns:")
for i in range(len(press_cols)):
    for j in range(i + 1, len(press_cols)):
        col1 = press_cols[i]
        col2 = press_cols[j]
        mannwhitneyu_result = stats.mannwhitneyu(df_analy[col1], df_analy[col2])
        print(f"Mann-Whitney U test between {col1} and {col2}: {mannwhitneyu_result}")
```
```text
Mann-Whitney U tests between pairs of 'flow_meter' columns:
Mann-Whitney U test between flow_meter1 and flow_meter2: MannwhitneyuResult(statistic=np.float64(9286826115.0), pvalue=np.float64(5.867493672775926e-138))
Mann-Whitney U test between flow_meter1 and flow_meter3: MannwhitneyuResult(statistic=np.float64(2689480268.5), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter1 and flow_meter4: MannwhitneyuResult(statistic=np.float64(12302332901.0), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter2 and flow_meter3: MannwhitneyuResult(statistic=np.float64(2797445947.5), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter2 and flow_meter4: MannwhitneyuResult(statistic=np.float64(12217371670.5), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter3 and flow_meter4: MannwhitneyuResult(statistic=np.float64(17448646105.0), pvalue=np.float64(0.0))

Mann-Whitney U tests between pairs of 'press' columns:
Mann-Whitney U test between 3_press and 12_press: MannwhitneyuResult(statistic=np.float64(11440737079.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 36_press: MannwhitneyuResult(statistic=np.float64(3160811819.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 50_press: MannwhitneyuResult(statistic=np.float64(0.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 60_press: MannwhitneyuResult(statistic=np.float64(7280732240.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 84_press: MannwhitneyuResult(statistic=np.float64(9466201202.0), pvalue=np.float64(4.350511392349533e-62))
Mann-Whitney U test between 3_press and 93_press: MannwhitneyuResult(statistic=np.float64(12540905752.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 112_press: MannwhitneyuResult(statistic=np.float64(4657881152.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 138_press: MannwhitneyuResult(statistic=np.float64(4023256355.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 139_press: MannwhitneyuResult(statistic=np.float64(13532269223.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 36_press: MannwhitneyuResult(statistic=np.float64(2282433199.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 50_press: MannwhitneyuResult(statistic=np.float64(0.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 60_press: MannwhitneyuResult(statistic=np.float64(6277447430.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 84_press: MannwhitneyuResult(statistic=np.float64(8368813478.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 93_press: MannwhitneyuResult(statistic=np.float64(10921125670.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 112_press: MannwhitneyuResult(statistic=np.float64(3501198295.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 138_press: MannwhitneyuResult(statistic=np.float64(2863681338.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 139_press: MannwhitneyuResult(statistic=np.float64(12485608351.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 50_press: MannwhitneyuResult(statistic=np.float64(1153984984.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 60_press: MannwhitneyuResult(statistic=np.float64(14316349494.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 84_press: MannwhitneyuResult(statistic=np.float64(16023508198.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 93_press: MannwhitneyuResult(statistic=np.float64(17533257645.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 112_press: MannwhitneyuResult(statistic=np.float64(12371959199.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 138_press: MannwhitneyuResult(statistic=np.float64(11911220002.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 139_press: MannwhitneyuResult(statistic=np.float64(18410570123.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 60_press: MannwhitneyuResult(statistic=np.float64(19644825600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 84_press: MannwhitneyuResult(statistic=np.float64(19644825600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 93_press: MannwhitneyuResult(statistic=np.float64(19644825600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 112_press: MannwhitneyuResult(statistic=np.float64(19644825600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 138_press: MannwhitneyuResult(statistic=np.float64(19644825600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 139_press: MannwhitneyuResult(statistic=np.float64(19644825600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 84_press: MannwhitneyuResult(statistic=np.float64(12281445532.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 93_press: MannwhitneyuResult(statistic=np.float64(14108920678.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 112_press: MannwhitneyuResult(statistic=np.float64(6881723445.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 138_press: MannwhitneyuResult(statistic=np.float64(6336216131.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 139_press: MannwhitneyuResult(statistic=np.float64(15244915056.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 93_press: MannwhitneyuResult(statistic=np.float64(12358277065.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 112_press: MannwhitneyuResult(statistic=np.float64(5047167831.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 138_press: MannwhitneyuResult(statistic=np.float64(4528643839.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 139_press: MannwhitneyuResult(statistic=np.float64(13037704041.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 93_press and 112_press: MannwhitneyuResult(statistic=np.float64(3250786903.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 93_press and 138_press: MannwhitneyuResult(statistic=np.float64(2709445872.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 93_press and 139_press: MannwhitneyuResult(statistic=np.float64(10612740477.0), pvalue=np.float64(5.94744152632351e-298))
Mann-Whitney U test between 112_press and 138_press: MannwhitneyuResult(statistic=np.float64(9014115633.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 112_press and 139_press: MannwhitneyuResult(statistic=np.float64(17777067066.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 138_press and 139_press: MannwhitneyuResult(statistic=np.float64(18242678766.5), pvalue=np.float64(0.0))
```
### 2.5 Correlation and Distribution Analysis
Calculating and visualizing the Pearson and Spearman correlation matrices, and the probability distributions.
```python
# Calculate and visualize the Pearson correlation matrix
correlation_matrix = df_analy.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Matrix of DataFrame Columns')
plt.show()

# Calculate and visualize the Spearman correlation matrix
spearman_correlation_matrix = df_analy.corr(method='spearman')
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Matrix of DataFrame Columns')
plt.show()

# Visualize the probability distribution of each column (Histograms)
for col in df_analy.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df_analy[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
```
![Texto Alternativo](images/Correlation_Matrix_of_DataFrame_Columns.png)
![Texto Alternativo](images/spearman_correlation_matrix.png)
## Summary:

### Key Findings

* Independent t-tests between pairs of 'flow_meter' columns and pairs of 'press' columns showed statistically significant differences between the means of most column pairs, with many p-values close to 0.
* ANOVA tests for both 'flow_meter' and 'press' columns resulted in high F-statistics and p-values of 0.0, indicating statistically significant differences in the means among the groups within each category.
* Assumptions for ANOVA (normality and homogeneity of variances) were violated for both 'flow_meter' and 'press' columns based on Shapiro-Wilk tests (low p-values) and Levene's tests (p-values of 0.0).
* Kruskal-Wallis tests, used as non-parametric alternatives due to ANOVA assumption violations, produced large test statistics and p-values of 0.0 for both 'flow_meter' and 'press' columns, confirming statistically significant differences in median values among the groups.
* A correlation matrix was calculated and visualized, showing the pairwise correlations between all columns in the DataFrame.
* Histograms with kernel density estimates were generated for each column, providing a visual representation of their probability distributions.

## 3 Machine Learning Classification Models
This section implements and evaluates three different classification approaches to predict the burst target variable.
### 3.1 Decision Tree Classification (DTC)
Configuring the columns to be used as features. First, .fit defines the columns to be used taking into account that the intercept term will not be added. Second, .transform applies the learned transformation to the dataframe.
```python
# Prepare data for modeling
model = MS(df_merged.columns.drop(['index','burst']), intercept=False)
D = model.fit_transform(df_merged)
feature_names = list(D.columns)
X = np.asarray(D)
y = df_merged['burst']

# Split data into training and testing sets (random_state=42)
(X_train,
X_test,
y_train,
y_test) = skm.train_test_split(X,
                               y,
                               test_size=0.3,
                               random_state=42)
```
Choosen parameters to the single decission tree calssifier:

- entropy - for function to measure the quality of a split.
- max_depth - for determining the maximum depth of the three.
- random_state - for controling the randomnes of the estimator. 

```python
# Train the initial classifier (max_depth=3)
classificador = DTC(criterion='entropy',
                    max_depth=3,
                    random_state=0)

print(f"X shape: {X.shape}")
print(f"y_train shape: {y_train.shape}")

classificador.fit(X_train, y_train)
```
Evaluation metrics:
For showing the accuracy of the classifier
```python
print(f"Initial Test Accuracy: {accuracy_score(y_test, classificador.predict(X_test))}")
```
For calculating the logarithmic loss (also known as cross-entropy loss) of the decission tree classifier ont he test set. log_loss is a common metric for evaluating the performance of the classification models that output probabilities. The classificador.predict_proba(x_test) part uses the the trained DTC to predict the probability of each class, in this case 0 or 1, for each instance in the test data, returning an array where each row conrresponds to adata point and each column the probability of belonging to a specific class.
```phyton
resid_dev = np.sum(log_loss(y_test, classificador.predict_proba(X_test)))
print(f"Initial Log Loss: {resid_dev}")
```
Plotting the decission tree classifier.
```phyton
# Visualize the initial tree
plt.figure(figsize=(12,12))
plot_tree(classificador,
          feature_names=feature_names,
          filled=True)
plt.title("Initial Decision Tree (Max Depth 3)")
plt.show()

# Print text representation of the classifier
print("\n--- Initial Decision Tree Text Summary ---")
print(export_text(classificador,
                  feature_names=feature_names,
                  show_weights=True))
```
Performing corss-validation to evaluate the performance of the classifier on the training data. skm.ShuffleSplit creates a ´ShuffleSplit´ object which is a cross-validation strategy.
Parameters of ´ShuffleSplit´ object:
- n_splits - for specifing the number of different training/testing sets that the data will be randomly shuffled and split.
- test_size - for setting the size of the test set for each split.
- random_state - for controling the randomnes of the shuffling and splitting.
skm.cross_validate perfomrs the cross-validation returning a dictionary containing various scores for each split. The default score for a classifier is accuracy
```python
# Cross-Validation
validation = skm.ShuffleSplit(n_splits=4,
                              test_size=100,
                              random_state=0)
results = skm.cross_validate(classificador,
                             X_train,
                             y_train,
                             cv=validation)
print(f"\nCross-Validation Scores: {results['test_score']}")

# Creating a confussion matrix for the classifier
confusion_matrix = confusion_table(classificador.predict(X),df_merged['burst'])
print(confusion_matrix)
```
Applying Random Forest
Here we create a new instance of the decission tree classifier. The chosen parameters are:
- entropy - for function to measure the quality of a split.
- random_state - for controling the randomnes of the estimator.
It is important to mention that in this case the parameter max_depth is not specified here, meaning the tree can grow to its full depth.
.fit trains the newly classifier using the training data and their corresponding labels, learning the decision rules from this data.
The accuracy of the model enchanced due to the classifier was trained with a potentially deeper decision tree than the previus one.
```python
# Unpruned decision tree
clf = DTC(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))

ax = subplots(figsize=(12,12))[1]
plot_tree(clf,
          feature_names=feature_names,
          ax=ax);
plt.show()
```
Preparing for cost complexity pruning of the decision tree and setting up a cross-validation strategy for selecting the optimal pruning parameter.
Here the classifier will have a the same parameters of the previous classifier, but with a max_depth equal to 10.
clf.cost_complexity_pruning_path calculates the ´ccp_alpha´ values, that results in a sequence of pruned trees from the newly classifier. It retunrs two arrays: the effective alphas (´ccp_alpha´) and the total imputirity of the leaves for each alpha (´impurities´). The training data set is used to calculate these paths.
skm.fold creates a ´KFold´ cross-validation object. The chosen parameters are:
- 10 - specifies that the training data will be split into 10 folds.
- random_state - for controling the randomnes of the shuffling and splitting.
- shuffle - True for shuffling the data before splitting into folds.
```python
# Cost-Complexity Pruning (CCP) and Grid Search
clf = DTC(criterion='entropy', max_depth = 10,random_state=0) 
ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(10,
                  random_state=1,
                  shuffle=True)
````
Using skm.GridSearchCV to find the optimal ´cc_alpha´ value. It creates a ´GridSearchCV´ object that works with multiple parameter combinations for determining which combination gives the best performance. The chosen parameters are:
-clf - the decision tree classifier
-ccp_path.ccp_alphas - It defines the grid of hyperpatameters to search over. The only hyperparameter being tune is ´cpp_alpha´, and the values to try are the effective alphas calculated in ´ccp_path´
- refit - True for refit rhe model using the entire training dataset with the best alpha after finding it.
- cv - for specifing the cross-validation strategy to use during the grid search. In this case, we used the ´KFold´ object.
- socring - for specifing the metric used to evaluate the performance of each pruned tree during cross-validation.
.fit starts the grid search process. ´GridSearchCV´ object iterates through each ´cpp_alpha´ value in the grid, performs the KFold cross validation and keeps track, of which ´ccp_alpha´ results in the highest average accuracy across the folds.
grid.best_score_ accesses the ´best_score_´ attribute of the fitted ´GridSearchCV´ object, which is the mean cross-validated accuracy of the best performing model found during the grid search. #agregar cambiar en la linea de de codigo de abajo "classificador" por "clf".
```python
grid = skm.GridSearchCV(clf,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True,
                        cv=kfold,
                        scoring='accuracy')
grid.fit(X_train, y_train)
grid.best_score_

# For plotting the best Pruned Decision Tree
ax = subplots(figsize=(12, 12))[1]
best_ = grid.best_estimator_
plot_tree(best_,
          feature_names=feature_names,
          ax=ax);

# Final evaluation of the best (pruned) tree
# Showing the number of leaves
print(best_.tree_.n_leaves)

#Printing the accuracy score and confusion matrix
print(accuracy_score(y_test,
                      best_.predict(X_test)))
confusion = confusion_table(best_.predict(X_test),
                            y_test)
print(confusion)
```
```text
Data Shape after Splitting:
(140160, 14)
(98112,)

Initial Test Accuracy:
0.9489630898021308

Residual Deviance (Log Loss):
0.19813065395600002
```
Initial Desicion Tree Visualization
![Texto Alternativo](images/initial_decision_tree.png)
```text
Decision Tree Logic (Text Export)
|--- flow_meter3 <= 29.26
|   |--- 139_press <= 33.11
|   |   |--- weights: [0.00, 101.00] class: 1.0
|   |--- 139_press >  33.11
|   |   |--- flow_meter1 <= 19.54
|   |   |   |--- weights: [13341.00, 105.00] class: 0.0
|   |   |--- flow_meter1 >  19.54
|   |   |   |--- weights: [1.00, 43.00] class: 1.0
|--- flow_meter3 >  29.26
|   |--- flow_meter2 <= 14.81
|   |   |--- 12_press <= 34.46
|   |   |   |--- weights: [4.00, 4.00] class: 0.0
|   |   |--- 12_press >  34.46
|   |   |   |--- weights: [0.00, 165.00] class: 1.0
|   |--- flow_meter2 >  14.81
|   |   |--- flow_meter1 <= 18.73
|   |   |   |--- weights: [1.00, 65.00] class: 1.0
|   |   |--- flow_meter1 >  18.73
|   |   |   |--- weights: [79289.00, 4993.00] class: 0.0
```
```text
Initial Cross-Validation Scores
array([0.93, 0.94, 0.92, 0.92])

Confusion MAtrix (Full Dataset)
Truth         0.0   1.0
Predicted              
0.0        132395  7248
1.0            16   501

Accuracy Score (clf - Unpruned Tree)
0.9743388508371386
```
Pruned decision tree
![Texto Alternativo](images/unpruned_decision_tree.png)
```text
Best Cross-Validation Score (After Pruning)
0.9547150614614586
```
Pruned decision tree
![Texto Alternativo](images/pruned_decision_tree.png)
```text
Number of Leaves (Complexity)
73

Final Test Accuracy (Pruned Tree)
0.9554794520547946

Confusion Matrix (Pruned Tree - Test Set)
Truth      0.0  1.0
Predicted          
0.0      39724  1827
1.0         45  452
```
### 3.2 Naive Bayes Classification (Gaussian)
Here we create a ´GaussainNB´ classifier that implements the Gaussian Naive Bayes algorithm.
.fit trains the newly classifier using the training data and their corresponding labels. The model learns the parameters of the Gaussian distributions for each feature within each class. 
```python
gnb = GaussianNB()
model_gnb = gnb.fit(X_train, y_train)

# For showing some characteristics of the classifier.
display(model_gnb.classes_) #for seeing the classes

display(model_gnb.class_prior_) #for seeing the prior probabilites stored

display(model_gnb.theta_) #parameter of the features
display(model_gnb.var_) #parameter of the features

# For showing the values of the parameters in each attribute.
display(X_train[y_train == 0].mean()) #for know the value of theta in each attr
display(X_train[y_train == 0].var(ddof=0)) #for know the value of variance in each attr

# Using the model for predicting values and print the values of accuracy, confusion matrix, and the number of mislabeled points.
y_pred = model_gnb.predict(X_test) #predicted values

print(accuracy_score(y_test,
                     y_pred))
display(confusion_table(y_pred, y_test)) #confussion table

#print the number of mislabeled points out of a total
miss_points = (X_test.shape[0], (y_test != y_pred).sum())
print("Mislabeled points out of a total %d points : %d" % miss_points)

y_pred_prob = model_gnb.predict_proba(X_test)[:10]
display(y_pred_prob) # Probability that each observation belongs to a particular class
```
```text




Features Means (Tetha) per Class
array([0., 1.])
array([0.94424739, 0.05575261])
array([[31.41777321, 32.35433502, 89.37671911, 27.67738044, 32.64250801,
        32.14845015, 35.94668752, 39.09659606, 33.67089768, 32.6074463 ,
        31.53093826, 34.91806978, 35.17227896, 31.33433346],
       [32.88619378, 34.70198464, 95.43207971, 29.03543382, 32.26268702,
        31.73759104, 35.5611234 , 39.08813108, 33.17585941, 32.12583108,
        31.0982106 , 34.56394369, 34.83978154, 31.02193565]])

Features Variances (Var) per Class
array([[7.98673996e+01, 1.48309713e+02, 1.73216800e+03, 7.40091555e+01,
        4.26122500e+00, 3.51129461e+00, 6.22686405e+00, 3.03961468e-03,
        6.56834615e+00, 6.44870415e+00, 6.22825096e+00, 4.87144114e+00,
        4.26850968e+00, 2.30324443e+00],
       [8.79416043e+01, 1.60610280e+02, 1.71379348e+03, 7.86672563e+01,
        4.91460082e+00, 4.46280875e+00, 7.21851135e+00, 3.38485392e-03,
        7.58533408e+00, 7.48170790e+00, 7.14775527e+00, 5.47738664e+00,
        4.80615918e+00, 2.74050384e+00]])

Mean of All Features for Class 0 (Combined)
37.13531528279367
Variance of All Features for Class 0 (Combined)
364.98982381380495
```
```text
Test Accuracy Score (GaussianNB)
0.8529299847792998

Confusion Matrix (GaussianNB - Test Set)
Truth      0.0   1.0
Predicted           
0.0      35509  1924
1.0       4260   355

Mislabeled points out of a total 42048 points : 6184
```
```text
Predicted Probabilities (For 10 Samples)
array([[0.87674626, 0.12325374],
       [0.91803406, 0.08196594],
       [0.98578978, 0.01421022],
       [0.99552902, 0.00447098],
       [0.94307491, 0.05692509],
       [0.93278848, 0.06721152],
       [0.98943922, 0.01056078],
       [0.869928  , 0.130072  ],
       [0.00324177, 0.99675823],
       [0.65773165, 0.34226835]])
```
### 3.3 Neural Networks (Scikit-learn MLP Classifier)
The ´torchinfo´ library is used to get a summary of a ´PyTorch´ model, which is helpful for understanding the model's arquitecture and paremeter count.
Preparing the data to be used with a neural network model built with PyTorch.
torch.tensor converts the training features from NumPy array to PyTorch tensor. The .unsqueeze(1) method adds an extra dimension to the tensor, often necessary for compatibility with certain PyTorch loss funcions or model architechtures that expect rhe target to have a specific shape.
TensorDataset creates a ´TensorDataset´ object, that wraps the features and label tensors together, making it easy to access corresponding samples.
DataLoader creates a ´DataLoader´ for iterating over datasets in batches. It uses the ´TensorDataset´ object, and the chosen parameters are:
- batch_size - number of samples to include in each batch
- shuffle - True for shuffling the data in the training set at the beginning of eaach epoch.

```python
# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # Add an extra dimension for binary output
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1) # Add an extra dimension for binary output

# Create TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoader
batch_size = 64 # You can adjust this batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data prepared for PyTorch model.")

print(X_train_tensor.shape)
```
Here we defined and traind a Multi-Layer Perceptron classifier.
MLPClassifier creates an instance of the Multi-Layer Perceptron classifier. The chosen parameters are:
- solver - It specifies the optimization algorithm to use for training the neural network. ´sdg´ stands for Stochastic Gradient Descent.
- learning_rate_init - It sets the initial learning rate for the optimizar.
- max_iter - It defines the maximum number of epochs, which are the iterarions over the entire training dataset, to train the model.
- shuffle - ´True´ indicates that the training data will be shuffled at each epoch.
-tol - It sets the tolerance for the optimization. If the loss does not improve by at least this amount for ´n_iter_no_change´ consecutive epochs, training will stop.
- hidden_layer_size - It defines the architecture of the hidden layers. In this case, there arte tow hidden lates, each with 100 neurons.
- batch_size - It defines the size of minibatches for stochastic optimizers. In this case we take the the value of the variable created previusly.
- alpha - This is the L2 penalty (regularization term) parameter that helps prevent overfitting.
- random_state - For controling the randomnes.
MLPmodel.fit trains the ´MLPClassifier´ model using the training data ´X_train_tensor´ and their corresponding labels ´y_train_tensor´. ´MLPClassifier´ is a model form scikit-learn that are designed to work directly with NumPy arrays or pandas DataFrames, and it handles batching and shuffling internally if needed by the specific algorithm. Then that is the reason of using directly the ´X_train_tensor´ and ´y_train_tensor´.

```python
# model
MLPmodel = MLPClassifier(solver='sgd',
                         learning_rate_init=1e-4,
                         max_iter=100,
                         shuffle=True,
                         tol = 1e-8,
                         hidden_layer_sizes=(100,100),
                         alpha=1e-4,
                         random_state=42,
                         batch_size = batch_size)
# training process
MLPmodel.fit(X_train_tensor,y_train_tensor)

# Using the model for predicting values
# prediction
y_predict = MLPmodel.predict(X_test_tensor)

# Printing the coefficients as biasses of the model. 
# coeficents
print(f'Coefficients of model: {MLPmodel.coefs_}')

# bias
print(f'Biasses of model: {MLPmodel.intercepts_}')

# Printing the probability that an observation is of one class or another.
# predict probability
prob = MLPmodel.predict_proba(X_test_tensor)

print(f'Model probabilities: {prob}')

print(MLPmodel.score(X_test_tensor,y_test_tensor.ravel()))

# Printing the accuracy of the model with the testing dataset and the confusion matrix.
print(MLPmodel.score(X_test_tensor,y_test_tensor))
display(confusion_table(y_predict, y_test_tensor)) #confussion table

# Heat map of the confusion matrix.
sns.heatmap(confusion_table(y_predict, y_test_tensor.ravel()),
        annot=True,cmap = 'RdYlBu')
plt.show()
```
```text
Coefficients of model: [array([[-0.05755116,  0.20675238,  0.10642063, ..., -0.0332385 ,
        -0.50697882, -0.17986867],
       [-0.21494355,  0.0625744 , -0.08515892, ...,  0.18216305,
         0.22678764,  0.12838495],
       [ 0.06515298, -0.19076397, -0.15521821, ..., -0.13035903,
         0.04255732, -0.19021007],
       ...,
       [-0.20282711,  0.2151875 ,  0.17605087, ...,  0.21508439,
         0.12425495, -0.16968743],
       [ 0.1184709 , -0.21808224, -0.21921224, ..., -0.18139327,
         0.03878179,  0.09471487],
       [-0.21487154,  0.20009997, -0.20552044, ..., -0.11527019,
         0.03213427,  0.09485317]]), array([[ 0.00660854, -0.00720988, -0.16428306, ..., -0.14565851,
        -0.14349782,  0.13651902],
       [-0.10671473, -0.06117123, -0.09466632, ..., -0.12635184,
         0.16038936,  0.01715343],
       [ 0.16132688, -0.02337782, -0.06517319, ..., -0.14819883,
        -0.14848546, -0.16897011],
       ...,
       [-0.09449569, -0.03822551,  0.03686922, ..., -0.12750499,
         0.0569169 , -0.09903141],
       [-0.01481349, -0.07557038,  0.05603506, ...,  0.06510398,
        -0.06059068,  0.00096387],
       [-0.12215867, -0.0164282 ,  0.13201901, ..., -0.03511769,
        -0.07940114,  0.17293333]]), array([[-0.06745511],
       [ 0.26572456],
       [ 0.32395956],
       [ 0.76169798],
       [-0.05967823],
       [ 0.16370543],
       [-0.09452762],
       [-0.16223875],
       [-0.2271832 ],
       [-0.05569595],
       [ 0.1966746 ],
       [-0.1551905 ],
       [-0.21131353],
       [ 0.23139803],
       [-0.2314663 ],
       [ 0.20570382],
       [-0.12642051],
       [-0.5458086 ],
       [ 0.39824101],
       [-0.17562029],
       [ 0.01716158],
       [-0.27836647],
       [-0.25646506],
       [-0.22211622],
       [-0.18735842],
       [-0.48873319],
       [ 0.04970381],
       [ 0.05498068],
       [-0.18943246],
       [-0.03984834],
       [ 0.10486552],
       [ 0.43502596],
       [ 0.43900851],
       [ 0.30258889],
       [-0.08789452],
       [-0.05794136],
       [-0.00237716],
       [ 0.02151961],
       [-0.0332143 ],
       [-0.04807187],
       [ 0.21019066],
       [ 0.15598516],
       [-0.0316874 ],
       [ 0.13385882],
       [-0.26251893],
       [-0.19498429],
       [-0.07191773],
       [-0.0783608 ],
       [-0.31191598],
       [ 0.30027165],
       [-0.06862209],
       [ 0.08557167],
       [-0.33829086],
       [-0.2137553 ],
       [ 0.42508826],
       [ 0.18466083],
       [ 0.23613633],
       [ 0.25142432],
       [-0.04090184],
       [ 0.01712159],
       [ 0.1746448 ],
       [ 0.41387984],
       [ 0.02212747],
       [ 0.14383681],
       [ 0.23712094],
       [-0.05612603],
       [ 0.26565133],
       [ 0.16217948],
       [ 0.1928801 ],
       [-0.45685283],
       [-0.19740999],
       [-0.17326475],
       [-0.23790456],
       [ 0.70115003],
       [-0.28941197],
       [-0.13655459],
       [-0.25638268],
       [ 0.23101737],
       [ 0.06857619],
       [-0.13730135],
       [-0.16363209],
       [-0.1614803 ],
       [ 0.00886939],
       [-0.11140499],
       [ 0.17063137],
       [ 0.0936503 ],
       [-0.21204361],
       [ 0.11173297],
       [-0.12053173],
       [ 0.23965116],
       [-0.23704978],
       [ 0.03447696],
       [ 0.09627308],
       [ 0.990302  ],
       [-0.27045043],
       [ 0.090718  ],
       [-0.05745204],
       [ 0.20818363],
       [ 0.20180957],
       [-0.16634589]])]
Biasses of model: [array([-0.15277165, -0.15250676, -0.21258973,  0.10759648,  0.07601067,
       -0.01196149,  0.15772024,  0.13962847,  0.03916327,  0.16897445,
       -0.13496931, -0.17968335, -0.10646172, -0.20334082,  0.01422758,
        0.20020836, -0.21136368, -0.1749263 , -0.02227228,  0.19921264,
       -0.0853761 ,  0.00342888, -0.21039592, -0.16067943,  0.22324962,
        0.213322  , -0.22659013,  0.20723466,  0.06428301,  0.16881249,
       -0.02121206,  0.00669926, -0.00511754,  0.07656257, -0.16533934,
       -0.21655266, -0.08788709,  0.0943346 , -0.13679064,  0.07957626,
        0.21622779, -0.18821975,  0.08152768, -0.02776337,  0.16891525,
       -0.14865763,  0.08838285,  0.15429988,  0.20313715,  0.08470349,
       -0.00111705,  0.0533498 ,  0.1684384 ,  0.02987583, -0.21547319,
        0.19682093,  0.08685167,  0.0818114 , -0.13045719,  0.07290165,
       -0.04768428,  0.06957352, -0.1805075 ,  0.07242439,  0.22904087,
       -0.20717073,  0.21956182, -0.04242598,  0.17011335,  0.12834692,
        0.03037411,  0.10796035,  0.17363171, -0.04470752, -0.07711446,
        0.07656454,  0.1414364 ,  0.12053369,  0.13787471, -0.03005661,
        0.14529566, -0.17456238,  0.020413  , -0.22677348, -0.08114376,
       -0.06138961, -0.04784617,  0.08968651, -0.05109139, -0.02363529,
       -0.12042301, -0.05812103, -0.12643192, -0.19583114,  0.04746547,
        0.0769014 ,  0.05519555, -0.01675008, -0.05517384,  0.16670891]), array([-0.06605414,  0.13428902,  0.06284507, -0.05586647,  0.01508075,
        0.11874214,  0.13219433, -0.11764301, -0.05577354, -0.05708036,
       -0.15864034,  0.03376304,  0.12248026, -0.04146709,  0.08311297,
        0.04447683, -0.04297701, -0.12462208,  0.07270264,  0.12282498,
        0.06424532, -0.16781697, -0.09915449, -0.14945812,  0.02438246,
        0.08872037, -0.13499628,  0.11702887,  0.01859444,  0.06580823,
        0.16372229, -0.14177078,  0.1392992 , -0.12821889, -0.03708016,
        0.12174194,  0.1407199 ,  0.00195807,  0.03242496,  0.07197436,
        0.11591742, -0.1312871 ,  0.10741116, -0.00570796, -0.01797457,
       -0.02633326,  0.00319099,  0.15412794,  0.01567674,  0.15869729,
       -0.11075075,  0.07849528,  0.10811321,  0.06148675,  0.03786983,
        0.02282492,  0.12598279,  0.11289698, -0.1305171 ,  0.12019695,
       -0.0297675 , -0.08516237,  0.12781251, -0.0235473 ,  0.10685554,
       -0.08252089, -0.17096058, -0.00177841, -0.13839244,  0.08599456,
       -0.0780967 , -0.01952172, -0.14376662, -0.07181896,  0.09654331,
       -0.12069372,  0.05071061,  0.14540524,  0.06217764,  0.10774107,
       -0.1354562 , -0.08698739,  0.13660706,  0.07447553,  0.04748057,
        0.05671219, -0.0304191 , -0.15777021,  0.09864281,  0.06245146,
       -0.00940655,  0.13848079, -0.05633233, -0.06760081,  0.12806665,
        0.1027557 ,  0.00547064, -0.12791724, -0.09862785,  0.05345004]), array([-0.17641879])]

Model Probabilities
[[0.99259066 0.00740934]
 [0.99297326 0.00702674]
 [0.98958867 0.01041133]
 ...
 [0.99075256 0.00924744]
 [0.99456992 0.00543008]
 [0.98564906 0.01435094]]

Model Score and Accuracy
0.947726407914764

Confusion Matrix (MLPClassifier - Test Set)
Truth      0.0   1.0
Predicted           
0.0      39711  1388
1.0         58   891
```
![Texto Alternativo](images/mlp_confusion_heatmap.png)

### 3.4 PyTorch Custom CNN-LSTM Model
```python
# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) # Use dtype=torch.long for CrossEntropyLoss
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long) # Use dtype=torch.long for CrossEntropyLoss

# Create TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoader
batch_size = 64 # You can adjust this batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data prepared for PyTorch model.")

class LeakDetectionModel(nn.Module):
    def __init__(self,
                 input_dim=14,        # Number of features
                 window_size=60,      # Timesteps per sample (Sequence Length)
                 cnn_channels=[32, 64],
                 lstm_hidden=128,
                 lstm_layers=2,
                 dropout=0.3,
                 num_classes=2):      # 2 classes: no burst / burst
        super().__init__()

        cnn_layers = []
        in_channels = input_dim

        # Automatic Convolutional Blocks
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM for CNN-processed sequence
        self.lstm = nn.LSTM(input_size=cnn_channels[-1],
                             hidden_size=lstm_hidden,
                             num_layers=lstm_layers,
                             batch_first=True,
                             dropout=dropout)

        # Final Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Expected Input: (batch, seq_len, features)
        # If input is 2D (batch, features), add seq_len dimension
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch, 1, features)

        # Now input is (batch, seq_len, features)
        x = x.permute(0, 2, 1)    # (batch, features, seq_len) for CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 1)    # (batch, seq_len, channels) for LSTM
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

model = LeakDetectionModel(
    input_dim=14,
    window_size=60,
    cnn_channels=[32, 64],
    lstm_hidden=128,
    lstm_layers=2,
    dropout=0.3,
    num_classes=2
)

print(model)

# Create a dummy tensor with the same batch size as X_train_tensor and shape (batch_size, 1, features)
x = X_train_tensor
y = model(x)
print(y.shape)

# Model, criterion, and optimizer definition
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

# =================================================================
# TRAINING LOOP (CORRECT WAY - USING DATALOADER)
# =================================================================
model.train() # Set model to training mode
print("\n--- STARTING TRAINING (5 EPOCHS) ---")

for epoch in range(epochs):
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward Pass and Optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss weighted by batch size
        total_loss += loss.item() * X_batch.size(0)
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}') 


# =================================================================
# MODEL EVALUATION (NEW - GETTING ACCURACY AND MATRIX)
# =================================================================
model.eval() # Set model to evaluation mode (disables dropout/BatchNorm)
y_pred_list = []
y_true_list = []

with torch.no_grad(): # Disable gradient calculation for efficiency
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, y_pred = torch.max(outputs, 1) # Get the index of the class with the highest probability
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.cpu().numpy())

# Convert to NumPy arrays to use Scikit-learn metrics
y_pred_final = np.array(y_pred_list)
y_true_final = np.array(y_true_list)

# Import metrics functions
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_true_final, y_pred_final)
cm = confusion_matrix(y_true_final, y_pred_final)

print("\n--- EVALUATION RESULTS (TEST SET) ---")
print(f"Accuracy: {accuracy}")

# Printing Confusion Matrix in the required output format
print("Matriz de Confusão:")
print("Truth 0.0 1.0")
print(f"Predicted 0.0 {cm[0, 0]} {cm[0, 1]}")
print(f"Predicted 1.0 {cm[1, 0]} {cm[1, 1]}")
```
```text
Model Architecture Summary
LeakDetectionModel(
  (cnn): Sequential(
    (0): Conv1d(14, 32, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.3, inplace=False)
    (4): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
    (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Dropout(p=0.3, inplace=False)
  )
  (lstm): LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3)
  (fc): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=64, out_features=2, bias=True)
  )
)

Output Tensor Shape
torch.Size([98112, 2])

Training Loss per Epoch (5 Epochs)
Epoch [1/5], Loss: 0.1539
Epoch [2/5], Loss: 0.1485
Epoch [3/5], Loss: 0.1420
Epoch [4/5], Loss: 0.1390
Epoch [5/5], Loss: 0.1328

Evaluation Results (Test Set)

Accuracy: 0.9597602739726028

Confusion Matrix:
Truth        0.0  1.0
Predicted
0.0        39675   25
1.0         1667  681
```
