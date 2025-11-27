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
for i in range(1):
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
|index|flow\_meter1|flow\_meter2|flow\_meter3|flow\_meter4|3\_press|12\_press|36\_press|50\_press|60\_press|84\_press|93\_press|112\_press|138\_press|139\_press|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|count|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|35040\.0|
|mean|31\.673598715753425|33\.62173932648402|90\.44670773401825|28\.59792970890411|32\.347362243150684|31\.880472260273976|35\.8208138413242|39\.094596318493146|33\.52866675228311|32\.4848078196347|31\.39055967465753|34\.864807163242006|35\.073812014840186|31\.14494192351598|
|std|9\.027985215728428|12\.23030737323786|42\.426462701271085|8\.68083274430615|2\.1095777595827823|1\.9306984132091372|2\.5038275387282245|0\.054798397581984876|2\.5736837170235614|2\.5404487765769104|2\.495707274067113|2\.1825432335846084|2\.063034753404274|1\.5528676057684172|
|min|17\.663|13\.978|19\.244|15\.032|25\.083|21\.54|25\.397|38\.942|26\.076|24\.858|23\.708|26\.591|29\.052|26\.126|
|25%|23\.54675|23\.079749999999997|55\.6985|20\.887500000000003|30\.969|30\.615|34\.176|39\.059|31\.855|30\.835|29\.758000000000003|33\.454|33\.745|30\.128|
|50%|34\.153|37\.237|104\.92|30\.9875|32\.031499999999994|31\.605|35\.516|39\.087|33\.177|32\.143|31\.0185|34\.5715|34\.799499999999995|30\.936|
|75%|38\.12025|42\.422|120\.87|34\.804|34\.334|33\.696|38\.22|39\.147|36\.013250000000006|34\.937|33\.75425|36\.972|37\.056|32\.605|
|max|60\.277|81\.844|171\.03|57\.974|35\.238|34\.52|39\.157|39\.168|36\.93|35\.843|34\.803000000000004|37\.742|37\.809|33\.262|
|mode|19\.081|16\.272000000000002|112\.4|16\.445999999999998|35\.052|34\.349000000000004|38\.99|39\.164|36\.78|35\.689|34\.582|37\.736|37\.8|33\.128|
|var|81\.50451705541107|149\.58041844387637|1800\.0047373423465|75\.35685713461784|4\.450318323726312|3\.72759636276828|6\.26915234369384|0\.003002864377553286|6\.623847875272215|6\.4538799864111205|6\.2285547978314995|4\.7634949664659585|4\.256112393753835|2\.411397801044936|
|std|9\.027985215728428|12\.23030737323786|42\.426462701271085|8\.68083274430615|2\.1095777595827823|1\.9306984132091372|2\.5038275387282245|0\.054798397581984876|2\.5736837170235614|2\.5404487765769104|2\.495707274067113|2\.1825432335846084|2\.063034753404274|1\.5528676057684172|
|sem|0\.048229059221675114|0\.06533641831574351|0\.22664950520974997|0\.04637451064820088|0\.01126972943225385|0\.0103141250107092|0\.013375879973286035|0\.0002927425221775977|0\.013749063765627364|0\.013571516962799285|0\.013332500114338874|0\.011659523620290487|0\.011021088639463513|0\.00829568745765646|
|skew|-0\.1703207333767162|-0\.24615266914919695|-0\.367251376231039|-0\.1903335864866669|-0\.19332078723101045|-0\.2294182472007986|-0\.24432167038590002|-0\.2248512771153826|-0\.24551756283865586|-0\.25026991947991223|-0\.2008637313723598|-0\.2467397076842774|-0\.24219856683851876|-0\.21421467106938044|
|kurtosis|-1\.1608209197303387|-1\.151036284381629|-1\.1442015048461702|-1\.1595625256875854|-0\.9482701932226849|-0\.7904117405229285|-0\.9132800051474295|-0\.9325513980243478|-0\.9212995228186669|-0\.9141002392068924|-0\.9414866450390722|-0\.91020511175953|-0\.9221283964852822|-0\.9289584954291681|


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
T-test between flow_meter1 and flow_meter2: TtestResult(statistic=np.float64(-23.989242308107286), pvalue=np.float64(1.1709943396170134e-126), df=np.float64(70078.0))
T-test between flow_meter1 and flow_meter3: TtestResult(statistic=np.float64(-253.63403700511086), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between flow_meter1 and flow_meter4: TtestResult(statistic=np.float64(45.96884981394347), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between flow_meter2 and flow_meter3: TtestResult(statistic=np.float64(-240.90740358040776), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between flow_meter2 and flow_meter4: TtestResult(statistic=np.float64(62.702449403142325), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between flow_meter3 and flow_meter4: TtestResult(statistic=np.float64(267.3441364378407), pvalue=np.float64(0.0), df=np.float64(70078.0))

Independent t-tests between pairs of 'press' columns:
T-test between 3_press and 12_press: TtestResult(statistic=np.float64(30.561545811193916), pvalue=np.float64(8.741629880035233e-204), df=np.float64(70078.0))
T-test between 3_press and 36_press: TtestResult(statistic=np.float64(-198.589659003892), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 3_press and 50_press: TtestResult(statistic=np.float64(-598.5022896924818), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 3_press and 60_press: TtestResult(statistic=np.float64(-66.44899118116206), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 3_press and 84_press: TtestResult(statistic=np.float64(-7.79140797071204), pvalue=np.float64(6.717187443553869e-15), df=np.float64(70078.0))
T-test between 3_press and 93_press: TtestResult(statistic=np.float64(54.8077114364143), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 3_press and 112_press: TtestResult(statistic=np.float64(-155.2466803202297), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 3_press and 138_press: TtestResult(statistic=np.float64(-172.96549568553263), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 3_press and 139_press: TtestResult(statistic=np.float64(85.92548299670725), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 12_press and 36_press: TtestResult(statistic=np.float64(-233.2848484410716), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 12_press and 50_press: TtestResult(statistic=np.float64(-699.1596514901961), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 12_press and 60_press: TtestResult(statistic=np.float64(-95.89369067932893), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 12_press and 84_press: TtestResult(statistic=np.float64(-35.45314928906782), pvalue=np.float64(6.865488773132033e-273), df=np.float64(70078.0))
T-test between 12_press and 93_press: TtestResult(statistic=np.float64(29.06396784028744), pvalue=np.float64(1.2882903463587585e-184), df=np.float64(70078.0))
T-test between 12_press and 112_press: TtestResult(statistic=np.float64(-191.71136123272908), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 12_press and 138_press: TtestResult(statistic=np.float64(-211.5557499056104), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 12_press and 139_press: TtestResult(statistic=np.float64(55.56918512740698), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 36_press and 50_press: TtestResult(statistic=np.float64(-244.69408442037025), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 36_press and 60_press: TtestResult(statistic=np.float64(119.49439560714572), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 36_press and 84_press: TtestResult(statistic=np.float64(175.07073669987858), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 36_press and 93_press: TtestResult(statistic=np.float64(234.582450999842), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 36_press and 112_press: TtestResult(statistic=np.float64(53.876963044673595), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 36_press and 138_press: TtestResult(statistic=np.float64(43.10098529573694), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 36_press and 139_press: TtestResult(statistic=np.float64(297.0783683996958), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 50_press and 60_press: TtestResult(statistic=np.float64(404.73071161471745), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 50_press and 84_press: TtestResult(statistic=np.float64(486.9206110546857), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 50_press and 93_press: TtestResult(statistic=np.float64(577.6996169265838), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 50_press and 112_press: TtestResult(statistic=np.float64(362.6611783158141), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 50_press and 138_press: TtestResult(statistic=np.float64(364.6977871200118), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 50_press and 139_press: TtestResult(statistic=np.float64(957.6914851945627), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 60_press and 84_press: TtestResult(statistic=np.float64(54.03282978653538), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 60_press and 93_press: TtestResult(statistic=np.float64(111.63987408452256), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 60_press and 112_press: TtestResult(statistic=np.float64(-74.11787337578322), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 60_press and 138_press: TtestResult(statistic=np.float64(-87.6875041060923), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 60_press and 139_press: TtestResult(statistic=np.float64(148.4458927470904), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 84_press and 93_press: TtestResult(statistic=np.float64(57.5170427901084), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 84_press and 112_press: TtestResult(statistic=np.float64(-133.0188741109686), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 84_press and 138_press: TtestResult(statistic=np.float64(-148.08812077127098), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 84_press and 139_press: TtestResult(statistic=np.float64(84.23589752445744), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 93_press and 112_press: TtestResult(statistic=np.float64(-196.15684065314028), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 93_press and 138_press: TtestResult(statistic=np.float64(-212.92967806225536), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 93_press and 139_press: TtestResult(statistic=np.float64(15.641778063178283), pvalue=np.float64(4.6860104913267954e-55), df=np.float64(70078.0))
T-test between 112_press and 138_press: TtestResult(statistic=np.float64(-13.027005605550858), pvalue=np.float64(9.530875991496502e-39), df=np.float64(70078.0))
T-test between 112_press and 139_press: TtestResult(statistic=np.float64(259.95701908907597), pvalue=np.float64(0.0), df=np.float64(70078.0))
T-test between 138_press and 139_press: TtestResult(statistic=np.float64(284.8182892784834), pvalue=np.float64(0.0), df=np.float64(70078.0))
```
```text
ANOVA test for 'flow_meter' columns:
F-statistic: 58482.55070340394
P-value: 0.0

ANOVA test for 'press' columns:
F-statistic: 47656.75028190731
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
Shapiro-Wilk test for flow_meter1: ShapiroResult(statistic=np.float64(0.9290840779132447), pvalue=np.float64(5.310906643292244e-81))
Shapiro-Wilk test for flow_meter2: ShapiroResult(statistic=np.float64(0.9255729564190031), pvalue=np.float64(4.150172155418699e-82))
Shapiro-Wilk test for flow_meter3: ShapiroResult(statistic=np.float64(0.9150530180221393), pvalue=np.float64(3.547779730847022e-85))
Shapiro-Wilk test for flow_meter4: ShapiroResult(statistic=np.float64(0.928304040166191), pvalue=np.float64(2.9869033356431504e-81))

Normality test (Shapiro-Wilk) for 'press' columns:
Shapiro-Wilk test for 3_press: ShapiroResult(statistic=np.float64(0.9389875795524175), pvalue=np.float64(1.324492141448592e-77))
Shapiro-Wilk test for 12_press: ShapiroResult(statistic=np.float64(0.9400594928622569), pvalue=np.float64(3.2917847888771494e-77))
Shapiro-Wilk test for 36_press: ShapiroResult(statistic=np.float64(0.9365552428719225), pvalue=np.float64(1.76354975177036e-78))
Shapiro-Wilk test for 50_press: ShapiroResult(statistic=np.float64(0.9365123669177908), pvalue=np.float64(1.7029847734756486e-78))
Shapiro-Wilk test for 60_press: ShapiroResult(statistic=np.float64(0.9350410901759658), pvalue=np.float64(5.1957239745660875e-79))
Shapiro-Wilk test for 84_press: ShapiroResult(statistic=np.float64(0.9351691217076247), pvalue=np.float64(5.755890709263658e-79))
Shapiro-Wilk test for 93_press: ShapiroResult(statistic=np.float64(0.9384795725683155), pvalue=np.float64(8.644361813522865e-78))
Shapiro-Wilk test for 112_press: ShapiroResult(statistic=np.float64(0.934976940014398), pvalue=np.float64(4.936217675357944e-79))
Shapiro-Wilk test for 138_press: ShapiroResult(statistic=np.float64(0.9353475164343089), pvalue=np.float64(6.640423930398681e-79))
Shapiro-Wilk test for 139_press: ShapiroResult(statistic=np.float64(0.9394646252883341), pvalue=np.float64(1.9827745512581185e-77))

Homogeneity of variances test (Levene) for 'flow_meter' columns:
Levene test for 'flow_meter' columns: LeveneResult(statistic=np.float64(28109.645644687767), pvalue=np.float64(0.0))

Homogeneity of variances test (Levene) for 'press' columns:
Levene test for 'press' columns: LeveneResult(statistic=np.float64(9966.67399817068), pvalue=np.float64(0.0))
/usr/local/lib/python3.12/dist-packages/scipy/stats/_axis_nan_policy.py:579: UserWarning: scipy.stats.shapiro: For N > 5000, computed p-value may not be accurate. Current N is 35040.
  res = hypotest_fun_out(*samples, **kwds)
```
```text
Kruskal-Wallis test for 'flow_meter' columns:
Kruskal-Wallis statistic: 42514.692701154534
P-value: 0.0

Kruskal-Wallis test for 'press' columns:
Kruskal-Wallis statistic: 178675.75217785718
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
Mann-Whitney U test between flow_meter1 and flow_meter2: MannwhitneyuResult(statistic=np.float64(540968330.0), pvalue=np.float64(2.4182593402743655e-163))
Mann-Whitney U test between flow_meter1 and flow_meter3: MannwhitneyuResult(statistic=np.float64(182941956.0), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter1 and flow_meter4: MannwhitneyuResult(statistic=np.float64(751994528.5), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter2 and flow_meter3: MannwhitneyuResult(statistic=np.float64(191270042.0), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter2 and flow_meter4: MannwhitneyuResult(statistic=np.float64(780219632.5), pvalue=np.float64(0.0))
Mann-Whitney U test between flow_meter3 and flow_meter4: MannwhitneyuResult(statistic=np.float64(1064596144.0), pvalue=np.float64(0.0))

Mann-Whitney U tests between pairs of 'press' columns:
Mann-Whitney U test between 3_press and 12_press: MannwhitneyuResult(statistic=np.float64(706741439.0), pvalue=np.float64(2.166573684804365e-263))
Mann-Whitney U test between 3_press and 36_press: MannwhitneyuResult(statistic=np.float64(194826789.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 50_press: MannwhitneyuResult(statistic=np.float64(0.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 60_press: MannwhitneyuResult(statistic=np.float64(440085923.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 84_press: MannwhitneyuResult(statistic=np.float64(572087411.0), pvalue=np.float64(5.754876809036927e-55))
Mann-Whitney U test between 3_press and 93_press: MannwhitneyuResult(statistic=np.float64(766078639.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 112_press: MannwhitneyuResult(statistic=np.float64(277480028.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 138_press: MannwhitneyuResult(statistic=np.float64(248181809.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 3_press and 139_press: MannwhitneyuResult(statistic=np.float64(821175993.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 36_press: MannwhitneyuResult(statistic=np.float64(137907586.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 50_press: MannwhitneyuResult(statistic=np.float64(0.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 60_press: MannwhitneyuResult(statistic=np.float64(386454400.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 84_press: MannwhitneyuResult(statistic=np.float64(504103353.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 93_press: MannwhitneyuResult(statistic=np.float64(667327048.0), pvalue=np.float64(1.447143708255953e-88))
Mann-Whitney U test between 12_press and 112_press: MannwhitneyuResult(statistic=np.float64(210563451.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 138_press: MannwhitneyuResult(statistic=np.float64(169943521.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 12_press and 139_press: MannwhitneyuResult(statistic=np.float64(763593387.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 50_press: MannwhitneyuResult(statistic=np.float64(83485821.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 60_press: MannwhitneyuResult(statistic=np.float64(889573154.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 84_press: MannwhitneyuResult(statistic=np.float64(983178954.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 93_press: MannwhitneyuResult(statistic=np.float64(1095490273.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 112_press: MannwhitneyuResult(statistic=np.float64(766776589.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 138_press: MannwhitneyuResult(statistic=np.float64(742010537.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 36_press and 139_press: MannwhitneyuResult(statistic=np.float64(1156175561.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 60_press: MannwhitneyuResult(statistic=np.float64(1227801600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 84_press: MannwhitneyuResult(statistic=np.float64(1227801600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 93_press: MannwhitneyuResult(statistic=np.float64(1227801600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 112_press: MannwhitneyuResult(statistic=np.float64(1227801600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 138_press: MannwhitneyuResult(statistic=np.float64(1227801600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 50_press and 139_press: MannwhitneyuResult(statistic=np.float64(1227801600.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 84_press: MannwhitneyuResult(statistic=np.float64(765173937.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 93_press: MannwhitneyuResult(statistic=np.float64(877849420.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 112_press: MannwhitneyuResult(statistic=np.float64(417648096.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 138_press: MannwhitneyuResult(statistic=np.float64(391345386.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 60_press and 139_press: MannwhitneyuResult(statistic=np.float64(940390362.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 93_press: MannwhitneyuResult(statistic=np.float64(774826749.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 112_press: MannwhitneyuResult(statistic=np.float64(314439837.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 138_press: MannwhitneyuResult(statistic=np.float64(290349268.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 84_press and 139_press: MannwhitneyuResult(statistic=np.float64(814094843.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 93_press and 112_press: MannwhitneyuResult(statistic=np.float64(207460853.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 93_press and 138_press: MannwhitneyuResult(statistic=np.float64(176539358.5), pvalue=np.float64(0.0))
Mann-Whitney U test between 93_press and 139_press: MannwhitneyuResult(statistic=np.float64(665046273.0), pvalue=np.float64(2.524165330127432e-81))
Mann-Whitney U test between 112_press and 138_press: MannwhitneyuResult(statistic=np.float64(564400600.5), pvalue=np.float64(2.696442766663645e-76))
Mann-Whitney U test between 112_press and 139_press: MannwhitneyuResult(statistic=np.float64(1122283045.0), pvalue=np.float64(0.0))
Mann-Whitney U test between 138_press and 139_press: MannwhitneyuResult(statistic=np.float64(1144966168.5), pvalue=np.float64(0.0))
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
