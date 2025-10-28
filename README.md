# Final Project: Burst Detection and Statistical Analysis in Water Distribution Systems

## 1. Project Overview
This repository contains the final project work for the discipline [**Discipline Name**]. The main objective of this study is two-fold:
1.  **Statistical Analysis:** Conduct comprehensive exploratory data analysis (EDA) and hypothesis testing (t-tests, ANOVA, non-parametric tests) on sensor data from a Water Distribution System (WDS).
2.  **Machine Learning Classification:** Implement and evaluate different classification models (Decision Trees, Naive Bayes, and Neural Networks) to predict the occurrence of a **burst** event (leak detection) based on flow and pressure readings.

---

## 2. Authors
* **Mateo H. Sanchez**
* **Jose A. Calvetty**
* **Milena R. de Sousa**

---

## 3. Dataset
The data is a collection of simulated sensor readings (flow meters and pressure sensors) designed for burst detection problems.
* **Source:** Generated datasets for burst detection in water distribution systems
* **Link:** [https://github.com/ArieleZanfei/generated-datasets-for-burst-detection-in-water-distribution-systems](https://github.com/ArieleZanfei/generated-datasets-for-burst-detection-in-water-distribution-systems)
* **Files Used:** The analysis merges the first four CSV files found in the dataset repository.

---

## 4. Technical Stack
The project was developed in a Google Colaboratory environment and utilizes the following tools and libraries:

| Category | Tools/Libraries |
| :--- | :--- |
| **Language** | Python 3.x |
| **Data Handling** | Pandas, NumPy |
| **Statistics** | SciPy (`scipy.stats`) |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (`sklearn`), ISLP |
| **Deep Learning** | PyTorch, `torchinfo` |

---

## 5. Analysis & Modeling Structure

The analysis is structured into two main parts within the Jupyter Notebook (`.ipynb` file):

### 5.1 Exploratory Data Analysis (EDA) and Hypothesis Testing

This section performs a deep dive into the raw sensor data:
* **Descriptive Statistics:** Calculation of mean, standard deviation, variance, skewness, and kurtosis.
* **Parametric Tests:** Independent **T-tests** (pairwise) and **ANOVA** (multi-group) to compare means of 'flow\_meter' and 'press' columns.
* **Non-Parametric Tests:** **Shapiro-Wilk** (Normality) and **Levene's** (Homogeneity of Variance) tests to check assumptions, followed by **Kruskal-Wallis** and **Mann-Whitney U** tests as alternatives.
* **Correlation:** Visualization of **Pearson** and **Spearman** correlation matrices.
* **Distribution Analysis:** Histograms with KDE for all features.

### 5.2 Machine Learning Classification

This section focuses on classifying the `burst` event (binary target):
* **Decision Tree:** Implementation of a Decision Tree Classifier (`DTC`) using **entropy**, including visualization, cross-validation, and **Cost-Complexity Pruning (CCP)** to find the optimal tree.
* **Naive Bayes:** Implementation of the **Gaussian Naive Bayes** classifier, focusing on model parameters ($\mu, \sigma^2$) and performance metrics.
* **Neural Networks:**
    * **Scikit-learn MLP Classifier:** A standard Multi-Layer Perceptron implementation.
    * **PyTorch Custom Model (CNN-LSTM):** Implementation of a complex deep learning model combining **1D Convolutional Layers** (CNN) for feature extraction and **Long Short-Term Memory** (LSTM) for sequence modeling, trained using `CrossEntropyLoss`.

---

## 6. Key Findings and Results Summary

| Model / Test | Key Statistic / Metric | Result |
| :--- | :--- | :--- |
| **T-tests / ANOVA** | P-values | [Example: All P-values were < 0.05, indicating significant differences between sensor means.] |
| **Kruskal-Wallis** | P-values | [Example: P-values were 0.0, confirming significant differences in medians.] |
| **Correlation** | Max Correlation | [Example: Strongest correlation was 0.98 between 'flow\_meter\_1' and 'flow\_meter\_2'.] |
| **Initial DTC** | Test Accuracy | **[0.XX]** |
| **Pruned DTC** | Test Accuracy | **[0.YY]** (Best model via CCP) |
| **Naive Bayes** | Test Accuracy | **[0.ZZ]** |
| **MLP Classifier (Sklearn)** | Test Accuracy | **[0.WW]** |
| **CNN-LSTM (PyTorch)** | Final Training Loss | **[X.XXX]** |

[**IMPORTANT:** Copy and paste your key findings from the final code cell's summary here. For example: "The Pruned Decision Tree achieved the highest accuracy of 95.2% on the test set."]

---

## 7. How to Run the Code

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    ```
2.  **Download the Data:**
    * The code is set up to read data from local CSV files. You must download the datasets from the source link and ensure the CSV files are present in the directory structure expected by the `glob` search.
3.  **Open in Colab/Jupyter:**
    * Upload or open the `[Your_Notebook_Name].ipynb` file in **Google Colaboratory** or **Jupyter Notebook**.
4.  **Install Dependencies:**
    * Ensure all `!pip install` commands at the start of the notebook are run.
5.  **Run All Cells:**
    * Run the notebook cells sequentially from top to bottom.
