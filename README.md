# 💰 Credit Risk Prediction (Home Credit Default Risk)

## 📝 Project Overview

The objective of this project is to build a machine learning model capable of assessing the credit risk of potential clients. Leveraging historical data from loan applications (based on the Home Credit competition), the project aims to predict the probability of a client defaulting on their loan obligations.

This is a **binary classification** problem dealing with a highly imbalanced dataset. The project places a strong emphasis on **Advanced Feature Engineering**, including aggregating data from multiple sources, and employs techniques like **SMOTE** to handle the class imbalance effectively.

### Key Highlights:
* Comprehensive Exploratory Data Analysis (EDA) and visualization of variable distributions.
* Advanced Feature Engineering involving data aggregation from relational tables.
* Implementation of **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training data.
* Performance comparison between linear models and ensemble tree-based methods.

---

## 🛠️ Technologies & Libraries

The project is implemented in **Python**, utilizing the following tech stack:

<p align="center">
  <img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/-Matplotlib-033069?style=flat&logo=matplotlib&logoColor=white">
  <img src="https://img.shields.io/badge/-Seaborn-77ACF1?style=flat&logo=seaborn&logoColor=white">
  <img src="https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white">
</p>

---

## ⚙️ Methodology

### 1. Data Exploration and Preprocessing
* Detailed analysis of missing values and imputation strategies (using median/mode).
* Correlation analysis to understand relationships between features and the `TARGET` variable.
* Encoding categorical variables using One-Hot Encoding and Label Encoding.

### 2. Advanced Feature Engineering & Aggregation
A critical part of this project was transforming raw data into meaningful predictors.
* **Data Aggregation:** Extensive use of `.agg` functions to summarize information from supplementary tables. This process involved calculating statistical measures (mean, max, min, sum, etc.) for historical transactions and previous applications to capture the client's full financial history.
* **Domain Features:** Creation of specific financial ratios and domain-based features to better represent the applicant's financial stability and ability to repay the loan.

### 3. Handling Imbalanced Data (SMOTE)
The dataset was characterized by a significant class imbalance, with a vast majority of clients repaying their loans. To prevent the model from becoming biased towards the majority class, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to generate synthetic examples for the minority class (`Default`) in the training set.

### 4. Model Training and Evaluation
Three primary modeling approaches were tested and compared:
* **Logistic Regression:** Used as a baseline model.
* **Random Forest Classifier:** An ensemble method to reduce variance.
* **Gradient Boosting Classifier:** An advanced boosting algorithm that typically yields the best performance on tabular data.

---

## 📊 Results & Conclusion

Models were evaluated primarily using the **ROC-AUC** metric, which is the industry standard for credit risk assessment as it is robust to class imbalance.

* The extensive feature engineering process, particularly the aggregation of historical data, significantly improved model performance compared to using raw features alone.
* Applying SMOTE proved essential for the model to correctly identify potential defaulters, which is the critical business objective.
* Tree-based ensemble models (**Gradient Boosting**) achieved the highest ROC-AUC scores, demonstrating their superior ability to capture non-linear relationships in complex financial data.
