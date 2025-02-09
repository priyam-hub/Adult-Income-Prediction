<div align="center">

# 🤖 **Adult Income Prediction**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) • [Installation](#installation) • [Documentation](#documentation) • [Usage](#usage) • [Contributing](#contributing)

</div>

---

# **Executive Summary**
This project provides an in-depth exploratory data analysis (EDA) and machine learning modeling for the Adult Income dataset. The objective is to analyze demographic and economic factors influencing income levels and predict whether an individual earns more than $50K per year. Key insights include:

- Education level and occupation significantly impact income levels.
- Individuals in managerial and professional occupations are more likely to earn above $50K.
- Age and capital gain contribute strongly to income prediction.
- Logistic Regression and Decision Tree models perform well for classification.

These findings help in understanding income distribution and improving targeted policy-making and financial planning.

---

# **Problem Statement**
To analyze and predict whether an individual earns more than $50K annually based on demographic and economic attributes. The analysis aims to:

- Identify key factors influencing income levels.
- Develop predictive models to classify income groups.
- Provide insights into income disparities across different demographic groups.
---

# **Dataset**
The Adult Income Dataset is a widely used dataset for classification tasks, primarily aimed at predicting whether an individual's income exceeds $50K per year based on various demographic and employment-related attributes. The dataset was originally extracted from the 1994 U.S. Census Bureau database.

### Dataset Link:
```bash
https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
```
### Dataset Details
- **Source**: Kaggle
- **Size**: 48,842 records (32,561 training + 16,281 test)
- **Attributes**: 14 features + 1 target variable
- **Task**: Binary classification (<=50K or >50K income)

### Features:
- **Demographics**: age, sex, race, native-country
- **Education**: education, education-num
- **Employment**: workclass, occupation, hours-per-week
- **Financial**: capital-gain, capital-loss
- **Marital Status & Relationships**: marital-status, relationship
- **Target Variable**: income (<=50K or >50K)
---

# **Methodology**
### Data Preprocessing
- Handled missing values and removed duplicates.
- Encoded categorical variables using OneHotEncoder and LabelEncoder.
- Standardized numerical features using StandardScaler.

### Tools Used
- **Python Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, SciPy
- **Machine Learning Models**: Logistic Regression, Decision Tree, Naive Bayes, Support Vector Classifier
---

# Visualization:

### Count Plot of the Target Variable:
![Count Plot](results/Count_Plot_of_Target_Variable.png)

### Skewness and Kurtosis:
![Skewness and Kurtosis](results/Skewness_and_Kurtosis.png)

# Key Findings

| Category            | Finding | Implication |
|---------------------|------------------------------------------------------------|--------------------------------------------------------------|
| **Demographic Analysis** | Education level significantly correlates with income. | Higher education leads to better job opportunities and higher earnings. |
| **Occupation Trends** | People in managerial and professional roles have a higher probability of earning above $50K. | Upskilling and career advancement in these sectors can lead to better financial outcomes. |
| **Capital Gain & Age Impact** | Older individuals with capital gains are more likely to earn above $50K. | Investments and financial planning contribute to income growth over time. |
---

# Feature Importance
![Feature Importance](results/Feature_Importance.png)


### Most Important Feature

**Relationship**: Strongly influences income classification, with stable relationships often correlating with higher income.

---

# Different Model Fitting - Insights

| Model | Accuracy (%) | Precision | Recall | F1 Score (%) | Notes |
|-------------------------|------------|-----------|--------|-------------|--------------------------------------------------------------|
| **Gaussian Naive Bayes** | 79.85 | Balanced | Balanced | 77.13 | Good overall performance. |
| **Multinomial Naive Bayes** | 78.01 | Lower than recall | Higher recall | 74.15 | Lower precision affects F1 score. |
| **Bernoulli Naive Bayes** | 78.48 | Balanced | Balanced | 78.20 | Precision and recall are almost equal. |
| **SVM (RBF Kernel)** | 78.97 | High | Lower recall | 72.35 | Good precision, but recall is low. |
| **SVM (Polynomial Kernel, Degree 2)** | 78.21 | High | Slightly lower recall | 71.16 | High precision but recall is slightly lower. |
| **SVM (Polynomial Kernel, Degree 3)** | 77.70 | Higher than recall | Lower recall | 70.27 | Precision is better than recall. |
| **Decision Tree (Gini Index)** | 81.92 | Balanced | Balanced | 81.93 | **Best model due to highest accuracy and balanced metrics.** |
| **Decision Tree (Entropy)** | 81.20 | Balanced | Balanced | 81.21 | Performs well but slightly lower than Gini Index. |

### Best Model for Final Fitting:  
✅ **Decision Tree (Gini Index)** is the best due to the highest accuracy and balanced metrics.  

# Final Model after Hyperparameter Tuning and Evaluation

#### Best Parameter Set
```bash
Best Parameters: {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 20}
```

#### Error Metric Table

| Metric | Value |
|-----------------------------|--------|
| **Type 1 Error (False Positive Rate)** | 0.0661 |
| **Type 2 Error (False Negative Rate)** | 0.3763 |
| **Accuracy** | 0.8610 |
| **Recall** | 0.8610 |
| **Precision** | 0.8555 |
| **F1 Score** | 0.8566 |

#### Confusion Matrix
![Confusion Matrix](results/Final_CM.png)

#### ROC Curve
![ROC Curve](results/ROC_Curve.png)
---

# Recommendations
- **Education & Training**: Encouraging professional education and skill development can enhance income levels.
- **Job Market Policies**: Government and corporate policies should focus on wage equality and skill-based hiring.
- **Financial Planning**: Capital investment and savings play a crucial role in wealth accumulation.

# Conclusion
This analysis provides key insights into the demographic and economic factors affecting income levels. The findings can be used for financial planning, policy-making, and improving career growth strategies. Future work could explore additional features and more advanced machine learning techniques to improve prediction accuracy.

---

<div align="center">

**Made with ❤️ by Priyam Pal - AI and Data Science Engineer**

[↑ Back to Top](#Executive Summary)

</div>

