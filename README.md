
# Income Classification ML Project

## Problem Statement
The objective of this project is to predict whether an individual earns more than $50K per year using various machine learning classification algorithms. The task is formulated as a binary classification problem where the target variable is income (>50K or ≤50K).

## Dataset Description
Dataset Used: UCI Adult Income Dataset
	•	Source: UCI Machine Learning Repository
	•	Number of Instances: 48,842
	•	Number of Features: 14
	•	Target Variable: Income (Binary Classification)
	•	Features include:
	•	Age
	•	Education
	•	Occupation
	•	Hours per week
	•	Capital gain/loss
	•	Marital status
	•	Work class
	•	etc.

Data preprocessing performed:
	•	Handling categorical variables using Label Encoding
	•	Feature scaling using StandardScaler
	•	Train-Test split (80-20)

## Models Used
The following six classification models were implemented:
	1.	Logistic Regression
	2.	Decision Tree Classifier
	3.	K-Nearest Neighbor (KNN)
	4.	Naive Bayes (Gaussian)
	5.	Random Forest (Ensemble)
	6.	XGBoost (Ensemble)

# Comparison Table
| ML Model Name               | Accuracy | AUC  | Precision | Recall | F1    | MCC   |
|-----------------------------|----------|------|-----------|--------|-------|-------|
| Logistic Regression         | 0.835    | 0.692| 0.679     | 0.442  | 0.535 | 0.455 |
| Decision Tree               | 0.820    | 0.759| 0.571     | 0.651  | 0.609 | 0.494 |
| kNN                         | 0.845    | 0.749| 0.658     | 0.581  | 0.617 | 0.522 |
| Naive Bayes                 | 0.815    | 0.629| 0.650     | 0.302  | 0.413 | 0.353 |
| Random Forest (Ensemble)    | 0.855    | 0.773| 0.675     | 0.628  | 0.651 | 0.560 |
| XGBoost (Ensemble)          | 0.865    | 0.762| 0.735     | 0.581  | 0.649 | 0.573 |

# observations on the performance of each model
| ML Model Name               | Observation about Model Performance |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Provides stable baseline performance with moderate precision but lower recall for high-income class. |
| Decision Tree               | Captures nonlinear relationships well; slightly higher recall but prone to overfitting. |
| kNN                         | Performs well after scaling; balanced precision-recall tradeoff. |
| Naive Bayes                 | Fast model but lower recall due to independence assumption of features. |
| Random Forest (Ensemble)    | Strong overall performance with improved F1 and MCC; handles feature interactions effectively. |
| XGBoost (Ensemble)          | Best performing model overall with highest accuracy and MCC; robust and well-generalized. |