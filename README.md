# 🤖 Machine Learning 
---

# 📚 Table of Contents

1. Machine Learning Basics
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Statistics for ML
5. Feature Engineering
6. Supervised Learning
7. Unsupervised Learning
8. Model Evaluation
9. Hyperparameter Tuning
10. Ensemble Learning
11. Dimensionality Reduction
12. Time Series Basics
13. Natural Language Processing Basics
14. Deep Learning Basics
15. Interview Questions
16. Cheat Sheets

---

# 1. Machine Learning Basics

## What is Machine Learning?

Machine Learning is the process of enabling computers to learn patterns from data without being explicitly programmed.

### Types of ML

| Type | Description | Algorithms |
|-------|-------------|------------|
| Supervised | Labeled Data | Linear Regression, Logistic Regression, SVM, Decision Tree |
| Unsupervised | Unlabeled Data | K-Means, DBSCAN, PCA |
| Reinforcement | Reward-based Learning | Q-Learning, DQN |

---

## ML Pipeline

Raw Data
↓
Data Cleaning
↓
EDA
↓
Feature Engineering
↓
Train Test Split
↓
Model Training
↓
Hyperparameter Tuning
↓
Evaluation
↓
Deployment

---

# 2. Data Preprocessing

## Missing Values

- Mean Imputation
- Median Imputation
- Mode Imputation
- KNN Imputer

When to use?

Mean → Normal Distribution

Median → Skewed Data

Mode → Categorical Features

---

## Encoding

### Label Encoding

Used for Ordinal Data

Example

Small → 0

Medium → 1

Large → 2

---

### One Hot Encoding

Used for Nominal Data

Red

Blue

Green

↓

100

010

001

---

## Feature Scaling

### Standardization

x−μ
────
σ

Mean = 0

Std = 1

Used in

- Logistic Regression
- SVM
- PCA
- KNN
- Neural Networks

---

### Normalization

x−min
──────────
max−min

Range

0 to 1

---

# 3. Exploratory Data Analysis

Objectives

✔ Understand Data

✔ Detect Outliers

✔ Check Correlation

✔ Visualize Distribution

Important Plots

- Histogram
- Box Plot
- Heatmap
- Pairplot
- Countplot
- Scatterplot

---

# 4. Statistics

## Mean

Average

---

## Median

Middle Value

---

## Mode

Most Frequent

---

## Variance

Spread of Data

---

## Standard Deviation

√Variance

---

## Covariance

Relationship between two variables

Positive

Negative

Zero

---

## Correlation

Range

-1 to +1

Pearson

Spearman

---

# 5. Feature Engineering

✔ Missing Value Handling

✔ Feature Scaling

✔ Feature Selection

✔ Feature Extraction

✔ Encoding

✔ Polynomial Features

---

## Feature Selection

Methods

- Correlation
- Chi-square
- ANOVA
- Mutual Information
- Recursive Feature Elimination

---

# 6. Supervised Learning

## Linear Regression

Used for Regression

Equation

y = mx + c

Loss Function

MSE

Evaluation

RMSE

MAE

R²

Assumptions

- Linear Relationship
- No Multicollinearity
- Homoscedasticity
- Normal Residuals

---

## Logistic Regression

Used for Classification

Sigmoid Function

1
────────
1+e^-x

Loss

Binary Cross Entropy

Pros

✔ Fast

✔ Interpretable

Cons

❌ Linear Boundary

---

## KNN

Idea

Nearest Neighbors vote.

Pros

- No Training

Cons

- Slow Prediction

Needs Scaling

---

## Decision Tree

Split using

Entropy

Information Gain

Gini Index

Pros

✔ Explainable

Cons

Overfitting

---

## Random Forest

Collection of Decision Trees

Advantages

✔ Less Overfitting

✔ High Accuracy

---

## Support Vector Machine

Goal

Maximum Margin

Kernel Trick

- Linear
- Polynomial
- RBF

Needs Feature Scaling

---

## Naive Bayes

Based on Bayes Theorem

Assumes Feature Independence

Best for

- Text Classification
- Spam Detection

---

# 7. Unsupervised Learning

## K-Means

Objective

Minimize WCSS

Steps

Initialize Centroids

↓

Assign Clusters

↓

Update Centroids

↓

Repeat

Choosing K

Elbow Method

Silhouette Score

---

## Hierarchical Clustering

Agglomerative

Divisive

Visualization

Dendrogram

---

## DBSCAN

Density Based Clustering

Advantages

✔ Detects Noise

✔ Arbitrary Shape

---

# 8. Model Evaluation

Regression

- MAE
- MSE
- RMSE
- R²

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Confusion Matrix

TP

TN

FP

FN

Precision

TP/(TP+FP)

Recall

TP/(TP+FN)

F1

2PR/(P+R)

---

# 9. Cross Validation

K Fold

Stratified K Fold

Leave One Out

Advantages

✔ Better Generalization

✔ Less Overfitting

---

# 10. Hyperparameter Tuning

Methods

Grid Search

Random Search

Bayesian Optimization

Libraries

GridSearchCV

RandomizedSearchCV

Optuna

---

# 11. Ensemble Learning

Bagging

↓

Random Forest

Boosting

↓

AdaBoost

Gradient Boosting

XGBoost

LightGBM

CatBoost

Stacking

Voting

---

# 12. PCA

Purpose

Reduce Dimensions

Benefits

✔ Faster Training

✔ Noise Reduction

✔ Visualization

Needs Feature Scaling

---

# 13. NLP Basics

Text Cleaning

Tokenization

Stopword Removal

Stemming

Lemmatization

Vectorization

Bag of Words

TF-IDF

Word2Vec

Embeddings

---

# 14. Deep Learning Basics

Neuron

Activation Functions

- ReLU
- Sigmoid
- Tanh
- Softmax

Optimizer

- SGD
- Adam
- RMSProp

Loss Functions

- BCE
- Categorical Cross Entropy
- MSE

---

# 15. Common Interview Questions

## Difference

Linear vs Logistic Regression

Decision Tree vs Random Forest

Bagging vs Boosting

L1 vs L2 Regularization

Bias vs Variance

Precision vs Recall

Standardization vs Normalization

PCA vs Feature Selection

Entropy vs Gini

---

# 16. Mathematical Formulas

Linear Regression

y = mx + c

MSE

Σ(y-ŷ)²/n

MAE

Σ|y-ŷ|/n

RMSE

√MSE

R²

1 - RSS/TSS

Sigmoid

1/(1+e^-x)

Softmax

e^x/Σe^x

Cross Entropy

-y log(p)

---

# Common Python Libraries

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

---

# Project Structure

```
Machine-Learning/
│
├── Linear Regression
├── Logistic Regression
├── Decision Trees
├── Random Forest
├── PCA
├── Feature Engineering
├── EDA
├── Model Evaluation
├── Hyperparameter Tuning
└── README.md
```

---

# Interview Tips

✔ Explain intuition before mathematics.

✔ Mention assumptions of every algorithm.

✔ Always discuss advantages and disadvantages.

✔ Know evaluation metrics.

✔ Understand when NOT to use an algorithm.

✔ Be able to compare algorithms.

✔ Practice implementing algorithms from scratch.



---

⭐ This repository serves as a complete Machine Learning revision guide for coding interviews, placements, and technical interviews.
