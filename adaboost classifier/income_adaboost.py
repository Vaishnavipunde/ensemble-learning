# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:19:54 2024

@author: rajendra
"""


import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
loan_data = pd.read_csv("c:/2-dataset/income.csv")

# Separate features and target variable
x = loan_data.iloc[:, 0:6]
y = loan_data.iloc[:, 6]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create and train AdaBoostClassifier model
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1)
ada_model.fit(x_train, y_train)

# Make predictions
y_pred = ada_model.predict(x_test)

# Print accuracy
print("AdaBoostClassifier accuracy:", metrics.accuracy_score(y_test, y_pred))

# Initialize Logistic Regression model
lr = LogisticRegression()

# Create AdaBoostClassifier model with Logistic Regression as base estimator
ada_lr_model = AdaBoostClassifier(base_estimator=lr, n_estimators=50, learning_rate=1)
ada_lr_model.fit(x_train, y_train)

# Make predictions
y_pred_lr = ada_lr_model.predict(x_test)

# Print accuracy
print("AdaBoostClassifier with Logistic Regression accuracy:", metrics.accuracy_score(y_test, y_pred_lr))











