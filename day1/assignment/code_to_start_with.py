# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd

data = pd.read_csv(r'D:\SummerCampProgram\aiSummerCamp2025-1\day1\assignment\data\train.csv')
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
df.sample(10)
# %% 
# separate the features and labels
y = df['Survived']
X = df.drop(columns=['Survived'])
# %%
# train-test split
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('X_train: {}'.format(np.shape(X_train)))
print('X_test: {}'.format(np.shape(X_test)))
print('y_train: {}'.format(np.shape(y_train)))
print('y_test: {}'.format(np.shape(y_test)))
# %%
# build model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Initialize models
svm_model = SVC(kernel='rbf', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# %%
# predict and evaluate
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\nEvaluation Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Evaluate SVM
print("\n" + "="*50)
print("SVM Model Evaluation")
print("="*50)
evaluate_model(svm_model, X_test, y_test)

# Evaluate KNN
print("\n" + "="*50)
print("KNN Model Evaluation")
print("="*50)
evaluate_model(knn_model, X_test, y_test)

# Evaluate Random Forest
print("\n" + "="*50)
print("Random Forest Model Evaluation")
print("="*50)
evaluate_model(rf_model, X_test, y_test)

# %%
# Feature Importance for Random Forest
import matplotlib.pyplot as plt

feature_importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

# %%
# Cross-validation for better evaluation
from sklearn.model_selection import cross_val_score

print("\nCross-Validation Scores:")
models = [('SVM', svm_model), ('KNN', knn_model), ('Random Forest', rf_model)]
for name, model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f} (±{scores.std():.4f})")