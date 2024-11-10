import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from forest import Forest
from forest_xg import Forest_XG
from forest_gbm import Forest_LGB
from DataMolder import DataMolder
from imblearn.over_sampling import RandomOverSampler
import joblib  # For saving/loading model results


# # Initialize DataMolder Class with input and output file paths
# data_molder = DataMolder("cleanedData.csv", "updatedCleanedData.csv")

# # Load, prepare, and save the data
# data_molder.load_data()  # Removed the parameter, it should load from self.input_filepath
# if data_molder.dataset is not None:  # Check if the data was loaded successfully
#     data_molder.prepare_data()
#     data_molder.save_data()
#     data_molder.display_data()  # Optional to see the first few rows of the processed data
# else:
#     print("Data loading failed; preparation will not proceed.")


# # Initialize the Forest class
# forest = Forest(n_estimators=200)

# # Load data
# forest.load_data('updatedCleanedData.csv')

# # Perform K-Fold Cross-Validation
# mean_accuracy, std_accuracy = forest.k_fold_cv(n_splits=5)
# print(f"Mean Accuracy: {mean_accuracy:.2f}, Standard Deviation: {std_accuracy:.2f}")

# # Split the data and handle class distribution and random over-sampling
# forest.split_data(test_size=0.2)

# # Define the parameter grid for hyperparameter tuning
# param_distributions = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_features': ['sqrt', 'log2'],  
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4, 6, 8],
#     'bootstrap': [True, False]
# }

# # Perform hyperparameter tuning
# best_params, best_score = forest.hyperparameter_tuning(param_distributions, n_iter=10, cv=3)
# print("Best Parameters:", best_params)
# print("Best Cross-Validation Score:", best_score)

# # Train the model and evaluate
# forest.train()  # Ensure that random over-sampling has been applied in split_data
# y_pred = forest.predict()
# accuracy, report, cm = forest.evaluate(y_pred)

# # Print evaluation results
# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:")
# print(report)
# print("Confusion Matrix:")
# print(cm)

# # Store the evaluation results
# results = {
#     'accuracy': accuracy,
#     'classification_report': report,
#     'confusion_matrix': cm,
#     'predictions': y_pred
# }

# Initialize the Forest_LGB class
forest = Forest_LGB(n_estimators=200)

# Load data
forest.load_data('updatedCleanedData.csv')

# Perform K-Fold Cross-Validation
mean_accuracy, std_accuracy = forest.k_fold_cv(n_splits=5)
print(f"Mean Accuracy: {mean_accuracy:.2f}, Standard Deviation: {std_accuracy:.2f}")

# Split the data and handle class distribution and random over-sampling
forest.split_data(test_size=0.2)

# Define the parameter grid for hyperparameter tuning specific to LGBM
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [-1, 10, 20, 30, 40],  # LightGBM uses -1 for no limit
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 63, 127, 255],  # Important parameter for LightGBM
    'subsample': [0.5, 0.75, 1.0],
    'colsample_bytree': [0.5, 0.75, 1.0],
    'min_child_samples': [20, 40, 60]  # Minimum number of data needed in a child
}

# Perform hyperparameter tuning
best_params, best_score = forest.hyperparameter_tuning(param_distributions, n_iter=10, cv=3)
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Train the model and evaluate
forest.train()  # Ensure that SMOTE has been applied in split_data
y_pred = forest.predict()
accuracy, report, cm = forest.evaluate(y_pred)

# Print evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

# Store the evaluation results
results = {
    'accuracy': accuracy,
    'classification_report': report,
    'confusion_matrix': cm,
    'predictions': y_pred
}