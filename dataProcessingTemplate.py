import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from forest import Forest
from DataMolder import DataMolder
import joblib  # For saving/loading model results


# # Initialize DataMolder Class with input and output file paths
# data_molder = DataMolder("testData.csv", "updatedCleanedData.csv")

# # Load, prepare, and save the data
# data_molder.load_data()
# data_molder.prepare_data()
# data_molder.save_data()
# data_molder.display_data()  # Optional to see the first few rows of the processed data


# Initialize the Forest class
forest = Forest(n_estimators=200)
forest.load_data('updatedCleanedData.csv')

# Perform K-Fold Cross-Validation
mean_accuracy, std_accuracy = forest.k_fold_cv(n_splits=5)
print(f"Mean Accuracy: {mean_accuracy:.2f}, Standard Deviation: {std_accuracy:.2f}")

# Split the data and perform hyperparameter tuning
forest.split_data(test_size=0.2)

# Define the parameter grid for hyperparameter tuning
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2'],  
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'bootstrap': [True, False]
}

# Perform hyperparameter tuning
best_params, best_score = forest.hyperparameter_tuning(param_distributions, n_iter=10, cv=3)
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Train the model and evaluate
forest.train()
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

# Save results to a file for later use
#joblib.dump(results, 'model_results.pkl')