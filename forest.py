import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

class Forest:
    def __init__(self, n_estimators=100, random_state=42, class_weight=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, class_weight=class_weight, n_jobs=-1)

    def load_data(self, filepath):
        """Load data from the specified CSV file and prepare features and target."""
        try:
            self.dataset = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: The file {filepath} was not found.")
            return
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
            return
        except pd.errors.ParserError:
            print("Error: The file could not be parsed.")
            return
        
        # Check for NaN values in the dataset after loading
        if self.dataset.isnull().values.any():
            print("Warning: NaN values detected in the dataset. Filling NaNs with 0.")
            self.dataset.fillna(0, inplace=True)  # Handle NaN values before processing

        # Define features (X) and target (y)
        self.X = self.dataset[['Time_Spent', 
                        'Current_Room_Bedroom', 
                        'Current_Room_Deck', 
                        'Current_Room_Den', 
                        'Current_Room_Front Door', 
                        'Current_Room_Garage', 
                        'Current_Room_Kitchen', 
                        'Current_Room_Living Room', 
                        'Current_Room_Stairway', 
                        'Current_Room_Study', 
                        'Previous Room Time', 
                        'Time of Day', 
                        'Day of Week',
                        '2Prior_Room_0.0',
                        '2Prior_Room_1.0', 
                        '2Prior_Room_2.0',
                        '2Prior_Room_3.0',  
                        '2Prior_Room_4.0',
                        '2Prior_Room_5.0',
                        '2Prior_Room_6.0',
                        '2Prior_Room_7.0',
                        '2Prior_Room_8.0']]

        # Target variable (y)
        self.y = self.dataset[['Next_Room_Bedroom', 
                                'Next_Room_Deck', 
                                'Next_Room_Den', 
                                'Next_Room_Front Door', 
                                'Next_Room_Garage', 
                                'Next_Room_Kitchen', 
                                'Next_Room_Living Room', 
                                'Next_Room_Stairway', 
                                'Next_Room_Study']]

    def split_data(self, test_size=0.2):
        """Split the dataset into training and testing sets and apply SMOTE selectively."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)

        # Convert y_train to single-label if multi-label (for classification)
        self.y_train_labels = np.argmax(self.y_train.values, axis=1)

        # Check for NaN values in the training features
        if self.X_train.isnull().values.any() or np.isnan(self.y_train_labels).any():
            print("Warning: NaN values detected in the training data. Filling NaNs with 0.")
            self.X_train.fillna(0, inplace=True)

        # Create separate datasets for the underrepresented classes
        underrepresented_classes = [1, 4]
        X_resampled, y_resampled = self.X_train.copy(), self.y_train_labels.copy()

        for class_label in underrepresented_classes:
            # Get the indices for the current class
            class_indices = np.where(y_resampled == class_label)[0]
            X_class = self.X_train.iloc[class_indices]
            y_class = y_resampled[class_indices]

            # Apply SMOTE to the class
            smote = SMOTE(random_state=self.random_state)
            X_class_resampled, y_class_resampled = smote.fit_resample(X_class, y_class)

            # Concatenate the resampled data back
            X_resampled = pd.concat([X_resampled, pd.DataFrame(X_class_resampled)], ignore_index=True)
            y_resampled = np.concatenate([y_resampled, y_class_resampled])

        # Update the training set with the resampled data
        self.X_train_resampled, self.y_train_resampled = X_resampled, y_resampled

        print(f"Original dataset shape: {self.y_train_labels.shape}")
        print(f"Resampled dataset shape: {self.y_train_resampled.shape}")

    def train(self):
        """Train the Random Forest model on the training data."""
        self.model.fit(self.X_train_resampled, self.y_train_resampled)

    def predict(self):
        """Make predictions on the test set."""
        return self.model.predict(self.X_test)

    def evaluate(self, y_pred):
        """Evaluate the model's performance."""
        if len(y_pred.shape) == 1:  # If it's 1D
            y_pred_labels = y_pred
        else:
            y_pred_labels = np.argmax(y_pred, axis=1) 

        y_test_labels = np.argmax(self.y_test.values, axis=1)
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        report = classification_report(y_test_labels, y_pred_labels, zero_division=0)
        cm = confusion_matrix(y_test_labels, y_pred_labels)

        return accuracy, report, cm

    def k_fold_cv(self, n_splits=5):
        """Perform K-Fold CV and return average metric"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        accuracies = []

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # Apply SMOTE within K-Fold CV
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, np.argmax(y_train.values, axis=1))

            self.model.fit(X_train_resampled, y_train_resampled)
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(np.argmax(y_test.values, axis=1), y_pred)
            accuracies.append(accuracy)

        return np.mean(accuracies), np.std(accuracies)

    def hyperparameter_tuning(self, param_distributions, n_iter=100, cv=3):
        """Perform Randomized Search Cross-Validation to find optimal hyperparameters."""
        random_search = RandomizedSearchCV(estimator=self.model,
                                           param_distributions=param_distributions,
                                           n_iter=n_iter,
                                           cv=cv,
                                           random_state=self.random_state,
                                           n_jobs=1)  # Run in single-threaded mode

        # Use SMOTE before fitting for hyperparameter tuning
        X_train_resampled, y_train_resampled = SMOTE(random_state=self.random_state).fit_resample(self.X_train, self.y_train_labels)
        
        random_search.fit(X_train_resampled, y_train_resampled)  # Fit the model

        # Update the model to the best estimator found
        self.model = random_search.best_estimator_

        return random_search.best_params_, random_search.best_score_


# This will make it easier to manage the Random Forest implementation and enhance code readability 
#     Data Loading: A method to load data from a CSV file.
#     Data Splitting: A method to split the dataset into training and testing sets.
#     Training: A method to train the Random Forest model.
#     Prediction: A method to make predictions on the test set.
#     Evaluation: A method to evaluate the model's performance and return metrics.
#     Hyperparameter Tuning: A method for tuning hyperparameters using grid search.
