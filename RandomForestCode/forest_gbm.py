import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.over_sampling import SMOTE

class Forest_LGB:
    def __init__(self, n_estimators=100, random_state=42, class_weight=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        # Using LightGBM's model instead of XGBoost or RandomForest
        self.model = lgb.LGBMClassifier(n_estimators=self.n_estimators, 
                                        random_state=self.random_state, 
                                        class_weight=class_weight)

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

        # Define features (X) including interaction features
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
                '2Prior_Room_8.0',
                # Add interaction features here
                'Time_Spent_Current_Room_Bedroom',
                'Time_Spent_Current_Room_Deck',
                'Time_Spent_Current_Room_Den',
                'Time_Spent_Current_Room_Front Door',
                'Time_Spent_Current_Room_Garage',
                'Time_Spent_Current_Room_Kitchen',
                'Time_Spent_Current_Room_Living Room',
                'Time_Spent_Current_Room_Stairway',
                'Time_Spent_Current_Room_Study']]

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

    def apply_adasyn(self, X_train, y_train):
        """Apply ADASYN to balance the training dataset."""
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        print("Applied ADASYN. Class distribution after resampling:", np.bincount(y_resampled))
        return X_resampled, y_resampled

    def split_data(self, test_size=0.2):
        """Split the dataset into training and testing sets and apply ADASYN or fallback methods with conditional feature removal."""
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )

        # Convert y_train to single-label if multi-label (for classification)
        self.y_train_labels = np.argmax(self.y_train.values, axis=1)

        # Initial attempt to apply ADASYN for oversampling
        try:
            self.X_train_resampled, self.y_train_labels_resampled = self.apply_adasyn(
                self.X_train, self.y_train_labels
            )
            print("Applied ADASYN successfully.")
        except ValueError as e:
            print(f"ADASYN failed: {e}. Attempting RandomOverSampler as a fallback.")
            
            # Fallback to RandomOverSampler if ADASYN fails
            ros = RandomOverSampler(random_state=self.random_state)
            try:
                self.X_train_resampled, self.y_train_labels_resampled = ros.fit_resample(
                    self.X_train, self.y_train_labels
                )
                print("Applied RandomOverSampler successfully. Class distribution after resampling:", np.bincount(self.y_train_labels_resampled))
            except ValueError as e:
                print(f"RandomOverSampler also failed: {e}. Proceeding to remove problematic classes.")

                # Identify underrepresented classes and remove them from training
                class_counts = np.bincount(self.y_train_labels)
                min_threshold = 1  # Arbitrary threshold for "sufficient" class data
                sufficient_classes = [i for i, count in enumerate(class_counts) if count > min_threshold]
                insufficient_classes = [i for i, count in enumerate(class_counts) if count <= min_threshold]

                # Print information about dropped classes
                if insufficient_classes:
                    print("Classes dropped due to insufficient data:", insufficient_classes)

                # Filter out instances that belong to insufficient classes
                mask = np.isin(self.y_train_labels, sufficient_classes)
                self.X_train_resampled = self.X_train[mask]
                self.y_train_labels_resampled = self.y_train_labels[mask]

                print("Classes retained for training:", sufficient_classes)
                print("Class distribution after filtering:", np.bincount(self.y_train_labels_resampled))

        # Check for NaN values in the training features
        if self.X_train_resampled.isnull().values.any() or np.isnan(self.y_train_labels_resampled).any():
            print("Warning: NaN values detected in the training data. Filling NaNs with 0.")
            self.X_train_resampled.fillna(0, inplace=True)

        # Print final dataset shape
        print(f"Original dataset shape: {self.y_train_labels.shape}")
        print(f"Resampled dataset shape: {len(self.y_train_labels_resampled)}")

    def train(self):
        """Train the LightGBM model on the training data."""
        self.model.fit(self.X_train, self.y_train_labels)

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

            self.model.fit(X_train, np.argmax(y_train.values, axis=1))
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

        random_search.fit(self.X_train, self.y_train_labels)  # Fit the model

        # Update the model to the best estimator found
        self.model = random_search.best_estimator_

        return random_search.best_params_, random_search.best_score_
