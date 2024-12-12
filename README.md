# PredictiveAXIOM
PredictiveAXIOM is a repository designed for exploring advanced machine learning models that predict room transitions based on historical occupancy data. This project leverages two powerful models—Linear Support Vector Classifier (SVC) and Forest_LGB (LightGBM Classifier)—to analyze temporal and spatial patterns in room-to-room movement.
Overview

The repository is designed to address the problem of predicting which room a person is likely to occupy next, based on event-driven occupancy data. This solution is deployed within the context of the Axiom Smart Home configuration developed my Dr. Karl Morris. This model was developed with his permission and supervision. 

The data powering these predictions is a CSV file with fields such as Device ID, Location, First Seen, and Last Seen, which represent the timestamps when a device (or person) was detected in a particular room. This data was provided by Dr. Karl Morris's Axiom Smart Home system, and accumulates to approximately 54 months. 

## Data Flow and Formats
Data Sources

The repository processes data through several transformations:

    testData.csv: The initial raw data, containing timestamps, room locations, and device IDs.
    cleanedData.csv: Derived from testData.csv, this intermediate file includes calculated fields such as time spent in rooms and contextual features like time of day and day of the week.
    updatedCleanedData.csv: The final formatted dataset, created from cleanedData.csv, which includes additional engineered features like one-hot encoded room transitions, interaction terms, and prior room states.

Each model utilizes different data formats depending on their requirements:

    LinearSVC: Operates directly on original_data.csv, a simplified dataset with essential features for classification.
    LGBMClassifier: Requires updatedCleanedData.csv, which undergoes a multi-step data preparation process.

Data Preparation Steps

    Test Data (testData.csv):
        Loaded into the DataMolder class.
        Basic preprocessing includes sorting by timestamps and calculating time spent in each room.
        Outputs cleanedData.csv.

    Cleaned Data (cleanedData.csv):
        Further processed by the DataMolder class to engineer advanced features, such as:
            Room transition encoding.
            Interaction features between time spent and specific rooms.
            Time of day and day of week encodings.
            Prior room states (previous room and two-prior room).
        Outputs updatedCleanedData.csv.

    Updated Cleaned Data (updatedCleanedData.csv):
        Used exclusively by the LGBMClassifier model to train and evaluate predictions.

## Models
Linear SVC

The LinearSVC model operates on original_data.csv, focusing on:

    Simplified feature engineering for fast training and evaluation.
    Room transition probabilities and contextual encodings like time of day.
    Predicting the next room based on a straightforward training pipeline.

Running the Model:

python linear_svc.py

LGBM Classifier

The LGBMClassifier model requires the enriched updatedCleanedData.csv and leverages advanced features like interaction terms and encoded room transitions. Its training process includes:

    Data resampling techniques like ADASYN and SMOTE to handle class imbalances.
    Grid search for hyperparameter tuning.
    Evaluation using K-Fold cross-validation and detailed metrics.

Running the Model:

python lgbm_classifier.py

Repository Structure

Repository Structure

    Data Files:
        testData.csv: Initial raw data.
        cleanedData.csv: Intermediate processed data.
        updatedCleanedData.csv: Final dataset for the LGBM model.
        original_data.csv: Dataset for the LinearSVC model.

    Scripts:
        DataMolder.py: Handles the transformation from testData.csv to updatedCleanedData.csv.
        linear_svc.py: Implements the LinearSVC model.
        lgbm_classifier.py: Implements the LGBMClassifier model.
        forest_lgb.py: Contains the Forest_LGB class, the core implementation for the LGBM model.

Usage

    Ensure all data files are located in the expected directories.
    Run the DataMolder script to prepare datasets:

python DataMolder.py

Train and evaluate the desired model:

    For LinearSVC:

python linear_svc.py

For LGBMClassifier:

python lgbm_classifi

Use Cases

    Smart home automation and resource planning.
    Occupancy analytics for commercial spaces.
    Predictive systems for dynamic environments.
