import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import threading
import time

# Debug message function
def print_debug_message():
    while True:
        time.sleep(600)  # Wait for 600 seconds (10 minutes)
        print("Program still running...")

# Start the debug thread
debug_thread = threading.Thread(target=print_debug_message, daemon=True)
debug_thread.start()

# Load dataset
data = pd.read_csv("/home/josh/PredictiveAXIOM/SVM/CSV_Data/original_data.csv")

# Parse datetime and calculate time spent
data['First Seen'] = pd.to_datetime(data['First Seen'])
data['Last Seen'] = pd.to_datetime(data['Last Seen'])
data['Time_Spent'] = (data['Last Seen'] - data['First Seen']).dt.total_seconds()

# Extract time of day and day of week
data['Time_of_Day'] = pd.cut(data['First Seen'].dt.hour, 
                             bins=[0, 6, 12, 18, 24], 
                             labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                             right=False)
data['Day_of_Week'] = data['First Seen'].dt.day_name()

# Prioritize rooms
room_counts = data['Location'].value_counts()
top_rooms = room_counts.index[:7]
data['Location'] = data['Location'].apply(lambda x: x if x in top_rooms else 'Other')

# Shift to create target variable
data['Next_Room'] = data['Location'].shift(-1)
data = data.dropna()

# Establish Transition Probabilities
data['Current_Room'] = data['Location']  # Duplicate column for clarity
# Compute transition counts
transition_counts = pd.crosstab(data['Current_Room'], data['Next_Room'])
# Normalize counts to probabilities
transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0)
print("Transition Probability Matrix:")
print(transition_probabilities)

# Identify most likely next room for each room
most_likely_rooms = transition_probabilities.idxmax(axis=1)
print("\nMost Likely Room Transitions:")
print(most_likely_rooms)

# Prepare Transition Probabilities as Features
def get_transition_features(row, transition_probabilities):
    """Get transition probabilities for the current room."""
    current_room = row['Current_Room']
    if current_room in transition_probabilities.index:
        return transition_probabilities.loc[current_room].values
    else:
        # Default to zeros if the room is not in the matrix
        return np.zeros(len(transition_probabilities.columns))

# Apply the transition probabilities to the dataset
transition_probabilities = transition_probabilities.fillna(0)  # Fill missing probabilities with 0
transition_features = data.apply(
    lambda row: get_transition_features(row, transition_probabilities), axis=1
)

# Add transition probabilities to the dataset as separate columns
transition_feature_columns = [f'Transition_Prob_{col}' for col in transition_probabilities.columns]
transition_features_df = pd.DataFrame(transition_features.tolist(), columns=transition_feature_columns)
data = pd.concat([data.reset_index(drop=True), transition_features_df], axis=1)

# Updated feature set
X = data[['Location', 'Time_of_Day', 'Day_of_Week', 'Time_Spent'] + transition_feature_columns]
y = data['Next_Room']

# Encoding categorical and continuous features
categorical_features = ['Location', 'Time_of_Day', 'Day_of_Week']
categorical_transformer = OneHotEncoder()

continuous_features = ['Time_Spent'] + transition_feature_columns
continuous_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', continuous_transformer, continuous_features)
    ]
)

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define LinearSVC model
svm = LinearSVC(max_iter=10000, class_weight='balanced', dual=False)

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('svm', svm)])

# Parameter grid for LinearSVC
param_grid = {
    'svm__C': [0.1, 1, 10, 100]
}

# Grid search with LinearSVC
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
