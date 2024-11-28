import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import threading
import time

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

# Split features and target
X = data[['Location', 'Time_of_Day', 'Day_of_Week', 'Time_Spent']]
y = data['Next_Room']

# Encoding categorical and continuous features
categorical_features = ['Location', 'Time_of_Day', 'Day_of_Week']
categorical_transformer = OneHotEncoder()

continuous_features = ['Time_Spent']
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

# Define SVM model
svm = SVC(kernel='rbf', probability=True, class_weight='balanced')

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('svm', svm)])

# Hyperparameter tuning
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svm__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
