import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime

# Load dataset
data = pd.read_csv("/home/josh/PredictiveAXIOM/SVM/CSV_Data/original_data.csv")

# Parse datetime and calculate time spent
data['First Seen'] = pd.to_datetime(data['First Seen'])
data['Last Seen'] = pd.to_datetime(data['Last Seen'])
data['Date'] = data['First Seen'].dt.date
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
transition_counts = pd.crosstab(data['Current_Room'], data['Next_Room'])
transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

def get_transition_features(row, transition_probabilities):
    current_room = row['Current_Room']
    if current_room in transition_probabilities.index:
        return transition_probabilities.loc[current_room].values
    return np.zeros(len(transition_probabilities.columns))

transition_features = data.apply(
    lambda row: get_transition_features(row, transition_probabilities), axis=1
)

transition_feature_columns = [f'Transition_Prob_{col}' for col in transition_probabilities.columns]
transition_features_df = pd.DataFrame(transition_features.tolist(), columns=transition_feature_columns)
data = pd.concat([data.reset_index(drop=True), transition_features_df], axis=1)

# Prepare for date-based predictions
unique_dates = sorted(data['Date'].unique())
results = []

# Initialize categorical and continuous feature transformers
categorical_features = ['Location', 'Time_of_Day', 'Day_of_Week']
continuous_features = ['Time_Spent'] + transition_feature_columns

# Update OneHotEncoder to handle unknown categories
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
continuous_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', continuous_transformer, continuous_features)
    ]
)

# Encode target
label_encoder = LabelEncoder()
data['Next_Room_Encoded'] = label_encoder.fit_transform(data['Next_Room'])

# Define LinearSVC model
svm = LinearSVC(max_iter=10000, class_weight='balanced', dual=False)

# Loop through each unique date
for i, target_date in enumerate(unique_dates[1:], start=1):
    # Train on data from all previous dates
    train_data = data[data['Date'] < target_date]
    test_data = data[data['Date'] == target_date]
    
    if train_data.empty or test_data.empty:
        continue

    X_train = train_data[['Location', 'Time_of_Day', 'Day_of_Week', 'Time_Spent'] + transition_feature_columns]
    y_train = train_data['Next_Room_Encoded']
    
    X_test = test_data[['Location', 'Time_of_Day', 'Day_of_Week', 'Time_Spent'] + transition_feature_columns]
    y_test = test_data['Next_Room_Encoded']

    # Build pipeline and train model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('svm', svm)])
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save results
    report = classification_report(
        y_test, y_pred, 
        labels=np.unique(y_test),  # Include only classes present in y_test
        target_names=label_encoder.inverse_transform(np.unique(y_test)),  # Map back to class names
        output_dict=True,
        zero_division=0  # Suppress undefined metric warnings
    )
    results.append({
        'Date': target_date,
        'Accuracy': accuracy,  # Use explicitly calculated accuracy
        'Classification Report': report,
        'Confusion Matrix': confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    })

# Save results for analysis
for result in results:
    with open("predictions_output.txt", "a") as file:
        file.write(f"Results for Date: {result['Date']}\n")
        file.write(f"Accuracy: {result['Accuracy']}\n")
        file.write("Classification Report:\n")
        file.write(pd.DataFrame(result['Classification Report']).transpose().to_string() + "\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(result['Confusion Matrix']) + "\n\n")
    print(f"Results for Date: {result['Date']}")
    print(f"Accuracy: {result['Accuracy']}")
    print("Classification Report:")
    print(pd.DataFrame(result['Classification Report']).transpose())
    print("Confusion Matrix:")
    print(result['Confusion Matrix'])
