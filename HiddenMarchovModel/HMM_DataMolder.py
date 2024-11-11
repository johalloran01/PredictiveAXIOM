import pandas as pd
import numpy as np
from sklearn.utils import resample

class HMM_DataMolder:
    def __init__(self, input_filepath, output_filepath, min_instance_threshold=3000):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.min_instance_threshold = min_instance_threshold

    def load_and_prepare_data(self):
        # Load the original data
        data = pd.read_csv(self.input_filepath)

        # Ensure that the data is sorted by timestamp
        data['First Seen'] = pd.to_datetime(data['First Seen'])
        data['Last Seen'] = pd.to_datetime(data['Last Seen'])  # Ensure 'Last Seen' is datetime
        data = data.sort_values(by='First Seen')

        # Add 'Time of Day' column using a granular mapping
        data['Hour'] = data['First Seen'].dt.hour
        data['Time of Day'] = data['Hour'].apply(self._map_time_of_day)
        data.drop(columns=['Hour'], inplace=True)

        # Add 'Weekday Indicator' column
        data['Weekday_Indicator'] = data['First Seen'].dt.weekday.apply(lambda x: 1 if x < 5 else 0)

        # Add 'Duration' column for time spent in a room
        data['Duration'] = (data['Last Seen'] - data['First Seen']).dt.total_seconds()

        # Map locations to numerical states
        location_mapping = {location: idx for idx, location in enumerate(data['Location'].unique())}
        data['Location_Code'] = data['Location'].map(location_mapping)

        # Create columns for the previous 3 room transitions (sequence context)
        for i in range(1, 4):
            data[f'Prev_Location_{i}'] = data['Location_Code'].shift(i)

        # Drop rows with NaN values created by shifting
        data = data.dropna(subset=['Prev_Location_1', 'Prev_Location_2', 'Prev_Location_3'])
        data = data.astype({'Location_Code': int, 'Prev_Location_1': int, 'Prev_Location_2': int, 'Prev_Location_3': int})

        # Balance the data by resampling
        balanced_data = self._balance_data(data, location_mapping)

        # Ensure the balanced data is sorted by 'First Seen' before saving
        balanced_data = balanced_data.sort_values(by='First Seen')

        # Save the formatted data for later use
        balanced_data.to_csv(self.output_filepath, index=False)
        print("Balanced data formatted and saved.")
        
        # Print a quick validation check for order
        print("First 5 timestamps after processing:", balanced_data['First Seen'].head())
        print("Last 5 timestamps after processing:", balanced_data['First Seen'].tail())
        
        return balanced_data, location_mapping
    
    def _map_time_of_day(self, hour):
        """Map hour to a more granular time of day category."""
        if 6 <= hour < 9:
            return 0  # Early Morning
        elif 9 <= hour < 12:
            return 1  # Late Morning
        elif 12 <= hour < 15:
            return 2  # Early Afternoon
        elif 15 <= hour < 18:
            return 3  # Late Afternoon
        elif 18 <= hour < 21:
            return 4  # Early Evening
        elif 21 <= hour < 24:
            return 5  # Late Evening
        else:
            return 6  # Night

    def _balance_data(self, data, location_mapping):
        """Balance the dataset by oversampling underrepresented classes."""
        balanced_data = pd.DataFrame()
        for location, code in location_mapping.items():
            location_data = data[data['Location_Code'] == code]
            
            if len(location_data) >= self.min_instance_threshold:
                balanced_data = pd.concat([balanced_data, location_data], axis=0)
            elif len(location_data) > 0:
                resampled_data = resample(
                    location_data,
                    replace=True,
                    n_samples=self.min_instance_threshold,
                    random_state=42
                )
                balanced_data = pd.concat([balanced_data, resampled_data], axis=0)
                print(f"Location {location} (code {code}) was resampled to {self.min_instance_threshold} instances.")
            else:
                print(f"Location {location} (code {code}) has no data and was excluded.")

        return balanced_data
