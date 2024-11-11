import pandas as pd
import numpy as np

class HMM_DataMolder:
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def load_and_prepare_data(self):
        """Load and format data from the input CSV."""
        # Load the original data
        data = pd.read_csv(self.input_filepath)

        # Ensure that the data is sorted by timestamp
        data['First Seen'] = pd.to_datetime(data['First Seen'])
        data = data.sort_values(by='First Seen')

        # Map locations to numerical states for the HMM
        location_mapping = {location: idx for idx, location in enumerate(data['Location'].unique())}
        data['Location_Code'] = data['Location'].map(location_mapping)

        # Save the formatted data for later use
        data.to_csv(self.output_filepath, index=False)
        print("Data formatted and saved.")
        print("Location Mapping:", location_mapping)

        return data, location_mapping