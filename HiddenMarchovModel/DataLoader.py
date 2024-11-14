import pandas as pd
from sklearn.model_selection import train_test_split
from TimeBinner import TimeBinner
import numpy as np

class DataLoader:
    def __init__(self):
        """Initialize with attributes for storing raw and processed data."""
        self.data = None
        self.processed_data = None

    def load_data(self, file_path):
        """Load data from a CSV file and store it in the `data` attribute."""
        try:
            self.data = pd.read_csv(file_path)
            print("Data successfully loaded.")
        except Exception as e:
            print(f"Error loading data: {e}")
        return self.data

    def format_data(self, time_binner, bin_size=15):
        """
        Clean and structure data for the HMM.
        Uses TimeBinner to bin timestamps and create emissions.
        """
        # Check if data is loaded
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        # Ensure required columns are present
        required_columns = {'Device ID', 'Location', 'First Seen', 'Last Seen'}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"Data is missing required columns: {required_columns}")

        # Ensure room codes mapping is applied correctly
        room_codes = {'Study': 1, 'Stairway': 2, 'Living Room': 3, 'Kitchen': 4, 'Den': 5, 'Bedroom': 6, 'Deck': 7, 'Garage': 8, 'Front Door': 9}
        self.data['Location'] = self.data['Location'].map(room_codes)
        
        # After mapping, check for unmapped (NaN) values due to any mismatches
        unmapped = self.data['Location'].isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} entries could not be mapped to room codes. Check room names.")

        # Apply time binning to create processed sequences
        time_binner = TimeBinner(bin_size=bin_size)
        self.processed_data, self.sequence_lengths = time_binner.apply_binning(self.data)  # Store sequence_lengths here
        
        print("Data successfully formatted for model.")
        return self.processed_data, self.sequence_lengths

    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Split processed data and sequence lengths into training and test sets.
        Assumes `processed_data` and `sequence_lengths` have been created.
        """
        if self.processed_data is None or self.sequence_lengths is None:
            raise ValueError("Data has not been formatted. Please format data first.")

        # Generate sequence indices to split at the sequence level
        num_sequences = len(self.sequence_lengths)
        indices = np.arange(num_sequences)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        # Split processed_data based on sequence_lengths
        train_data = []
        test_data = []
        train_lengths = [self.sequence_lengths[i] for i in train_idx]
        test_lengths = [self.sequence_lengths[i] for i in test_idx]

        # Track the start position in self.processed_data
        current_position = 0
        for i, length in enumerate(self.sequence_lengths):
            if i in train_idx:
                train_data.extend(self.processed_data[current_position:current_position + length])
            elif i in test_idx:
                test_data.extend(self.processed_data[current_position:current_position + length])
            current_position += length

        print("Data successfully split into training and test sets.")
        
        # Convert lists to numpy arrays
        return np.array(train_data), np.array(test_data), train_lengths, test_lengths