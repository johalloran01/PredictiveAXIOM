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

    def format_data(self, time_binner, transition_matrix, bin_size=15):
        """
        Clean and structure data for the HMM.
        Uses TimeBinner to bin timestamps and create emissions.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        # Ensure required columns are present
        required_columns = {'Device ID', 'Location', 'First Seen', 'Last Seen'}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"Data is missing required columns: {required_columns}")

        # Map room names to numerical codes
        room_codes = {
            'Study': 1, 'Stairway': 2, 'Living Room': 3, 'Kitchen': 4,
            'Den': 5, 'Bedroom': 6, 'Deck': 7, 'Garage': 8, 'Front Door': 9
        }
        self.data['Location'] = self.data['Location'].map(room_codes)

        # Validate the completeness of the mapping
        unmapped_rooms = self.data[self.data['Location'].isna()]['Location'].unique()
        if unmapped_rooms.size > 0:
            print(f"Warning: {unmapped_rooms.size} room names could not be mapped:")
            print(f"Unmapped room names: {unmapped_rooms}")

        # Apply time binning with transition matrix
        print("Formatting data...")
        print(f"Data sample before formatting:\n{self.data.head()}")
        self.processed_data, self.sequence_lengths = time_binner.apply_binning(self.data, transition_matrix)
        print("Data successfully formatted.")
        print(f"Processed data sample:\n{self.processed_data[:5]}")
        print(f"Sequence lengths: {self.sequence_lengths[:5]}")

        # Validate sequence lengths
        if any(length <= 0 for length in self.sequence_lengths):
            raise ValueError("Sequence lengths contain non-positive values. Check the binning logic.")

        return self.processed_data, self.sequence_lengths

    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Efficiently split processed data and sequence lengths into training and test sets.
        """
        if self.processed_data is None or self.sequence_lengths is None:
            raise ValueError("Data has not been formatted. Please format data first.")

        # Generate sequence indices to split at the sequence level
        num_sequences = len(self.sequence_lengths)
        indices = np.arange(num_sequences)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        # Use indices to slice processed_data and sequence_lengths
        train_lengths = [self.sequence_lengths[i] for i in train_idx]
        test_lengths = [self.sequence_lengths[i] for i in test_idx]

        # Efficiently construct train and test data
        train_data = []
        test_data = []

        current_position = 0
        for i, length in enumerate(self.sequence_lengths):
            if i in train_idx:
                train_data.extend(self.processed_data[current_position:current_position + length])
            elif i in test_idx:
                test_data.extend(self.processed_data[current_position:current_position + length])
            current_position += length

        print("Data successfully split into training and test sets.")
        print(f"Train data size: {len(train_data)}, Train lengths: {len(train_lengths)}")
        print(f"Test data size: {len(test_data)}, Test lengths: {len(test_lengths)}")

        # Convert to numpy arrays
        return np.array(train_data), np.array(test_data), train_lengths, test_lengths
