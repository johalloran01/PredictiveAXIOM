import pandas as pd

class DataMolder:
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath = input_filepath  # Input CSV file with initial format
        self.output_filepath = output_filepath  # Output CSV file with desired format
        self.dataset = None

    def load_data(self):
        """Load the input dataset from the specified CSV file."""
        try:
            self.dataset = pd.read_csv(self.input_filepath)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file {self.input_filepath} was not found.")
            return
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
            return
        except pd.errors.ParserError:
            print("Error: The file could not be parsed.")
            return

    def prepare_data(self):
        """Prepare the dataset by adding Previous Room, Next Room, Time of Day, Day of Week, time spent in the previous room, and 2Prior room."""
        # Sort by 'First Seen'
        self.dataset.sort_values(by='First Seen', inplace=True)

        # Calculate Time Spent
        self.dataset['Time_Spent'] = (pd.to_datetime(self.dataset['Last Seen']) - pd.to_datetime(self.dataset['First Seen'])).dt.total_seconds()

        # One-hot encode the current Location column
        current_room_encoded = pd.get_dummies(self.dataset['Location'], prefix='Current_Room')

        # Add Previous Room column
        self.dataset['Previous Room'] = self.dataset['Location'].shift(1)

        # Add Next Room column
        self.dataset['Next Room'] = self.dataset['Location'].shift(-1)

        # One-hot encode the Previous Room and Next Room columns
        prev_room_encoded = pd.get_dummies(self.dataset['Previous Room'], prefix='Prev_Room')
        next_room_encoded = pd.get_dummies(self.dataset['Next Room'], prefix='Next_Room')

        # Add 2Prior room (the room before the previous room)
        self.dataset['2Prior Room'] = self.dataset['Previous Room'].shift(1)  # Room 2 ago

        # Map 2Prior Room to numeric codes
        room_mapping = {
            'Living Room': 1,
            'Kitchen': 2,
            'Bedroom': 3,
            'Den': 4,
            'Garage': 5,
            'Deck': 6,
            'Study': 7,
            'Front Door': 8,
            None: 0  # Use 0 for no room
        }

        # Apply mapping to the 2Prior Room and fill NaN values with 0
        self.dataset['2Prior Room'] = self.dataset['2Prior Room'].map(room_mapping).fillna(0)

        # One-hot encode the 2Prior Room column
        two_prior_room_encoded = pd.get_dummies(self.dataset['2Prior Room'], prefix='2Prior_Room')

        # Concatenate the encoded columns back to the dataset
        self.dataset = pd.concat([self.dataset, current_room_encoded, prev_room_encoded, next_room_encoded, two_prior_room_encoded], axis=1)

        # Create Time of Day column
        self.dataset['Time of Day'] = pd.to_datetime(self.dataset['First Seen']).dt.hour

        # Create Day of the Week column
        self.dataset['Day of Week'] = pd.to_datetime(self.dataset['First Seen']).dt.dayofweek  # 0=Monday, 6=Sunday

        # Add Time Spent in Previous Room
        self.dataset['Previous Room Time'] = self.dataset['Time_Spent'].shift(1)  # Time spent in the previous room

        # Handle missing values for Previous Room Time
        self.dataset['Previous Room Time'].fillna(0, inplace=True)

        # Create interaction features
        for room in current_room_encoded.columns:
            # Create a new feature representing the interaction between Time_Spent and each room
            interaction_feature = f'Time_Spent_{room}'
            self.dataset[interaction_feature] = self.dataset['Time_Spent'] * self.dataset[room]

        # Drop 'First Seen', 'Last Seen', 'Device ID', and 'Location' columns if not needed
        self.dataset = self.dataset.drop(columns=['First Seen', 'Last Seen', 'Device ID', 'Location'])

        print("Data preparation complete.")

    def save_data(self):
        """Save the updated dataset to the specified output CSV file."""
        self.dataset.to_csv(self.output_filepath, index=False)
        print(f"Data saved to {self.output_filepath}.")

    def display_data(self, num_rows=5):
        """Display the first few rows of the dataset."""
        print(self.dataset.head(num_rows))
