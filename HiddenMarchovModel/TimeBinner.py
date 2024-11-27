from datetime import datetime, timedelta
import numpy as np

class TimeBinner:
    def __init__(self, bin_size=15, location_col='Location', next_location_col='NextLocation'):
        self.bin_size = bin_size
        self.location_col = location_col
        self.next_location_col = next_location_col

    def bin_time(self, timestamp):
        """Convert a timestamp to a time bin based on bin size."""
        bin_minutes = (timestamp.minute // self.bin_size) * self.bin_size
        return timestamp.replace(minute=bin_minutes, second=0, microsecond=0)
    
    def get_time_of_day_vector(self, timestamp):
        """
        Convert a timestamp into a one-hot vector for time of day.
        Morning: [1,0,0,0]
        Afternoon: [0,1,0,0]
        Evening: [0,0,1,0]
        Night: [0,0,0,1]
        """
        hour = timestamp.hour
        if 6 <= hour < 12:
            return [1, 0, 0, 0]  # Morning
        elif 12 <= hour < 18:
            return [0, 1, 0, 0]  # Afternoon
        elif 18 <= hour < 24:
            return [0, 0, 1, 0]  # Evening
        else:
            return [0, 0, 0, 1]  # Night

    def get_day_of_week_vector(self, timestamp):
        """
        Convert a timestamp into a one-hot vector for day of the week.
        Monday: [1,0,0,0,0,0,0]
        ...
        Sunday: [0,0,0,0,0,0,1]
        """
        day = timestamp.weekday()  # Monday=0, Sunday=6
        return [1 if i == day else 0 for i in range(7)]

    def apply_binning(self, data, transition_matrix):
        """
        Apply time binning with additional features and debugging enhancements.
        """
        binned_sequences = []
        sequence_lengths = []

        for index, row in data.iterrows():
            start_time = datetime.strptime(row['First Seen'], '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(row['Last Seen'], '%Y-%m-%d %H:%M:%S')

            binned_start = self.bin_time(start_time)
            binned_end = self.bin_time(end_time)

            duration_bins = max(1, int((binned_end - binned_start).total_seconds() // (self.bin_size * 60)))
            current_location = row['Location']
            next_location = row.get('NextLocation', None)

            time_of_day_vector = self.get_time_of_day_vector(binned_start)
            day_of_week_vector = self.get_day_of_week_vector(binned_start)

            transition_prob = (
                transition_matrix.get_transition_probability(current_location, next_location)
                if next_location else 0.0
            )

            # Debugging Logs
            if transition_prob == 0.0 and next_location:
                print(f"Warning: No transition probability for {current_location} -> {next_location}")

            if duration_bins <= 0:
                print(f"Warning: Invalid duration_bins for index {index}, Start: {binned_start}, End: {binned_end}")

            # Append feature vector for each bin
            for _ in range(duration_bins):
                binned_sequences.append([
                    current_location,  # Room code
                    *time_of_day_vector,  # Time of day one-hot
                    *day_of_week_vector,  # Day of week one-hot
                    transition_prob      # Transition probability
                ])
            sequence_lengths.append(duration_bins)

        # Final Validation
        if sum(sequence_lengths) != len(binned_sequences):
            print(f"Total Mismatch: sum(sequence_lengths) = {sum(sequence_lengths)}, "
                    f"total samples in binned_sequences = {len(binned_sequences)}")
            raise ValueError("Mismatch detected in final alignment of lengths and sequences.")

        return np.array(binned_sequences), sequence_lengths
    
    def get_time_of_day(self, timestamp):
        """
        Classify a timestamp into one of four time-of-day bins.
        """
        hour = timestamp.hour
        if 6 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 18:
            return 1  # Afternoon
        elif 18 <= hour < 24:
            return 2  # Evening
        else:
            return 3  # Night
