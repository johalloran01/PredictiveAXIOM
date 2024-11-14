from datetime import datetime, timedelta
import numpy as np

class TimeBinner:
    def __init__(self, bin_size=15):
        """Initialize with a configurable bin size in minutes."""
        self.bin_size = bin_size

    def bin_time(self, timestamp):
        """Convert a timestamp to a time bin based on bin size."""
        bin_minutes = (timestamp.minute // self.bin_size) * self.bin_size
        return timestamp.replace(minute=bin_minutes, second=0, microsecond=0)

    def apply_binning(self, data):
        """Apply time binning to the entire dataset."""
        binned_sequences = []
        sequence_lengths = []
        total_appended = 0  # Track cumulative samples added

        for index, row in data.iterrows():
            start_time = datetime.strptime(row['First Seen'], '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(row['Last Seen'], '%Y-%m-%d %H:%M:%S')

            # Bin start and end times to nearest bin
            binned_start = self.bin_time(start_time)
            binned_end = self.bin_time(end_time)

            # Calculate duration in bins; ensure at least one bin even if binned_start == binned_end
            duration_bins = max(1, int((binned_end - binned_start).total_seconds() // (self.bin_size * 60)))

            # Append the location for each bin
            binned_sequences.extend([[row['Location']]] * duration_bins)
            sequence_lengths.append(duration_bins)
            total_appended += duration_bins

            # Debugging output to check each entry
            actual_added = len(binned_sequences) - total_appended + duration_bins
            if duration_bins != actual_added:
                print(f"Index {index} mismatch: Expected {duration_bins} bins, but added {actual_added} bins.")
                print(f"Row: Start - {start_time}, End - {end_time}")
                print(f"Binned start: {binned_start}, Binned end: {binned_end}")

        # Final validation to check if total length matches
        if sum(sequence_lengths) != len(binned_sequences):
            print(f"Total Mismatch: sum(sequence_lengths) = {sum(sequence_lengths)}, "
                  f"total samples in binned_sequences = {len(binned_sequences)}")
            print(f"Sample sequence lengths: {sequence_lengths[:10]}")
            raise ValueError("Mismatch detected in final alignment of lengths and sequences.")

        return np.array(binned_sequences), sequence_lengths
