import numpy as np
from hmmlearn import hmm
import pandas as pd

class MarchovModel:
    def __init__(self, n_states, random_state=42):
        """Initialize the HMM with the given number of states."""
        self.model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=random_state)

    def train(self, csv_filepath):
        """Train the HMM model with the provided CSV data."""
        data = pd.read_csv(csv_filepath)
        
        # Combine Location_Code and Time of Day into a structured sequence array
        sequences = data[['Location_Code', 'Time of Day']].values

        # Validate sequences for training
        if np.any(sequences < 0) or np.isnan(sequences).any():
            raise ValueError("Invalid values found in sequences. Ensure nonnegative integers with no NaNs.")

        # Ensure the data type is integer
        sequences = sequences.astype(int)

        # Log data structure for debugging
        print("Sequences before fitting:")
        print(sequences[:10])  # Print the first 10 sequences for verification

        # Fit the model
        self.model.fit(sequences)
        print("Model training completed.")

        # Display the trained parameters
        print("Transition Matrix:")
        print(self.model.transmat_)
        print("Emission Probabilities:")
        print(self.model.emissionprob_)

    def predict(self, sequence):
        """Predict the next state given a sequence of observations."""
        if sequence.ndim != 2 or sequence.shape[1] != 2:
            raise ValueError("Input sequence must be a 2D array with two columns: Location_Code and Time of Day")
        
        # Ensure input is valid
        if np.any(sequence < 0) or np.isnan(sequence).any():
            raise ValueError("Invalid values in input sequence. Ensure nonnegative integers and no NaNs.")
        
        # Ensure data type consistency
        sequence = sequence.astype(int)

        # Log shape for debugging
        print(f"Input sequence shape: {sequence.shape}")

        # Perform prediction
        logprob, hidden_states = self.model.decode(sequence, algorithm="viterbi")

        # Log output shape for verification
        print(f"Predicted hidden states shape: {hidden_states.shape}")
        
        return hidden_states
