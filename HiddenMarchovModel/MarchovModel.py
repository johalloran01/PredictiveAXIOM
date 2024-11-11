import numpy as np
from hmmlearn import hmm
import pandas as pd

class MarchovModel:
    def __init__(self, n_states, random_state=42):
        """Initialize the HMM with the given number of states."""
        self.model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=random_state)

    def train(self, csv_filepath):
        """Train the HMM model with the provided CSV data."""
        # Load the formatted data
        data = pd.read_csv(csv_filepath)
        
        # Combine Location_Code and Time of Day into a structured sequence array
        sequences = data[['Location_Code', 'Time of Day']].values

        # Fit the model to the training sequences
        self.model.fit(sequences)
        print("Model training completed.")

        # Display the trained parameters
        print("Transition Matrix:")
        print(self.model.transmat_)
        print("Emission Probabilities:")
        print(self.model.emissionprob_)

    def predict(self, sequence):
        """Predict the next state given a sequence of observations."""
        # Ensure the input sequence includes both Location_Code and Time of Day
        logprob, hidden_states = self.model.decode(sequence, algorithm="viterbi")
        return hidden_states
