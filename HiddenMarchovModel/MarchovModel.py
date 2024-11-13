import numpy as np
from hmmlearn import hmm

class MarchovModel:
    def __init__(self, n_states, random_state=42):
        """Initialize the Categorical HMM with the given number of states."""
        self.model = hmm.CategoricalHMM(n_components=n_states, n_iter=100, random_state=random_state)

    def train(self, sequences, lengths):
        """Train the Categorical HMM model using preprocessed sequences and lengths."""
        # Ensure integer type consistency
        sequences = np.vstack(sequences).astype(int)

        # Log sample data for debugging
        print("Sequences before fitting (first 10 entries):")
        print(sequences[:10])

        # Fit the model with categorical features (Location_Code and Time of Day)
        # Directly fit the model with sequences and lengths
        self.model.fit(sequences, lengths)
        print("Model training completed.")
        
        # Display trained model parameters
        print("Transition Matrix:")
        print(self.model.transmat_)
        print("Emission Probabilities:")
        print(self.model.emissionprob_)

    def predict(self, sequence):
        """Predict the hidden states for a given observation sequence."""
        # Validation: Check sequence shape and type
        if sequence.ndim != 2 or sequence.shape[1] != 1:
            raise ValueError("Input sequence must be a 2D array with one column for the combined feature.")
        if np.any(sequence < 0) or np.isnan(sequence).any():
            raise ValueError("Invalid values in input sequence. Ensure nonnegative integers and no NaNs.")

        # Decode the sequence to find the most likely hidden states
        logprob, hidden_states = self.model.decode(sequence, algorithm="viterbi")
        
        # Log output shape for verification
        print(f"Predicted hidden states shape: {hidden_states.shape}")
        
        return hidden_states
