import numpy as np
from hmmlearn import hmm

class RoomTransitionHMM:
    def __init__(self, n_states, random_state=42, n_iter=100):
        """Initialize the Categorical HMM with the given number of states and other parameters."""
        self.model = hmm.CategoricalHMM(n_components=n_states, n_iter=n_iter, random_state=random_state)

    def train(self, sequences, lengths):
        """
        Train the Categorical HMM model using preprocessed sequences and lengths.
        Assumes sequences contain multiple features, with the first column as the categorical feature (e.g., Location).
        """
        # Extract only the categorical feature for training
        categorical_sequences = sequences[:, 0].astype(int).reshape(-1, 1)
        total_samples = len(categorical_sequences)

        if sum(lengths) != total_samples:
            raise ValueError(
                f"Mismatch: sum(lengths) is {sum(lengths)}, but total samples is {total_samples}. "
                f"Check your binning or input data."
            )

        # Fit the model with categorical features
        self.model.fit(categorical_sequences, lengths)

        print("Training complete. Model Parameters:")
        print("Transition Matrix:", self.model.transmat_)
        print("Emission Probabilities:", self.model.emissionprob_)

    def predict(self, sequence):
        """Predict the hidden states for a given observation sequence."""
        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)

        # Extract only the categorical feature (e.g., location code)
        categorical_sequence = sequence[:, 0].astype(int).reshape(-1, 1)

        # Decode sequence using Viterbi algorithm
        logprob, hidden_states = self.model.decode(categorical_sequence, algorithm="viterbi")
        return hidden_states

