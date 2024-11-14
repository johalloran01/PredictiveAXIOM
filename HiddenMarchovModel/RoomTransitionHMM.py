import numpy as np
from hmmlearn import hmm

class RoomTransitionHMM:
    def __init__(self, n_states, random_state=42, n_iter=100):
        """Initialize the Categorical HMM with the given number of states and other parameters."""
        self.model = hmm.CategoricalHMM(n_components=n_states, n_iter=n_iter, random_state=random_state)

    def train(self, sequences, lengths):
        """Train the Categorical HMM model using preprocessed sequences and lengths."""
        sequences = np.vstack(sequences).astype(int)
        total_samples = len(sequences)
        if sum(lengths) != total_samples:
            raise ValueError(f"Mismatch: sum(lengths) is {sum(lengths)}, but total samples is {total_samples}. Check your binning or input data.")

        
        # Fit the model with categorical features (Location_Code and Time of Day)
        self.model.fit(sequences, lengths)
        
        print("Training complete. Model Parameters:")
        print("Transition Matrix:", self.model.transmat_)
        print("Emission Probabilities:", self.model.emissionprob_)

    def predict(self, sequence):
        """Predict the hidden states for a given observation sequence."""
        logprob, hidden_states = self.model.decode(sequence, algorithm="viterbi")
        #print(f"Predicted hidden states (sample): {hidden_states[:10]}")
        return hidden_states
