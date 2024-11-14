class ModelConfig:
    def __init__(self, num_states=10, bin_size=15, random_state=42, n_iter=200, test_size=0.2):
        """Store model parameters for easy access and modification."""
        self.num_states = num_states  # Number of states in HMM
        self.bin_size = bin_size      # Time bin size in minutes
        self.random_state = random_state  # Seed for reproducibility
        self.n_iter = n_iter          # Number of training iterations for HMM
        self.test_size = test_size    # Split ratio for training and test sets

    def display_config(self):
        """Display the current configuration settings."""
        print(f"Model Configuration:\n- Num States: {self.num_states}\n- Bin Size: {self.bin_size} minutes\n"
              f"- Random State: {self.random_state}\n- Iterations: {self.n_iter}\n- Test Size: {self.test_size}")
