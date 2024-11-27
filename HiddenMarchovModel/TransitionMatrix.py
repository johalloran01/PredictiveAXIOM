from collections import defaultdict, Counter
import numpy as np
import scipy.sparse as sp
import pandas as pd

class TransitionMatrix:
    def __init__(self, epsilon=1e-6):
        """
        Initialize the TransitionMatrix with smoothing and data structures.
        :param epsilon: Smoothing constant for unseen transitions.
        """
        self.count_matrix = defaultdict(Counter)  # Counts transitions
        self.prob_matrix = None  # Transition probabilities stored as a sparse matrix
        self.room_index = {}  # Maps room names to matrix indices
        self.index_room = {}  # Reverse mapping for debugging
        self.epsilon = epsilon  # Smoothing constant

    def calculate_transitions(self, data):
        """
        Calculate transition counts and normalize to create a transition probability matrix.
        :param data: Pandas DataFrame containing 'Location' column.
        """
        # Validate input data
        if 'Location' not in data.columns:
            raise ValueError("The input data must contain a 'Location' column.")

        # Step 1: Populate the count matrix by iterating over consecutive rows
        for i in range(len(data) - 1):
            current_room = data.iloc[i]['Location']
            next_room = data.iloc[i + 1]['Location']
            self.count_matrix[current_room][next_room] += 1

        # Step 2: Map unique room names to indices
        all_rooms = set(self.count_matrix.keys())
        for transitions in self.count_matrix.values():
            all_rooms.update(transitions.keys())
        self.room_index = {room: i for i, room in enumerate(sorted(all_rooms))}
        self.index_room = {i: room for room, i in self.room_index.items()}

        # Step 3: Create a sparse transition probability matrix
        row, col, data_values = [], [], []
        for current_room, transitions in self.count_matrix.items():
            current_index = self.room_index[current_room]
            total_transitions = sum(transitions.values()) + self.epsilon * len(self.room_index)
            for next_room in self.room_index.keys():  # Include all possible next rooms
                next_index = self.room_index[next_room]
                count = transitions.get(next_room, 0) + self.epsilon  # Apply smoothing
                row.append(current_index)
                col.append(next_index)
                data_values.append(count / total_transitions)  # Normalize probabilities

        # Store the sparse matrix
        self.prob_matrix = sp.csr_matrix((data_values, (row, col)), shape=(len(self.room_index), len(self.room_index)))

        # Debugging output
        print(f"Transition matrix calculated with {len(self.room_index)} rooms.")
        print(f"Non-zero transitions: {self.prob_matrix.nnz}")
        print(f"Matrix shape: {self.prob_matrix.shape}")

    def get_transition_probability(self, current_room, next_room):
        """
        Get the transition probability from current_room to next_room.
        :param current_room: Name of the current room.
        :param next_room: Name of the next room.
        :return: Transition probability.
        """
        if current_room not in self.room_index or next_room not in self.room_index:
            return 0.0
        current_index = self.room_index[current_room]
        next_index = self.room_index[next_room]
        return self.prob_matrix[current_index, next_index]

    def display_matrix(self, max_rows=10):
        """
        Display the transition probability matrix as a dense array for debugging.
        :param max_rows: Maximum number of rows to display.
        """
        if self.prob_matrix is not None:
            dense_matrix = self.prob_matrix.toarray()
            print("Transition Probability Matrix (truncated view):")
            for i, row in enumerate(dense_matrix[:max_rows]):
                room_name = self.index_room[i]
                print(f"{room_name}: {np.round(row, 4)}")
        else:
            print("Transition matrix has not been calculated yet.")
