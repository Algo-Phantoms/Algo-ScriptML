import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz as gv


class HiddenMarkovModel:
    def __init__(
        self,
        observable_states,
        hidden_states,
        transition_matrix,
        emission_matrix,
        title="HMM",
    ):
        """Initialization function for HiddenMarkovModel

        Attributes:
            observable_states (list): A list containing the name of each observable state.
            hidden_states (list): A list containing the name of each hidden state.
            transition_matrix (2-D list): A matrix containing the transition probabilities.
            emission_matrix (2-D list): A matrix containing the emission probabilities.
            title (str): Title for the HMM project. Output files will be named with this attribute.
        """

        self.observable_states = observable_states
        self.hidden_states = hidden_states
        self.transition_matrix = pd.DataFrame(
            data=transition_matrix, columns=hidden_states, index=hidden_states
        )
        self.emission_matrix = pd.DataFrame(
            data=emission_matrix, columns=observable_states, index=hidden_states
        )
        self.pi = self._calculate_stationary_distribution()
        self.title = title

    def print_model_info(self):
        """Prints the model in a readable manner."""

        print("*" * 50)
        print(f"Observable States: {self.observable_states}")
        print(f"Emission Matrix:\n{self.emission_matrix}")
        print(f"Hidden States: {self.hidden_states}")
        print(f"Transition Matrix:\n{self.transition_matrix}")
        print(f"Initial Probabilities: {self.pi}")

    def visualize_model(self, output_dir="outputs", notebook=False):
        """Creates a transition and emission graph of the model.

        Args:
            output_dir (str): A directory will be created with this name. If the directory already exists then an error will be raised.
            notebook (bool): Whether the model should be visualized for a notebook or a script. If False, then a png will be displayed. If True then the output will be displayed in the IPython cell.
        """

        try:
            os.mkdir(output_dir)
        except FileExistsError:
            raise FileExistsError(
                "Directory already exists! Please provide a different output directory!"
            )
        output_loc = output_dir + "/" + self.title

        G = nx.MultiDiGraph()
        G.add_nodes_from(self.hidden_states)

        # Get transition probabilities
        hidden_edges = self._get_markov_edges(self.transition_matrix)
        for (origin, destination), weight in hidden_edges.items():
            G.add_edge(origin, destination, weight=weight, label=weight, color="blue")

        # Get emission probabilities
        emission_edges = self._get_markov_edges(self.emission_matrix)
        for (origin, destination), weight in emission_edges.items():
            G.add_edge(origin, destination, weight=weight, label=weight, color="red")

        # Create graph and draw with edge labels
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        edge_labels = {(n1, n2): d["label"] for n1, n2, d in G.edges(data=True)}
        nx.drawing.nx_pydot.write_dot(G, output_loc + ".dot")

        s = gv.Source.from_file(output_loc + ".dot", format="png")
        if notebook:
            from IPython.display import display

            display(s)
            return
        s.view()

    def forward(self, input_seq):
        """Runs the Forward Algorithm.

        Args:
            input_seq (list): A list of the observed input sequence.

        Returns:
            alpha (np.array): A matrix of the alpha values.
            probs (numpy.float64): The computed probability of the input sequence.
        """

        input_seq = np.array(input_seq)
        n_states = len(self.hidden_states)
        T = len(input_seq)

        # Convert DataFrame to np.array
        emission_matrix = self.emission_matrix.values
        transition_matrix = self.transition_matrix.values

        # Initialize alpha
        alpha = np.zeros((n_states, T))
        alpha[:, 0] = self.pi * emission_matrix[:, input_seq[0]]

        for t in range(1, T):
            for s in range(n_states):
                alpha[s, t] = emission_matrix[s, input_seq[t]] * np.sum(
                    alpha[:, t - 1] * transition_matrix[:, s]
                )
        probs = alpha[:, -1].sum()
        return alpha, probs

    def backward(self, input_seq):
        """Runs the Backward Algorithm.

        Args:
            input_seq (list): A list of the observed input sequence.

        Returns:
            beta (np.array): A matrix of the beta values.
            probs (numpy.float64): The computed probability of the input sequence.
        """

        input_seq = np.array(input_seq)
        n_states = len(self.hidden_states)
        T = len(input_seq)

        # Convert DataFrame to np.array
        emission_matrix = self.emission_matrix.values
        transition_matrix = self.transition_matrix.values

        # Initialize beta starting from last
        beta = np.zeros((n_states, T))
        beta[:, T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            for s in range(n_states):
                beta[s, t] = np.sum(
                    emission_matrix[:, input_seq[t + 1]]
                    * beta[:, t + 1]
                    * transition_matrix[s, :]
                )
        probs = sum(self.pi * emission_matrix[:, input_seq[0]] * beta[:, 0])
        return beta, probs

    def viterbi(self, input_seq):
        """Runs the Viterbi Algorithm.

        Args:
            input_seq (list): A list of the observed input sequence.

        Returns:
            path (np.array): The output path for given input sequence.
            delta (np.array): A matrix of the delta values.
            phi (numpy.array): A matrix of the phi values.
        """

        input_seq = np.array(input_seq)
        n_states = len(self.hidden_states)
        T = len(input_seq)

        # Convert DataFrame to np.array
        emission_matrix = self.emission_matrix.values
        transition_matrix = self.transition_matrix.values

        # Initial blank path
        path = np.zeros(T, dtype=int)
        # Delta = Highest probability of any path that reaches state i
        delta = np.zeros((n_states, T))
        # Phi = Argmax by time step for each state
        phi = np.zeros((n_states, T))

        # Initialize delta
        delta[:, 0] = self.pi * emission_matrix[:, input_seq[0]]

        print("*" * 50)
        print("Starting Forward Walk")

        for t in range(1, T):
            for s in range(n_states):
                delta[s, t] = (
                    np.max(delta[:, t - 1] * transition_matrix[:, s])
                    * emission_matrix[s, input_seq[t]]
                )
                phi[s, t] = np.argmax(delta[:, t - 1] * transition_matrix[:, s])
                print(f"State={s} : Sequence={t} | phi[{s}, {t}]={phi[s, t]}")

        print("*" * 50)
        print("Start Backtrace")
        path[T - 1] = np.argmax(delta[:, T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = phi[path[t + 1], [t + 1]]
            print(f"Path[{t}]={path[t]}")

        return path, delta, phi

    def _calculate_stationary_distribution(self):
        """Calculates the initial stationary distribution for the model.

        Returns:
            stationary (np.array): The stationary distribution.
        """

        eig_vals, eig_vects = np.linalg.eig(self.transition_matrix.T.values)
        _eig_vects = eig_vects[:, np.isclose(eig_vals, 1)]
        _eig_vects = _eig_vects[:, 0]
        stationary = _eig_vects / _eig_vects.sum()
        stationary = stationary.real
        return stationary

    def _get_markov_edges(self, matrix):
        """Returns the edges between two states.

        Args:
            matrix (pd.DataFrame): A matrix attribute of the model.

        Returns:
            edges: A dictionary of the edges between each state.
        """

        edges = {}
        for col in matrix.columns:
            for row in matrix.index:
                edges[(row, col)] = matrix.loc[row, col]
        return edges


def print_forward_result(alpha, a_prob):
    """Prints the result of the Forward Algorithm.

    Args:
        alpha (np.array): A matrix of the alpha values.
        a_prob (numpy.float64): The computed probability from the alpha values.
    """

    print("*" * 50)
    print(f"Alpha:\n{alpha}\nProbability of sequence: {a_prob}")


def print_backward_result(beta, b_prob):
    """Prints the result of the Backward Algorithm.

    Args:
        beta (np.array): A matrix of the beta values.
        b_prob (numpy.float64): The computed probability from the beta values.
    """

    print("*" * 50)
    print(f"Beta:\n{beta}\nProbability of sequence: {b_prob}")


def print_viterbi_result(input_seq, observable_states, hidden_states, path, delta, phi):
    """Prints the result of the Viterbi Algorithm.

    Args:
        input_seq (list): A list of the observed input sequence.
        observable_states (list): A list containing the name of each observable state.
        hidden_states (list): A list containing the name of each hidden state.
        path (np.array): The output path for given input sequence.
        delta (np.array): A matrix of the delta values.
        phi (numpy.array): A matrix of the phi values.
    """

    print("*" * 50)
    print("Viterbi Result")
    print(f"Delta:\n{delta}")
    print(f"Phi:\n{phi}")

    state_path = [hidden_states[p] for p in path]
    inv_input_seq = [observable_states[i] for i in input_seq]

    print(
        f"Result:\n{pd.DataFrame().assign(Observation=inv_input_seq).assign(BestPath=state_path)}"
    )

# OUTPUT

# **************************************************
# Observable States: ['Sad', 'Happy']
# Emission Matrix:
#        Sad  Happy
# Rainy  0.8    0.2
# Sunny  0.4    0.6
# Hidden States: ['Rainy', 'Sunny']
# Transition Matrix:
#        Rainy  Sunny
# Rainy    0.5    0.5
# Sunny    0.3    0.7
# Initial Probabilities: [0.375 0.625]
# **************************************************
# Alpha:
# [[0.3    0.18   0.0258]
#  [0.25   0.13   0.1086]]
# Probability of sequence: 0.13440000000000002
# **************************************************
# Beta:
# [[0.256  0.4    1.    ]
#  [0.2304 0.48   1.    ]]
# Probability of sequence: 0.13440000000000002
# **************************************************
# Starting Forward Walk
# State=0 : Sequence=1 | phi[0, 1]=0.0
# State=1 : Sequence=1 | phi[1, 1]=1.0
# State=0 : Sequence=2 | phi[0, 2]=0.0
# State=1 : Sequence=2 | phi[1, 2]=0.0
# **************************************************
# Start Backtrace
# Path[1]=0
# Path[0]=0
# **************************************************
# Viterbi Result
# Delta:
# [[0.3   0.12  0.012]
#  [0.25  0.07  0.036]]
# Phi:
# [[0. 0. 0.]
#  [0. 1. 0.]]
# Result:
#   Observation BestPath
# 0         Sad    Rainy
# 1         Sad    Rainy
# 2       Happy    Sunny