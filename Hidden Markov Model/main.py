import hmm

# Hidden
hidden_states = ["Rainy", "Sunny"]
transition_matrix = [[0.5, 0.5], [0.3, 0.7]]

# Observable
observable_states = ["Sad", "Happy"]
emission_matrix = [[0.8, 0.2], [0.4, 0.6]]

# Inputs
input_seq = [0, 0, 1]

model = hmm.HiddenMarkovModel(
    observable_states, hidden_states, transition_matrix, emission_matrix
)

model.print_model_info()
model.visualize_model()

alpha, a_probs = model.forward(input_seq)
hmm.print_forward_result(alpha, a_probs)

beta, b_probs = model.backward(input_seq)
hmm.print_backward_result(beta, b_probs)

path, delta, phi = model.viterbi(input_seq)
hmm.print_viterbi_result(input_seq, observable_states, hidden_states, path, delta, phi)