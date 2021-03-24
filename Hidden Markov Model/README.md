# Hidden Markov Model

## What is a Hidden Markov Model?

A Hidden Markov Model (HMM) is a statistical Markov model in with the system being modeled is assumed to be a Markov process with hidden states.

An HMM allows us to talk about both observed events (like words that we see in the input) and hidden events (like Part-Of-Speech tags).

An HMM is specified by the following components:

![Markov Model Parameters](assets/model.png)

**State Transition Probabilities** are the probabilities of moving from state i to state j.
  
![State Transition Probability](assets/state.png)

**Observation Probability Matrix** also called emission probabilities, express the probability of an observation Ot being generated from a state i.

![Observation Probability Matrix](assets/observation.png)

**Initial State Distribution** $\pi$i is the probability that the Markov chain will start in state i. Some state j with $\pi$j=0 means that they cannot be initial states.

Hence, the entire Hidden Markov Model can be described as,

![Initial State Distribution](assets/initial.png)

# Example

For the example in ```main.py``` the Hidden Markov Model is as follows:

![Output Image](outputs/HMM.dot.png)