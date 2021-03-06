## Introduction:

Neural Network (Artificial) ANN is a high-performance computing device whose core theme is inspired by biological neural networks. The human brain comprises billions of neurons, each of which is linked to several other neurons to form a network, allowing it to recognize and process images. Each biological neuron can process a variety of inputs and generate output. Neurons in the human brain are capable of making extremely complex decisions, which means they can perform several tasks parallel. All of these concepts led to the development of a computer model of the brain using an artificial neural network.
        The primary goal of an artificial neural network is to create a system that can perform a variety of computational tasks faster than conventional systems. Pattern recognition and classification, approximation, optimization, and data clustering are some of these functions. ANN collects a large number of units that are linked in some way to enable contact between them. These modules, also known as nodes or neurons, are basic processors that work in a parallel fashion.

## Elements of a Neural Network:

Input Layer - Input features are provided to this layer.  It includes information from the outside world to the network; no computation is done at this layer. Nodes here only pass on the data (features) to the hidden layer.

Hidden Layer - This layer's nodes aren't visible to the outside world; they're part of the abstraction that every neural network provides. The hidden layer computes all of the features entered via the input layer and sends the results to the output layer.

Output Layer - This layer communicates the network's acquired knowledge to the outside world.


## Artificial Neuron

![image](https://user-images.githubusercontent.com/67017422/115409740-a6b4ad80-a20f-11eb-9abe-eb7c31af1af9.png)


Artificial neurons are the basic unit of a neural network. The artificial neuron takes one or more inputs and adds them together to create an output. Perceptrons are another name for artificial neurons. An artificial neuron is:

Y= Σ (weights * input) + bias

wights= It controls the signal between two neurons (or the intensity of the connection) To put it another way, a weight determines how much of an impact the input has on the output.

Bias= Constant biases are an extra input into the next layer that often has the value of one. The bias unit ensures that even though all of the inputs are zeros, the neuron will still be activated.

## Activation Function:

The activation function calculates a weighted number and then adds bias to it to determine if a neuron should be activated or not. For non-linear complex functional mappings between the inputs and the required variable, activation functions are used. The activation function's goal is to introduce non-linearity into a neuron's output.

Some commonly used activation functions are:

## Sigmoid Function - 

f(x) = 1 / 1 + exp(-x)

![image](https://user-images.githubusercontent.com/67017422/115409928-d499f200-a20f-11eb-9bd0-481f5decdca8.png)

As per looking at the graph its range can be defined from 0 to 1.

Disadvantages:

Slow convergence
Vanishing gradient problem
The Sigmoid's output is not zero-centered, causing its gradient to shift in different directions.

## tanh Function:

The hyperbolic tangent function is represented as

f(x) = 1 — exp(-2x) / 1 + exp(-2x)

![image](https://user-images.githubusercontent.com/67017422/115410299-23e02280-a210-11eb-82e6-5e47ca388cae.png)


As per looking at the graph its range can be defined from -1 to 1.

Unlike the sigmoid function, the output of tanh function is zero-centered. But the vanishing gradient problem still prevails.

## ReLu Function:

Rectified linear units function is the most commonly used function as it solves the problem that the above two functions could not solve. If the function receives any negative input, it returns 0; however, if the function receives any positive value x, it returns that value. It can be represented as

f(x)= max(0,x)

![image](https://user-images.githubusercontent.com/67017422/115410522-54c05780-a210-11eb-8190-4cd20d03ea97.png)

As per looking at the graph its range can be defined from 0 to infinite.



For more you can also go through this video
https://www.youtube.com/watch?v=aircAruvnKk

