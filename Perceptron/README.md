# Perceptron

<p> This script is based on the deep understanding of Neural Networks and Perceptron. </p>
<p> Neurons in Neural Network are inspired from biological neurons. This Neural Network would able to do various tasks like classifying images, prediction, and so on. Alexa and Siri use neural network. 
<br>
A Perceptron is an algorithm used for supervised learning of binary classifiers. Binary classifiers decide whether an input, usually represented by a series of vectors, belongs to a specific class. In short, a perceptron is a single-layer neural network.
</p>

- <p> <h3> <u>Working of Perceptron</u> </h3>

Perceptron looks like the following structure:

<img src = "https://www.allaboutcircuits.com/uploads/thumbnails/how-to-train-a-basic-perceptron-neural-network_rk_aac_image1.jpg" width="700" height="300"> 

We have 3 inputs X,Y and Z. When this inputs get into the net input fucntion it firstly get weighted with some weigh value that is w1,w2 and w3 respectively. The net input fucntion(Z) sums up all the value i.e. Z = (Xi * Wi). This Z will determine if a neuron fires or not. Firing of the neuron depends on a function which is called Activation Function. If the sum of the input signals exceeds a certain threshold, it either outputs a signal or does not return an output. </p>

- <p> <h3> <u>Perceptron Function</u></h3> 

Perceptron is a function that maps its input “x,” which is multiplied with the learned weight coefficient; an output value ”f(x)”is generated.

``` If w.x + b > 0, then f(x) = 1 ; else f(x) = 0 ```

In the equation given above:

“w” = vector of real-valued weights

“b” = bias (an element that adjusts the boundary away from origin without any dependence on the input value)

“x” = vector of input x values

``` Σ (Wi * Xi) ```

“m” = number of inputs to the Perceptron

The output can be represented as “1” or “0.”  It can also be represented as “1” or “-1” depending on which activation function is used.

- <p> <h3> Perceptron Training <h3> 
Refer:
[Perceptron Training](Perceptron_Training.ipynb)