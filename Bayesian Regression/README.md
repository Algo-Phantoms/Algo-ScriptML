# Bayesian Linear Regression

# Introduction
  Bayesian linear regression is an approach to linear regression in which the statistical analysis is undertaken within the context of Bayesian inference. When the regression model has errors that have a normalIn baye distribution, and if a particular form of prior distribution is assumed, explicit results are available for the posterior probability distributions of the model's parameters.Bayesian linear regression pushes the idea of the parameter prior a step further and does nt even attempt to compute a point estimate of the parameters,but instead the full posterior distribution over the parameters is taken into account when making predicitions.This means we do not fit any parameters,but we compute a mean over all the plausible parameters settings(ccording to the posterior).
 P( &#x3B8;/y,x)=(P(y/ &#x3B8;,x)*P( &#x3B8;/x))/p(y/x)
 here P( &#x3B8;/y,x) is the posterior probability distribution  of the model parameters given the input and the output
 Posterior =(Likelihood*Prior)/Normalization where:
 Priors
 <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>P</mi>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mfrac>
      <mi>y</mi>
      <mi>/</mi>
      <mi>x</mi>
    </mfrac>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mo>=</mo>
  <msubsup>
    <mo data-mjx-texclass="OP">&#x222B;</mo>
    <mrow></mrow>
    <mrow></mrow>
  </msubsup>
  <mi>P</mi>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mfrac>
      <mi>y</mi>
      <mi>/</mi>
      <mi>x</mi>
    </mfrac>
    <mo>,</mo>
    <mi>&#x3B8;</mi>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mo>.</mo>
  <mi>p</mi>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mi>&#x3B8;</mi>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mtext>&#xA0;</mtext>
  <mi>d</mi>
  <mi>&#x3B8;</mi>
</math>
 
 this is = E[ p(y/x,&#x3B8;)] where E stands for the expectation of  the distribution p wrt &#x3B8;(in lyman it's the average over the enitre distribution)
 for all plausible parameters &#x3B8; according to the prior distribution only require us to specify the input x,but not training data.

Posterior:
P(&#x3B8;/x,y)=(P(y/x,&#x3B8;)P(&#x3B8;))/P(y/x)
# Implementing the Bayesian Linear regression 
The basic procedure for implementing Bayesian Linear Modellling includes:
1.Specifiying priors for the model parameters( normal distributions preferable <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>N</mi>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mi>&#x3BC;</mi>
    <mo>,</mo>
    <msup>
      <mi>&#x3C3;</mi>
      <mn>2</mn>
    </msup>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
</math>) 

2.Create a model mapping the training inputs to the training outputs,and 
3. have a Markov Chain maonte carlo (MCMC) algorithm to draw samples from the posterior distributions for the model parameters.
The end result will be posterior distribution for the parameters.
# Application
When we want show the linear fit from a Bayesian model, instead of showing only estimate, we can draw a range of lines, with each one representing a different estimate of the model parameters. As the number of datapoints increases, the lines begin to overlap because there is less uncertainty in the model parameters.
![1_8bA09THSC_Cy5LeijM8oEA](https://user-images.githubusercontent.com/70088281/111058347-02bf4000-84b4-11eb-94d4-1f008040f470.png) 

![1_C8eR-V648On7Nb11eMqWlg](https://user-images.githubusercontent.com/70088281/111058375-369a6580-84b4-11eb-8913-202339fe03d8.png)

When using less data points, the fits have a lot of variance, which means that the model is more unpredictable. Since the priors are washed out by the likelihoods from the data, the OLS and Bayesian Fits are virtually similar with all of the data points.<br />

When predicting the output for a single datapoint using our Bayesian Linear Model, we also do not get a single value but a distribution.

![1_vKbWqDqfz_crZ1C2Ew9dxA](https://user-images.githubusercontent.com/70088281/111058448-b9bbbb80-84b4-11eb-9d9c-c39495874b6d.png)

 probability density plot for the number of calories burned exercising for 15.5 minutes. The red vertical line indicates the point estimate from OLS.
 The chance of burning a certain amount of calories ranges at about 89.3, although the full approximation is a variety of potential values.
 # Applications
 The Bayesian Linear Regression framework will integrate prior data while still showing our uncertainty. The Bayesian method is reflected in Bayesian Linear Regression: we construct an initial approximation and refine it as more evidence is gathered. The Bayesian perspective is a natural way of seeing the universe.The inference(bayesian) is a much better alternative to its frequentist counterpart.
 # Advantages
 The bayesian regression algorithm is much better alternative than regular(frequentist) method since MLE can lead to severe overfitting,in particular in small data regime.Maximum apriori approximation does not give a good representation of our uncertainities hence Bayesian regression is considered as a good choice .it does not even attempt to compute a point estimate of the parameters,but instead the full posterior distribution over the parameters is taken into account when making predictions.
 # Disadvantages
 It does not tell you how to select a prior. There is no correct way to choose a prior. Bayesian inferences require skills to translate subjective prior beliefs into a mathematically formulated prior.
 # References
 https://statswithr.github.io/book/introduction-to-bayesian-regression.html
 https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7#:~:text=The%20aim%20of%20Bayesian%20Linear,from%20a%20distribution%20as%20well.
 https://www.youtube.com/watch?v=0F0QoMCSKJ4
 https://www.youtube.com/watch?v=LzZ5b3wdZQk&t=112s
 books:mathematics for machine learning:Marc Peter Deisenroth,A.Aldo Faisal,Cheng Soon Ong
 
# Thanks For Reading
