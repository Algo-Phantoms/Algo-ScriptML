# Bayesian Linear Regression
## In the normal(frequentist) Linear regression we generally estimate model parameters ( &#x3B8;) by means of maximum likelihood or MAP estimation.Since MLE can lead to severe overfitting,in particular,to small data regime.MAP does not give a good representation of our uncertainitiee.
## Bayesian Linear regression pushes the idea of the parameter prior a step further and does not even attempt to compute a point estimte of the parameters but instead the full posterior distirbution is taken into account when making predictions.
 P( &#x3B8;/y,x)=(P(y/ &#x3B8;,x)*P( &#x3B8;/x))/p(y/x)
## here P( &#x3B8;/y,x) is the posterior probability distribution  of the model parameters given the input and the output
# Posterior =(Likelihood*Prior)/Normaliztion
# Priors
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
 
 # this is = E[ p(y/x,&#x3B8;)] where E stands for the expectation of  the distribution p wrt &#x3B8;(in lman it's the average over the enitre distribution)
## for all plausible parameters &#x3B8; according to the prior distribution only require us to specify the input x,but not training data.
# Posterior
## The result of performing Bayesian Linear regression is a distribution of possible model parameters basedon the data and the prior.This allows us to quantify our uncertainity about the model:IF we have fewer data points,the posterior distribution will be more spread out.
P(&#x3B8;/x,y)=(P(y/x,&#x3B8;)P(&#x3B8;))/P(y/x)
# Implementing the Bayesian Linear regression 
## in practice , evaluating the posterior distribution for model parameters i intractable for continiuous variables,so we use sampling methods to draw samples from posterior in order to approximate the posterior is one application of monte carlo method.There are no of algos for monte carlo sampling,most common one being variants of markov chain monte carlo
# Monte carlo sampling
## Monte carlo is a technique for randomly sampling a probability distribution and approximating a desired quantity.

## Monte carlo methods typically assume that we can efficiently draw samples from the target distribution.From the samples that are drawn,we can then estimate the sum or integral quantity as the mean or variance of the drawn sample.

## Markov Chain is a systematic method for generating a sequence of random variables where the current value is probabilistically dependent on the value of the prior variable.Specifically,selecting the next variable is only dependent upon the last variable in the chain.
# Markov chain Monte carlo
## Combining these two methods,Markov chain and  monte carlo,above random sampling of high-dimensionality probability distributions that honors the probability dependence between samples by contributing a Markov Chain that comprise the Monte carlo sample.Specifically,MCMC is for performing inference for probability distributions where independent samples from the distributions cannot be drawn or not drawn easily.The idea is that the chain will settle on(find equilibrium) to the desired quantity we are infering.
## Yet,we are still sampling from the target probability distributions with the goal of approximating a desired quantity, so it ia appropriate to refer to the resulting collection of samples as a monte carlo sample,e.g.  extend of samples drawn often from one long Markov chain
## There are many markov chain monte carlo algorithms that mostly define different ways of constructing the markov chain when performing each Monte carlo sample.Some of them include Gibbs sampling Algorithms,Metropolis-Hashing-algorithm and so on and o forth.
# Bayesian Linear Modeling Application
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
## When we want show the linear fit from a Bayesian model, instead of showing only estimate, we can draw a range of lines, with each one representing a different estimate of the model parameters. As the number of datapoints increases, the lines begin to overlap because there is less uncertainty in the model parameters.
![1_8bA09THSC_Cy5LeijM8oEA](https://user-images.githubusercontent.com/70088281/111058347-02bf4000-84b4-11eb-94d4-1f008040f470.png) 

![1_C8eR-V648On7Nb11eMqWlg](https://user-images.githubusercontent.com/70088281/111058375-369a6580-84b4-11eb-8913-202339fe03d8.png)

When using less data points, the fits have a lot of variance, which means that the model is more unpredictable. Since the priors are washed out by the likelihoods from the data, the OLS and Bayesian Fits are virtually similar with all of the data points.<br />

When predicting the output for a single datapoint using our Bayesian Linear Model, we also do not get a single value but a distribution.

![1_vKbWqDqfz_crZ1C2Ew9dxA](https://user-images.githubusercontent.com/70088281/111058448-b9bbbb80-84b4-11eb-9d9c-c39495874b6d.png)

## probability density plot for the number of calories burned exercising for 15.5 minutes. The red vertical line indicates the point estimate from OLS.
## The chance of burning a certain amount of calories ranges at about 89.3, although the full approximation is a variety of potential values.
## The Bayesian Linear Regression framework will integrate prior data while still showing our uncertainty. The Bayesian method is reflected in Bayesian Linear Regression: we construct an initial approximation and refine it as more evidence is gathered. The Bayesian perspective is a natural way of seeing the universe.The inference(bayesian) is a much better alternative to its frequentist counterpart.
# Thanks For Reading
