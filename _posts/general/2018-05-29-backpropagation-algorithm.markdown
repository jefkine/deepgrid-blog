---
layout: post
title: "Backpropagation Algorithm"
description: "Backpropagation Algorithm."
author: Jefkine
comments: true
date: 2018-05-29 20:36:02 +0500
meta: A closer look at the backward propagation algorithm
cover-image: convolution.png
source: http://www.jefkine.com
category: general
---

### Introduction ###
Backpropagation [1, 2] is a supervised learning algorithm used in artificial neural networks to calculate the rate change in individual weights on every iteration where data samples are passed through the network.

Optimization of a neural network involves minimizing a loss function where we iteratively take steps in the opposite direction of steepest descent i.e in the direction of the negative gradient to be able to reach the local minima. This is known as gradient descent, an algorithm that makes use of backpropagation to adjusts the weights in a neural network by calculating the gradient of the loss function.

The error calculated at the output of the network is distributed back through the network layers hence the name backpropagation.

### Notation ###
1. $$ l $$ is the $$ l^{th} $$ layer where $$ l=1 $$ is the first layer and $$ l=L $$ is the last layer.
2. $$ o_{i,j}^l $$ is the output vector at layer $$ l $$
\$$
\begin{align}
o_{i,j}^{l} &= \sum_{i =1}^{n_l+1} w_{i\rightarrow j}^l a_{p,i}^l
\end{align}
\$$
3. $$ w_{i\rightarrow j}^l $$ is the weight vector connecting neuron $$ i $$ of layer $$ l $$ with neuron $$ j $$ of layer $$ l+1 $$.
4. $$ a^l $$ is the activated output vector for a hidden layer $$ l $$.
\$$
\begin{align}
a_{p,j}^{l} &= f(o_{p,j}^{l-1})
\end{align}
\$$
5. $$a^L$$ is the activated output of the last layer $$L$$, which is also denoted $$\hat{y}$$: the predicted output.
6. $$f(x)$$ is the activation function.
7. $$ P $$ is the total number of patterns for network training.

For a dataset of $$N$$ input-output value pairs $$\left\{ (\vec{x_i}, y_i)\right\}_{i=1}^{N}$$, the $$i^{th}$$ training example is given by $$(\vec{x_i}, y_i)$$ where:

* $$\vec{x_i} \in \mathbb{R}^{n}$$ is the vector of n input units
* $$y_i \in \mathbb{R}$$ is the target output (ground truth)
* $$\hat{y_i} \in \mathbb{R}$$ is the predicted output

Learning will then be achieved by adjusting the weights such that $$\hat{y_i}$$ is as close as possible or equals to $$y$$.

Input output error for an individual pattern $$p$$ is given by $$E_p$$. The overall error measure is taken on the output units over all the $$ P $$ patterns is $$E = \sum_{p} E_p$$

In the classical back-propagation algorithm, the weights are changed according to the gradient descent direction of an error surface $$ E $$ where
$$
\begin{align}
E &= \sum_{p =1}^P \left( \frac{1}{2N} \sum_{i}^{N} \left(y_{p,i} - \hat{y}_{p,i}\right)^2 \right) \tag{1}
\end{align}
$$

For the derivations in this article, we shall use the overall error which is combination of all the individual input output pair patterns given by
$$
\begin{align}
E &= \frac{1}{2N} \sum_{i}^{N} \left(y_{i} - \hat{y}_{i}\right)^2  \tag{2a} \\
&= \frac{1}{2} \left(y - \hat{y}\right)^2  \tag{2b}
\end{align}
$$



### Single Layer Perceptron ###

### References ###
1. Rumelhart, David E.; Hinton, Geoffrey E.; Williams, Ronald J. (8 October 1986). "Learning representations by back-propagating errors".
[Nature 323 (6088): 533–536.doi:10.1038/323533a0.](https://www.nature.com/articles/323533a0){:target="_blank"}
2. Rumelhart, David E., and Geoffrey E. Hintonf. "Learning representations by back-propagating errors." NATURE 323 (1986): 9. [[pdf]](http://www.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf){:target="_blank"}, [[pdf]](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf){:target="_blank"}
