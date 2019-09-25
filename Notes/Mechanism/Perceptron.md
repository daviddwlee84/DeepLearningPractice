# Perceptron

Linear Threshold Unit (LTU) or Linear Threshold Gate (LTG)

$$
a = \sum_{i=1}^n \mathbb{w_i} \mathbb{x_i}
$$

* Net input to unit: defined as linear combination
* Output of unit: *threshold (activation function)* $\theta$ on net input

## [Activation Function](../Element/Activation_Function.md)

* threshold
* linear
* piece-wise linear
* sigmoid
* tanh

## Geometric Interpretation

Decision hyperplane (line): $\mathbb{w} \cdot \mathbb{x} = \theta$

Bias: let $\theta$ be one of the *weights*

$\mathbb{w} \cdot \mathbb{x} = \theta$

$x_0 = 1$ and $w_0 = -\theta$ => $\mathbb{w} \cdot \mathbb{x} = 0$

## Representability

> that's the main reason why the neural network was been attacked in the early age

* Can only represent some function
  * Linear seprable
  * logic gates
    * AND
    * OR
    * NOT
* Not representable
  * Non-linearly seprable

> Solution: use networks of perceptrons (LTUs)

## Training Perceptron

* Use the *training data* to learn a perceptron
* Hypotheses Space: $H = \{w|w\in R^{n+1}\}$
* Two search method
  * Perceptron Training Rule
  * Gradient Descent

> goal: find "good" weights (minimize the error)

### Perceptron Learning Rule

TBD

> It's proved will be converge in finite steps, if the data are *linear seprable*

Pseudocode

```pseudocode
Repeat
    for each training vector pair (x, t)
        evaluate the output o when x is the input
        if o != t then
            form a new weight vector w' according to w' = w + η(t-o)x
        else
            do nothing
        end if
    end for
Until o=t for all training vector pairs
```

### Gradient Descent

If the data are not separable lineary, Perceptron Learning Rule may oscillate (no convergence)

Thus we can only find a *approximately separable line* by using Grdient Descent

* Define an *error function*
* Search for weights that *minimize the error*, i.e. find weights that zero the error gradient

#### Squared Error

Consider linear unit *without threshold* and *continuous output* o (not just -1, 1)

$$
o = w_0 + w_1x_1+\dots+w_nx_n
$$

Squared Error

$$
E(\mathbb{w}) = \frac{1}{2}\sum_{d\in D}(t_d-o_d)^2
$$

Minimize Suared Error

$$
\frac{\partial E}{\partial w_i} = \frac{\partial}{\partial w_i}\frac{1}{2}\sum_{d\in D}(t_d-o_d)^2 = \cdots = \sum_{d\in D}(t_d-o_d)(-x_{id})
$$

#### Online vs. Batch

TBD

> In real world, we use the combination of these two => **minibatch**

### Conclusion - Perceptron Rule vs. Gradient Descent Rule

Perceptron Rule

* Derived from manipulation of decision surface (*threshold output*). Applied in TLU (linear Separable)

Gradient Descent Rule

* Derived from minimization of error function E by means of gradient descent (*unthreshold for output*)

## Links

* [Wiki - Perceptron](https://en.wikipedia.org/wiki/Perceptron)
* [Perceptron Learning Rule](http://hagan.okstate.edu/4_Perceptron.pdf)
  * [slide](http://ecee.colorado.edu/~ecen4831/Demuth/Ch4_pres.pdf)
* [Perceptron Learning Algorithm: A Graphical Explanation Of Why It Works](https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975)
