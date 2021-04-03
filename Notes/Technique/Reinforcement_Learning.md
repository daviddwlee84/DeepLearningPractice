# Reinforcement Learning

## Overview

* Reinforcement learning optimizs an agent for sparse, time delayed labels called *rewards* in an environment.

![the basic idea and elements involved in a reinforcement learning model](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

### Terms of Reinforcement Learning

* Agent
* Action (A)
* Discount factor (gamma γ)
* Environment: Physical world in which the agent operates
* State (S): Current situation of the agent
* Reward (R): Feedback from the environment
  * $Q_n \doteq \frac{\sum_{i=1}^{n-1} R_i}{n-1}$
  * $Q_{n+1} = Q_n + \frac{1}{n} [R_n - Q_n]$
  * $Q_{n+1} = Q_n + \alpha [R_n - Q_n] = (1 - \alpha)^n Q_1 + \sum_{i=1}^n \alpha (1 - \alpha)^{n-i} R_i$
    * $\alpha$ is a constant, *step-size parameter*, $0 < \alpha \leq 1$
* Policy (π): Method to map agent’s state to actions
* Value (V): Future reward that an agent would receive by taking an action in a particular state
* Q-value or action-value (Q)
  * Projection of Action-Reward
    * $q^*(a) \doteq \mathbb{E}\{R|A_i=a\}$ The ideal (expected) reward of action $a$
    * $Q_t(a)$ The expectation reward of action $a$ at time $t$
* Trajectory

Strategy

* Exploration 探索法: Keep searching for new strategies
  * Probability $\epsilon$ to explore
* Exploitation 利用法: Exploiting the best strategies found thus far
  * Probability $1 - \epsilon$ to exploit
  * $\displaystyle A^* \doteq \arg\max_a Q_t(a)$

### Other Terms in RL

(Some terms using in Monte Carlo Tree Search (MCTS))

* State Value: How good a given state is for an agent to be in. A measurement of the expected oucome (or reward) if we're in this state with respect to our final goal.
* Playout: A simulation of the game that mimics the cause and effect of random action until reach a 'terminal' point. (reliant on a *forward model* that can tell us outcomes of an action of any state)

### Action Selection Strategy

* greedy
* $\epsilon$-greedy
* UCB (Upper Confidential Bounder)
* Gradient

### Problem Space of Reinforcement Learning

| -                        | Single State                    | Associative                             | Multiple State          |
| ------------------------ | ------------------------------- | --------------------------------------- | ----------------------- |
| **Instructive feedback** | Averaging                       | Supervised Learning                     | -                       |
| **Evaluative feedback**  | Bandits (Function optimization) | Associative Search (Contextual bandits) | Markov Decision Process |

> * Instructive feedback: human labeled output (which is good or bad action)
> * Evaluative feedback: the procedure don't need human involve. evaluate environment

* MDP vs. Bandit
  * Bandit: the environment is consistent
  * MDP: your actions will change the environment(state)

## Background

> * Markov Chain: The future is only related to "current state", and it has nothing to do with the past.
> * MDP Framework
>   * Abstract of physical world
>   * Learning by interaction of "action"

### Markov Decision Processes (MDPs)

Markov Decision Process (MDP) - a mathematical framework for modeling decisions using *states*, *actions*, and *rewards*.

* A Mathematical formulation of the RL problem
* **Markov property**: Current state completely characterises the state of the word
* It is a random process without memory
* The reinforcement learning "with model"
  * (Q-learning is RL "without model")

Definitions: $(S, A, R, \mathbb{P}, \gamma)$

* $S$: set of possible states
* $A$: set of possible acitons
* $R$: distribution of reward given (state, action) pair
* $\mathbb{P}$: transition probability (i.e. distribution over next state given (state, aciton) pair)
* $\gamma$: discount factor

Pseudo-process of MDP

1. At time t = 0, environment samples initial states s
2. Do until done:
   1. Agent selects action $a_t$
   2. Environment samples reward $r_t \sim R(.|s_t, a_t)$
   3. Environment samples next state $s_{t+1} \sim P(.|s_t, a_t)$
   4. Agent receives reward $r_t$ and next state $s_{t+1}$

> * Policy
>   * For every *state* project to an *action*
>   * Calculate a Policy such that maximize the cumulated reward

* A **policy** $\pi$ is a function from $S$ to $A$ that specifies what aciton to take in each state.
* Objective: find policy $\pi^*$ that maximizes cumulative discounted reward: $\displaystyle\sum_{t\geq 0} \gamma^t r_t$

**The optimal policy $\pi^*$**: Maximize the **expected sum of rewards**

Formally:

$$
\pi^* = \arg \max_\pi \mathbb{E} \begin{bmatrix}\displaystyle \sum_{t\geq 0} \gamma^t r_t | \pi \end{bmatrix}
$$

**Value function** at state $s$ under a policy $pi$: the expected cumulative reward from following the policy $\pi$ from state $s$ *(Represent how good is a state)*

> * value function: functions of states (or of state-action pairs) that estimate *how good* it is for the aget to be in a give state (or how good it is to perform a given action in a given state)
> * how good: is defined in terms of future rewards that can be expected, or, to be precise, in terms of **expected return**

$$
V_\pi(s) = \mathbb{E} \begin{bmatrix}\displaystyle \sum_{t\geq 0} \gamma^t r_t | s_0 = s, \pi \end{bmatrix}
$$

**Action-value Function** (**Q-value function**) taking action $a$ at state $s$ under policy $\pi$: the expected cumulative reward from taking aciton a in state s and then following the policy *(Represent how good is a state-action pair)*

> a function that calculates the quality of a state-action combination

$$
Q_\pi(s, a) = \mathbb{E} \begin{bmatrix}\displaystyle \sum_{t\geq 0} \gamma^t r_t | s_0 = s, a_0 = a, \pi \end{bmatrix}
$$

optimal Q-value function Q* is the maximum expected cumulative reward achievable from a given (state, aciton) paird

$$
Q^*(s, a) = \max_\pi \mathbb{E} \begin{bmatrix}\displaystyle \sum_{t\geq 0} \gamma^t r_t | s_0 = s, a_0 = a, \pi \end{bmatrix}
$$

> Backup diagrams for (a) $V^*$ and (b) $Q^*$
>
> ![backup diagram](http://www.incompleteideas.net/book/ebook/figtmp13.png)

**Bellman equation**: expresses a relationshp between the vlaue of a state and the vlues of its sucessor states

$$
V_\pi(s) = \underbrace{\sum_a \pi(a|s)}_\text{prob. of taking action $a$} \underbrace{\sum_{s', r} p(s', r| s, a)}_\text{prob. from $(s, a)$ to $s'$} [r + \gamma v_\pi(s')] \quad \forall s \in S
$$

**Bellman optimality equation**: expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state (that $Q^*$ satisfied)

$$
Q^*(s, a) = \mathbb{E}_{s' \sim \varepsilon} \begin{bmatrix}\displaystyle r + \gamma \max_{a'} Q^*(s', a') | s, a \end{bmatrix}
$$

Intuiiton: if the optimal state-aciton values for the next time-stemp $Q^*(s', a')$ are known, then the optimal strategy is to take the action that maximizes the expected value of $r + \gamma Q^*(s', a')$

So the optimal policy $\pi^*$ corresponds to taking the best action in any state as specified by $Q^*$

**Value Iteration Algorithm**: Use Bellman equation as an iterative update

$$
Q_{i+1}(s, a) = \mathbb{E}_{s' \sim \varepsilon} \begin{bmatrix}\displaystyle r + \gamma \max_{a'} Q_{i}(s', a') | s, a \end{bmatrix}
$$

**Q-learning**: Use a function approximator to estimate the action-value funciton ($\theta$ is funciton parameters i.e. weights)

$$
Q(s, a; \theta) \approx Q^*(s, a)
$$

If the **funciton approximator** is a deep neural network, then it's called [**Deep Q-Learning**](#Deep-Q-network-(DQN))

## Q-Learning

Q Learning is a strategy that finds the optimal action selection *policy* for any Markov Decision Process

* It revolves around the notion of updating Q values which denotes value of doing action a in state s. The value update rule is the core of the Q-learning algorithm.

![Q learning algorithm](https://wikimedia.org/api/rest_v1/media/math/render/svg/47fa1e5cf8cf75996a777c11c7b9445dc96d4637)

### Training Strategies

Background

* Learning from batches of consecutive samples is bad.
    * Samples are correlated => Inefficient learning
    * Can lead to bad feedback loops (bad Q-value leads to bad actions and maybe leads to bad states and so on)

#### Experience Replay

## Policy Gradients

Definitions

* Class of parametrized policies: $\prod = \{\pi_\theta \theta \in \mathbb{R}^m \}$
* Value for each policy: $J(\theta) = \mathbb{E} \begin{bmatrix}\displaystyle \sum_{t\geq 0} \gamma^t r_t | \pi_\theta \end{bmatrix}$
* Optimal policy: $\theta^* = \arg \max_\theta J(\theta)$

To find the optimal policy: **Gradient ascent on policy parameters!**

### REINFORCE algorithm

When sampling a trajectory $\tau$, we can estimate $J(\theta)$ with the gradient estimator:

$$
\nabla_\theta J(\theta) \approx \sum_{t\geq 0} r(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

Intuiiton

* If reward of trajectory $r(\tau)$ is high, push up the probabilities of the actions seen
* If reward of trajectory $r(\tau)$ is low, push down the probabilities of the actions seen

**Variance Reduction** (typically used in Vanilla REINFORCE)

* First idea: Push up probabilities of an aciton seen, only by the cumulative future reward from that state
* Second idea: Use discount factor $\gamma$ to ignore delayed effects

Baseline function

* Third idea: Introduce a baseline function dependent on the state. (process the raw value of a trajectory)

## Actor-Critic Algorithm

Combine **Policy Gradients** and **Q-learning** by training both an **actor** (the policy ($\theta$)) and a **critic** (the Q-function ($\phi$))

$$
A^\pi (s, a) = Q^\pi (s, a) - V^\pi (s)
$$

## Deep Q-network (DQN)

Deep version of Q-learning

## DDPG(Deep Deterministic Policy Gradient)

Actor-Critic + DQN

## Summary

| Policy Gradients                                                          | Q-Learning                                                                                                        |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| very general but suffer from high variance so requires a lot of samples   | does not always work but when it works, usually more sample-efficient                                             |
| Challenge: sample-efficiency                                              | Challenge: exploration                                                                                            |
| Grarantees: Converges to a local minima of $J(\theta)$, often good enough | Guarantees: Zero guarantees since you are approximating Bellman equation with a complicated funciton approximator |

| Learning \ Sampling | $\epsilon$ | greedy | random       |
| ------------------- | ---------- | ------ | ------------ |
| $\epsilon$          | Sarsa      | ?      | ?            |
| (Max) greedy        | Q-learning | ?      | (off-policy) |
| random              | ?          | ?      | (on-policy)  |

## Resources

### Book

Reinforcement Learning: An Introduction - [pdf with outlines](http://incompleteideas.net/book/bookdraft2017nov5.pdf) - ([old online version](http://www.incompleteideas.net/book/ebook/the-book.html))

* Introduction
* I. Tabular Solution Methods
  * Ch2 Multi-armed Bandits 多臂老虎機
    * Ch2.2 Action-value Methods
    * Ch2.7 Upper-Confidence-Bound Action Selection
    * Ch2.8 Gradient Bandit Algorithms
  * Ch3 Finite Markov Decision Processes
    * Ch3.2 Goals and Rewards
    * Ch3.3 Returns and Episodes
    * Ch3.5 Pilicies and Value Functions
    * Ch3.6 Optimal Policies and Optimal Value Functions

Others' Solution

* [ShangtongZhang/reinforcement-learning-an-introduction: Python Implementation of Reinforcement Learning: An Introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [matteocasolari/reinforcement-learning-an-introduction-solutions: Implementations for solutions to programming exercises of Reinforcement Learning: An Introduction, Second Edition (Sutton & Barto)](https://github.com/matteocasolari/reinforcement-learning-an-introduction-solutions)

### Tutorial

* [**TensorFlow and deep reinforcement learning, without a PhD (Google I/O '18)**](https://youtu.be/t1A3NTttvBA)
    * [code](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong)
    * [slides](https://docs.google.com/presentation/d/1qLVvgKxZlM6_oOZ4-ZoOAB0wTh2IdhbFvuBhsMvmK9I/pub)
* [**Stanford - Deep Reinforcement Learning**](https://youtu.be/lvoHnicueoE)
    * [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf)
* [**莫煩 - Reinforcement Lerning**](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)
    * [github - Reinforcement Learning Methods and Tutorials](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
    * [English Youbube playlist](https://www.youtube.com/playlist?list=PLXO45tsB95cIplu-fLMpUEEZTwrDNh6Ba)
    * [Youtube playlist - 強化學習 Reinforcement Learning Python 教學 教程](https://www.youtube.com/playlist?list=PLXO45tsB95cJYKCSATwh1M4n8cUnUv6lT)
        * [#4.1 DQN 算法更新 using Tensorflow (強化學習 Reinforcement Learning 教學)](https://www.youtube.com/watch?v=llWPgtGi1O0)
        * [#5.1 Policy Gradients 算法更新 (強化學習 Reinforcement Learning 教學)](https://www.youtube.com/watch?v=A54GU_WqLmQ)
        * [#5.2 Policy Gradients 思維決策 (強化學習 Reinforcement Learning 教學)](https://youtu.be/DwrGHh9Nkvg)
            * [code](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py)
    * [Youtube playlist - 秒懂強化學習 Reinforcement Learning](https://www.youtube.com/playlist?list=PLXO45tsB95cK7G-raBeTVjAoZHtJpiKh3)
        * [什麼是 Q Learning (Reinforcement Learning 強化學習)](https://www.youtube.com/watch?v=HTZ5xn12AL4)

Q Learing

* [Siraj Raval - How to use Q Learning in Video Games Easily](https://youtu.be/A5eihauRQvo)
* [Siraj Raval - Deep Q Learning for Video Games](https://youtu.be/79pmNdyxEGo)

### MOOC

* [Stanford CS234](http://web.stanford.edu/class/cs234/index.html)

### Article

* [**A Beginner's Guide to Deep Reinforcement Learning**](https://skymind.ai/wiki/deep-reinforcement-learning) - Very good explain
* [**A (Long) Peek into Reinforcement Learning**](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
* [Reinforcement Learning — Policy Gradient Explained](https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146)
* [DeepMind - Human-level control through Deep Reinforcement Learning](https://deepmind.com/research/dqn/)

### Github

* [rlcode/reinforcement-learning](https://github.com/rlcode/reinforcement-learning) - Minimal and Clean Reinforcement Learning Examples
* [Deep Q Learning in Tensorflow for ATARI.](https://github.com/mrkulk/deepQN_tensorflow)

Good Example

* [PacmanDQN](https://github.com/tychovdo/PacmanDQN)
    * [UC Berkeley CS188 The Pac-Man Projects](http://ai.berkeley.edu/project_overview.html)

### Wikipedia

* [Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
    * [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process)
    * [Q-learning](https://en.wikipedia.org/wiki/Q-learning)
