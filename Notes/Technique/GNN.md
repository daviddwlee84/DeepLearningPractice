# Graph Neural Network

## Concept

### Graph Representations

#### Algebraic Representation

* $A/W$ Adjacency Matrix
  * $a_{ii} = 0$ $\forall i$ if the graphs has no self-loops
  * $a_{ij} = a_{ji}$ if the graph is undirected
* $D$ Degree Matrix
  * $D = \operatorname{diag}(A1)$
  * $\operatorname{diag}(x)$ is a diagonal matrix with the entries of $x$ along the diagonal
* Laplacian Matrix: used to develop the concepts of graph frequency
  * Combinatorial Graph Laplacian Matrix
    * $L = D - A$
  * Random Walk Graph Laplacian Matrix
    * $\mathcal{T} = I - D^{-1}A$
    * all nodes have degree 1
  * Symmetric Normalized Graph Laplacian Matrix
    * $\mathcal{L} = D^{-1/2}LD^{-1/2} = I -  D^{-1/2}AD^{-1/2}$
    * $(\mathcal{L})_{ij} = -a_{i,j}\frac{1}{\sqrt{d_i}\sqrt{d_j}}$, $i \neq j$
    * all nodes have degree 1

### Graph Filter

#### Vertex Domain

$$
y = Tx
$$

* Linear Graph Filter $T$
  * Simple 1-hop filters:
    * $T$ is chosen to be $A$
      * 1-hop operator
      * i.e. the sum of neighboring nodes
            $$y(i) = \sum_{j\in N_i} a_{ij}x(j)$$
      * can be used to infer unknown neighbor node using the average
      * prediction error
        * $y^{n(i)} = x(i) - y(i)$
    * $D^{-1}A$
    * $\mathcal{T} = I - D^{-1}A$
    * $L$

#### Spatial domain

## Resources

* [Must-read papers on GNN](https://github.com/thunlp/GNNPapers)
* [[1312.6203] Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)
